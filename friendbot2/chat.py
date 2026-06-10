"""
Chat cog: the bot lurks in allowed channels, keeps a rolling transcript of
recent messages, and replies in the style of a chosen server persona whenever it
is @mentioned or someone replies to one of its messages.

Transcript lines use the same ``Name: message`` format the fine-tuning data is
built in (tools/build_dataset.py), so the LoRA-tuned model is always continuing
text shaped exactly like its training distribution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import deque
from random import choice

import discord
from discord import app_commands
from discord.ext import commands

from . import config
from .llm_backend import LLMBackend

log = logging.getLogger(__name__)

_CUSTOM_EMOJI = re.compile(r"<a?(:\w+:)\d+>")
# Bare :name: shortcodes the model emits (training data stores emoji this way).
_EMOJI_SHORTCODE = re.compile(r":(\w+):")
# The "[Persona] " tag we prepend to our own replies (see _respond).
_PERSONA_TAG = re.compile(r"^\[([^\]\n]+)\]\s*")

# Pseudo-persona: pick a random listed persona for each reply.
RANDOM_PERSONA = "randomized"


def _normalize(text: str) -> str:
    """Match build_dataset.py's text normalization: one line, plain emoji names."""
    text = _CUSTOM_EMOJI.sub(r"\1", text)
    return " ".join(text.split())


class ChatCog(commands.Cog):
    """Persona chat backed by a local fine-tuned LLM."""

    def __init__(self, bot: commands.Bot, backend: LLMBackend):
        self.bot = bot
        self.backend = backend
        # channel id -> deque of "Name: message" lines
        self.transcripts: dict[int, deque[str]] = {}
        configured = config.PERSONA or None
        if configured and configured.lower() == RANDOM_PERSONA:
            configured = RANDOM_PERSONA
        self.persona: str | None = configured
        self.persona_counts: dict[str, int] = {}
        self.known_personas: list[str] = []
        self._loader: asyncio.Task | None = None

    # -- lifecycle ----------------------------------------------------------
    async def cog_load(self) -> None:
        self.persona_counts = self._read_personas_file()
        # Listing threshold: below this there isn't enough of a user in the
        # training data for the impression to land.
        self.known_personas = [
            name
            for name, count in self.persona_counts.items()
            if count >= config.PERSONA_MIN_MESSAGES
        ]
        if self.persona is None and self.known_personas:
            self.persona = self.known_personas[0]
            log.info("No FRIENDBOT_PERSONA set; defaulting to %r", self.persona)
        self._loader = asyncio.create_task(self._load_backend())

    async def cog_unload(self) -> None:
        if self._loader is not None:
            self._loader.cancel()
        self.backend.shutdown()

    async def _load_backend(self) -> None:
        try:
            log.info("Loading LLM %s ...", self.backend.base_model)
            await self.backend.load()
            log.info("LLM ready.")
        except Exception:  # noqa: BLE001 - surface any load failure in the logs
            log.exception("Failed to load LLM; chat disabled.")

    @staticmethod
    def _read_personas_file() -> dict[str, int]:
        """User -> message count, from tools/build_dataset.py output.

        Ordered most-active first, matching the file. Unfiltered: callers
        apply the listing/warning thresholds.
        """
        try:
            entries = json.loads(config.PERSONAS_FILE.read_text())
            return {e["name"]: e.get("messages", 0) for e in entries}
        except (OSError, ValueError, KeyError):
            return {}

    # -- transcript bookkeeping ----------------------------------------------
    def _persona_label(self) -> str:
        if self.persona:
            return self.persona
        return self.bot.user.display_name if self.bot.user else "FriendBot"

    def _resolve_reply_persona(self) -> str:
        """The persona to use for one reply (rolls the dice in randomized mode)."""
        if self.persona == RANDOM_PERSONA and self.known_personas:
            return choice(self.known_personas)
        if self.persona == RANDOM_PERSONA:
            return self.bot.user.display_name if self.bot.user else "FriendBot"
        return self._persona_label()

    def _render_emoji(self, text: str, guild: discord.Guild | None) -> str:
        """Turn :name: shortcodes into real custom emoji where one exists.

        The model emits emoji in the bare :name: form it was trained on;
        Discord only renders the full <:name:id> form from bots. Names that
        don't match any visible emoji (hallucinated, deleted, or plain-unicode
        shortcodes) are left as text, which is how they read in training too.
        """

        def replace(match: re.Match) -> str:
            name = match.group(1)
            pools = (guild.emojis if guild else ()), self.bot.emojis
            for pool in pools:
                emoji = discord.utils.get(pool, name=name)
                if emoji is None:  # tolerate the model changing case
                    emoji = discord.utils.find(
                        lambda e: e.name.lower() == name.lower(), pool
                    )
                if emoji is not None:
                    return str(emoji)
            return match.group(0)

        return _EMOJI_SHORTCODE.sub(replace, text)

    def _line_for(self, message: discord.Message) -> str | None:
        """Render a message as a transcript line, or None if it shouldn't appear."""
        if message.author.bot and message.author != self.bot.user:
            return None
        text = _normalize(message.clean_content)
        name = message.author.display_name
        if message.author == self.bot.user:
            # Our own replies carry a "[Persona] " tag: recover who "spoke"
            # from it (the current persona may differ, e.g. randomized mode)
            # and strip it so the transcript stays in the plain
            # "Name: message" format the model was trained on.
            tag = _PERSONA_TAG.match(text)
            name = tag.group(1) if tag else self._persona_label()
            text = _PERSONA_TAG.sub("", text)
        # Drop the bot's own @mention so the model doesn't learn to echo pings.
        me = message.guild.me if message.guild else self.bot.user
        if me is not None:
            text = text.replace(f"@{me.display_name}", "").strip()
        if not text or text.startswith(config.COMMAND_PREFIX):
            return None
        return f"{name}: {text}"

    async def _transcript(
        self, channel: discord.abc.Messageable, before: discord.Message | None = None
    ) -> deque[str]:
        """Get the channel transcript, seeding it from history on first use."""
        dq = self.transcripts.get(channel.id)
        if dq is None:
            dq = deque(maxlen=config.CHAT_CONTEXT_MESSAGES)
            self.transcripts[channel.id] = dq
            try:
                seed = [
                    m
                    async for m in channel.history(
                        limit=config.CHAT_CONTEXT_MESSAGES, before=before
                    )
                ]
            except discord.DiscordException:
                seed = []
            for msg in reversed(seed):
                line = self._line_for(msg)
                if line:
                    dq.append(line)
        return dq

    # -- message handling -----------------------------------------------------
    def _is_triggered(self, message: discord.Message) -> bool:
        if self.bot.user is None:
            return False
        if self.bot.user in message.mentions:
            return True
        ref = message.reference.resolved if message.reference else None
        return getattr(getattr(ref, "author", None), "id", None) == self.bot.user.id

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if config.ALLOWED_CHANNEL_IDS and message.channel.id not in config.ALLOWED_CHANNEL_IDS:
            return

        if message.author == self.bot.user:
            # Record our own replies so the transcript stays coherent.
            dq = self.transcripts.get(message.channel.id)
            line = self._line_for(message)
            if dq is not None and line:
                dq.append(line)
            return
        if message.author.bot:
            return

        dq = await self._transcript(message.channel, before=message)
        line = self._line_for(message)
        if line:
            dq.append(line)

        if self._is_triggered(message):
            await self._respond(message, dq)

    async def _respond(self, message: discord.Message, dq: deque[str]) -> None:
        if not self.backend.ready:
            await message.reply(
                "I'm still waking up — give me a minute and ping me again."
            )
            return
        persona = self._resolve_reply_persona()
        try:
            async with message.channel.typing():
                reply = await self.backend.chat(list(dq), persona)
            reply = self._render_emoji(reply, message.guild)
            await message.reply(f"[{persona}] {reply}")
        except Exception:  # noqa: BLE001
            log.exception("Chat generation failed.")
            try:
                await message.reply("I think something went wrong, sorry.")
            except discord.DiscordException:
                pass

    # -- commands -------------------------------------------------------------
    @commands.hybrid_command(
        name="persona", description="Show or set which user I chat as."
    )
    @app_commands.describe(name="Display name of the persona to mimic.")
    async def persona_cmd(self, ctx: commands.Context, *, name: str | None = None) -> None:
        if name is None:
            await ctx.reply(f"I'm currently chatting as **{self._persona_label()}**.")
            return
        name = name.strip()
        if name.lower() == RANDOM_PERSONA:
            self.persona = RANDOM_PERSONA
            await ctx.reply("Okay, I'll do a different impression for each message.")
            return
        # Case-insensitive match, canonicalized to the stored capitalization:
        # the name becomes the model's "Name:" conditioning prefix, where exact
        # case is what it saw in training.
        canonical = next(
            (known for known in self.persona_counts if known.lower() == name.lower()),
            None,
        )
        if canonical is not None:
            name = canonical
        if (
            self.persona_counts
            and self.persona_counts.get(name, 0) < config.PERSONA_WARN_MESSAGES
        ):
            await ctx.reply(
                f"I wasn't trained on much from **{name}** — switching anyway, "
                f"but `/personas` lists who I do better impressions of."
            )
        else:
            await ctx.reply(f"Okay, I'm **{name}** now.")
        self.persona = name

    @commands.hybrid_command(
        name="personas", description="List the personas I know how to mimic."
    )
    async def personas_cmd(self, ctx: commands.Context) -> None:
        if not self.known_personas:
            await ctx.reply(
                "I don't know anyone well enough to imitate yet — run "
                "tools/build_dataset.py, or lower FRIENDBOT_PERSONA_MIN_MESSAGES."
            )
            return
        listing = "\n".join(f"- {name}" for name in self.known_personas[:15])
        listing += f"\n- {RANDOM_PERSONA} (someone different each message)"
        await ctx.reply(f"People I can do an impression of:\n{listing}")
