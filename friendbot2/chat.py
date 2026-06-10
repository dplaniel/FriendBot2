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

import discord
from discord import app_commands
from discord.ext import commands

from . import config
from .llm_backend import LLMBackend

log = logging.getLogger(__name__)

_CUSTOM_EMOJI = re.compile(r"<a?(:\w+:)\d+>")
# The "[Persona] " tag we prepend to our own replies (see _respond).
_PERSONA_TAG = re.compile(r"^\[[^\]\n]+\]\s*")


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
        self.persona: str | None = config.PERSONA or None
        self.known_personas: list[str] = []
        self._loader: asyncio.Task | None = None

    # -- lifecycle ----------------------------------------------------------
    async def cog_load(self) -> None:
        self.known_personas = self._read_personas_file()
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
    def _read_personas_file() -> list[str]:
        """Top users by message count, written by tools/build_dataset.py.

        Users below PERSONA_MIN_MESSAGES are left out: there isn't enough of
        them in the training data for the impression to land.
        """
        try:
            entries = json.loads(config.PERSONAS_FILE.read_text())
            return [
                e["name"]
                for e in entries
                if e.get("messages", 0) >= config.PERSONA_MIN_MESSAGES
            ]
        except (OSError, ValueError, KeyError):
            return []

    # -- transcript bookkeeping ----------------------------------------------
    def _persona_label(self) -> str:
        if self.persona:
            return self.persona
        return self.bot.user.display_name if self.bot.user else "FriendBot"

    def _line_for(self, message: discord.Message) -> str | None:
        """Render a message as a transcript line, or None if it shouldn't appear."""
        if message.author.bot and message.author != self.bot.user:
            return None
        text = _normalize(message.clean_content)
        if message.author == self.bot.user:
            # Strip the "[Persona] " tag off our own replies so the transcript
            # stays in the plain "Name: message" format the model was trained on.
            text = _PERSONA_TAG.sub("", text)
        # Drop the bot's own @mention so the model doesn't learn to echo pings.
        me = message.guild.me if message.guild else self.bot.user
        if me is not None:
            text = text.replace(f"@{me.display_name}", "").strip()
        if not text or text.startswith(config.COMMAND_PREFIX):
            return None
        name = (
            self._persona_label()
            if message.author == self.bot.user
            else message.author.display_name
        )
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
        persona = self._persona_label()
        try:
            async with message.channel.typing():
                reply = await self.backend.chat(list(dq), persona)
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
        if self.known_personas and name not in self.known_personas:
            await ctx.reply(
                f"I wasn't trained on much from **{name}** — switching anyway, "
                f"but `/personas` lists who I actually know."
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
        await ctx.reply(f"People I can do an impression of:\n{listing}")
