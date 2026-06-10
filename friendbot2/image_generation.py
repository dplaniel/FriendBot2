"""
Image-generation cog: turns ``/artistic <prompt>`` (or ``!artistic ...``) into a
FLUX.1-schnell image.

Prompts are placed on a bounded queue and processed one at a time by a single
background worker, so only one generation ever touches the GPU at once. Per-user
caps and a queue-full check keep any one person from monopolizing the bot.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from random import choice

import discord
from discord import app_commands
from discord.ext import commands

from . import config, dialog
from .flux_backend import FluxBackend

log = logging.getLogger(__name__)


class ImageGenerationCog(commands.Cog):
    """FLUX.1-schnell image generation with a fair, bounded prompt queue."""

    def __init__(self, bot: commands.Bot, backend: FluxBackend):
        self.bot = bot
        self.backend = backend
        self.queue: asyncio.Queue = asyncio.Queue(config.MAX_QUEUE_SIZE)
        # user id -> number of their prompts currently queued/in flight
        self.user_counts: defaultdict[int, int] = defaultdict(int)
        self.slow_cap = 0.50  # warn once the queue passes this fraction full
        self._worker: asyncio.Task | None = None
        self._loader: asyncio.Task | None = None

    # -- lifecycle ----------------------------------------------------------
    async def cog_load(self) -> None:
        # Load the model in the background so the bot can come online immediately;
        # commands report "not ready" until the pipeline finishes loading.
        self._loader = asyncio.create_task(self._load_backend())
        self._worker = asyncio.create_task(self._process_queue())

    async def cog_unload(self) -> None:
        for task in (self._worker, self._loader):
            if task is not None:
                task.cancel()
        self.backend.shutdown()

    async def _load_backend(self) -> None:
        try:
            log.info(
                "Loading %s pipeline from %s ...",
                self.backend.model,
                self.backend.repo_path,
            )
            await self.backend.load()
            log.info("%s pipeline ready.", self.backend.model)
        except Exception:  # noqa: BLE001 - surface any load failure in the logs
            log.exception("Failed to load image pipeline; image generation disabled.")

    # -- queue worker -------------------------------------------------------
    async def _process_queue(self) -> None:
        while True:
            prompt, author, channel = await self.queue.get()
            try:
                self.user_counts[author.id] -= 1
                if self.user_counts[author.id] <= 0:
                    self.user_counts.pop(author.id, None)
                async with channel.typing():
                    path = await self.backend.generate(prompt)
                await channel.send(
                    f"{author.mention} How's this?", file=discord.File(path)
                )
                if config.DELETE_IMAGES:
                    try:
                        os.remove(path)
                    except OSError:
                        log.warning("Could not delete %s", path)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("Generation failed for prompt: %r", prompt)
                try:
                    await channel.send(
                        f"{author.mention} I think something went wrong, sorry."
                    )
                except discord.DiscordException:
                    pass
            finally:
                self.queue.task_done()

    # -- command ------------------------------------------------------------
    @commands.hybrid_command(
        name="artistic",
        aliases=["prompt", "txt2img", "generate", "gen", "ig"],
        description="Generate an image from a text prompt using FLUX.1-schnell.",
    )
    @app_commands.describe(prompt="Describe the image you want me to make.")
    async def artistic(self, ctx: commands.Context, *, prompt: str) -> None:
        # Channel allow-list (empty == everywhere).
        if config.ALLOWED_CHANNEL_IDS and ctx.channel.id not in config.ALLOWED_CHANNEL_IDS:
            await ctx.reply("I don't make art in this channel.", ephemeral=True)
            return

        if not self.backend.ready:
            await ctx.reply(
                "My art supplies aren't ready yet — give me a moment and try again.",
                ephemeral=True,
            )
            return

        prompt = prompt.strip(" \t\n'\"")
        if not prompt:
            await ctx.reply("Give me something to draw.", ephemeral=True)
            return

        if self.queue.full():
            await ctx.reply(dialog.channel_queue_full, ephemeral=True)
            return

        # Per-user cap.
        user_ct = self.user_counts[ctx.author.id]
        if user_ct >= config.USER_PROMPT_CAP:
            if config.SNARKY:
                reply = choice(dialog.snarky_usercaps).format(
                    member=ctx.author.display_name,
                    plus_delimited_prompt="+".join(prompt.split()),
                )
            else:
                reply = dialog.user_at_cap.format(
                    member=ctx.author.display_name, usercap=config.USER_PROMPT_CAP
                )
            await ctx.reply(reply, ephemeral=True)
            return

        # Enqueue. No await between full() check and put_nowait, so this is safe.
        self.user_counts[ctx.author.id] += 1
        self.queue.put_nowait((prompt, ctx.author, ctx.channel))
        await ctx.reply(
            f"Okay, I have {self.queue.qsize()} prompt(s) in the queue, "
            f"{self.user_counts[ctx.author.id]} from you."
        )

        # Slowdown warning once the queue gets busy.
        if self.queue.qsize() >= int(self.slow_cap * self.queue.maxsize):
            if config.SNARKY:
                warning = choice(dialog.snarky_channel_slowdowns).format(
                    n_slow_cap=int(self.slow_cap * self.queue.maxsize),
                    cmd_prefix=config.COMMAND_PREFIX,
                )
            else:
                warning = dialog.channel_slowdown.format(
                    slow_cap=int(100 * self.slow_cap)
                )
            await ctx.channel.send(warning)
