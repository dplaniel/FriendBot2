"""Bot construction: intents, cog registration, and slash-command syncing."""

from __future__ import annotations

import logging

import discord
from discord.ext import commands

from . import config
from .flux_backend import FluxBackend
from .image_generation import ImageGenerationCog

log = logging.getLogger(__name__)


class FriendBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        # Required for prefix (``!artistic``) commands; slash commands work without
        # it. message_content is a privileged intent — enable it for the bot in the
        # Discord developer portal.
        intents.message_content = True
        super().__init__(command_prefix=config.COMMAND_PREFIX, intents=intents)
        self.backend = FluxBackend(config.FLUX_REPO_PATH, model=config.MODEL)

    async def setup_hook(self) -> None:
        await self.add_cog(ImageGenerationCog(self, self.backend))

        # Register slash commands. A guild-scoped sync is instant and ideal for
        # development; a global sync (no guild) can take up to an hour to appear.
        if config.GUILD_ID:
            guild = discord.Object(id=config.GUILD_ID)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            log.info("Synced %d slash command(s) to guild %s", len(synced), config.GUILD_ID)
        else:
            synced = await self.tree.sync()
            log.info("Synced %d global slash command(s)", len(synced))

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id: %s)", self.user, getattr(self.user, "id", "?"))
