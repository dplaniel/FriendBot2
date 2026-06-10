"""
Configuration for FriendBot2, loaded from environment variables.

Copy ``.env.example`` to ``.env`` and fill in the values, or export the
variables in your shell. Nothing secret is committed to the repository.
"""

import os
from pathlib import Path

try:
    # Optional: load a local .env file if python-dotenv is installed.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a convenience, not required
    pass


def _int_set(raw: str) -> set[int]:
    """Parse a comma/space separated list of ints into a set."""
    return {int(tok) for tok in raw.replace(",", " ").split() if tok.strip()}


def _bool(raw: str, default: bool = False) -> bool:
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Discord ---------------------------------------------------------------
TOKEN = os.environ.get("FRIENDBOT_TOKEN", "")
COMMAND_PREFIX = os.environ.get("FRIENDBOT_PREFIX", "!")

# Optional guild id for instant slash-command syncing during development.
# Global syncs can take up to an hour to propagate; a guild sync is immediate.
GUILD_ID = int(os.environ["FRIENDBOT_GUILD_ID"]) if os.environ.get("FRIENDBOT_GUILD_ID") else None

# Channels the bot will accept prompts in. Empty == every channel.
ALLOWED_CHANNEL_IDS = _int_set(os.environ.get("FRIENDBOT_CHANNELS", ""))

# User ids with administrative privileges (reserved for future commands).
PRIVILEGED_USERS = _int_set(os.environ.get("FRIENDBOT_PRIVILEGED_USERS", ""))

# --- Image generation ------------------------------------------------------
# Path to the local flux repo that holds generate.py and the (pre-quantized)
# models/ directory. Defaults to a sibling ``flux`` checkout next to this repo.
FLUX_REPO_PATH = Path(
    os.environ.get("FLUX_REPO_PATH", Path(__file__).resolve().parent.parent.parent / "flux")
).expanduser()

# Which model generate.py loads: "sd35" (Stable Diffusion 3.5 Large Turbo) or
# "flux" (FLUX.1-schnell). Mirrors generate.py's --model option.
MODEL = os.environ.get("FRIENDBOT_MODEL", "sd35").strip().lower()
if MODEL not in ("flux", "sd35"):
    raise ValueError(f"FRIENDBOT_MODEL must be 'flux' or 'sd35', got {MODEL!r}")

# Queue / fairness knobs.
MAX_QUEUE_SIZE = int(os.environ.get("FRIENDBOT_MAX_QUEUE", "16"))
USER_PROMPT_CAP = int(os.environ.get("FRIENDBOT_USER_CAP", "5"))

# Delete generated images from disk after they are uploaded.
DELETE_IMAGES = _bool(os.environ.get("FRIENDBOT_DELETE_IMAGES"), default=False)

# Reply with snarky messages when the queue or a user hits their cap.
SNARKY = _bool(os.environ.get("FRIENDBOT_SNARKY"), default=True)
