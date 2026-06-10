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


REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Discord ---------------------------------------------------------------
TOKEN = os.environ.get("FRIENDBOT_TOKEN", "")
COMMAND_PREFIX = os.environ.get("FRIENDBOT_PREFIX", "!")

# Which application to run: "image" (diffusion via flux/generate.py) or "chat"
# (local LLM persona chat). The two models don't fit in VRAM together, so this
# is an either/or switch. Also overridable as ``python -m friendbot2 <mode>``.
MODE = os.environ.get("FRIENDBOT_MODE", "image").strip().lower()
if MODE not in ("image", "chat"):
    raise ValueError(f"FRIENDBOT_MODE must be 'image' or 'chat', got {MODE!r}")

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

# --- Chat mode (local LLM) ---------------------------------------------------
# Base model for persona chat. Llama-3.2-3B (base, not Instruct) is the default:
# completion-style transcript continuation suits a base model, it QLoRA-trains
# comfortably on a 12 GB RTX 3080, and inference in NF4 takes ~2.5 GB VRAM.
# Note: Meta's repos are gated — accept the license on HF and `hf auth login`.
LLM_BASE_MODEL = os.environ.get("FRIENDBOT_LLM_BASE", "meta-llama/Llama-3.2-3B")

# LoRA adapter produced by tools/train_lora.py. If the path doesn't exist the
# bot runs on the raw base model (useful for testing the plumbing).
LLM_ADAPTER_PATH = Path(
    os.environ.get(
        "FRIENDBOT_LLM_ADAPTER", REPO_ROOT / "models" / "adapters" / "friendbot-lora"
    )
).expanduser()

# Load the LLM in 4-bit NF4 (recommended on GPU; disable for CPU-only testing).
LLM_4BIT = _bool(os.environ.get("FRIENDBOT_LLM_4BIT"), default=True)

# Sampling / context knobs.
LLM_MAX_NEW_TOKENS = int(os.environ.get("FRIENDBOT_LLM_MAX_NEW_TOKENS", "120"))
LLM_TEMPERATURE = float(os.environ.get("FRIENDBOT_LLM_TEMPERATURE", "0.9"))
LLM_TOP_P = float(os.environ.get("FRIENDBOT_LLM_TOP_P", "0.95"))
LLM_CONTEXT_TOKENS = int(os.environ.get("FRIENDBOT_LLM_CONTEXT_TOKENS", "1536"))

# How many recent channel messages to keep as the chat transcript context.
CHAT_CONTEXT_MESSAGES = int(os.environ.get("FRIENDBOT_CHAT_CONTEXT", "25"))

# Which user the bot speaks as. Empty -> most prolific user in personas.json
# (written by tools/build_dataset.py), falling back to the bot's own name.
PERSONA = os.environ.get("FRIENDBOT_PERSONA", "").strip()
PERSONAS_FILE = REPO_ROOT / "data" / "sft" / "personas.json"

# Only offer personas for users with at least this many messages in the
# training data; below that the impression is too thin to be fun.
PERSONA_MIN_MESSAGES = int(os.environ.get("FRIENDBOT_PERSONA_MIN_MESSAGES", "1000"))

# /persona <name> still allows any name, but warns below this many messages.
PERSONA_WARN_MESSAGES = int(os.environ.get("FRIENDBOT_PERSONA_WARN_MESSAGES", "10000"))
