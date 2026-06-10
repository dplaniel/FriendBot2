#!/usr/bin/env python3
"""Quick-and-dirty demo: run the epoch-1 LoRA checkpoint through the bot's own
inference path and show the top-10 personas continuing a few conversations.

Usage:  python demo_personas.py [checkpoint_dir]

Scratch script — not part of the repo, delete freely.
"""

import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from friendbot2.llm_backend import LLMBackend  # noqa: E402

ADAPTER_ROOT = REPO / "models" / "adapters" / "friendbot-lora"
N_PERSONAS = 10


def pick_checkpoint() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    cps = sorted(
        ADAPTER_ROOT.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])
    )
    if not cps:
        sys.exit(f"No checkpoint-* under {ADAPTER_ROOT}")
    return cps[0]  # lowest step == end of epoch 1


def contexts() -> list[tuple[str, list[str]]]:
    """Two real val-set conversation openings + one synthetic scenario."""
    out = []
    val = (REPO / "data" / "sft" / "val.jsonl").read_text().splitlines()
    # Grab two reasonably long chunks from different parts of the val set.
    picks = [val[3], val[len(val) // 2]]
    for i, raw in enumerate(picks, 1):
        lines = json.loads(raw)["text"].split("\n")[:8]
        out.append((f"real conversation #{i} (val set)", lines))
    out.append((
        "synthetic scenario",
        [
            "Alice: ok important question for the group",
            "Alice: what should we play tonight",
            "Bob: i'm down for anything that isn't another 4 hour session of losing",
        ],
    ))
    return out


async def main() -> None:
    ckpt = pick_checkpoint()
    base = json.loads((ckpt / "adapter_config.json").read_text())[
        "base_model_name_or_path"
    ]
    personas = [
        e["name"]
        for e in json.loads((REPO / "data" / "sft" / "personas.json").read_text())
    ][:N_PERSONAS]

    print(f"checkpoint: {ckpt.name}   base: {base}")
    print(f"personas:   {', '.join(personas)}\n")

    backend = LLMBackend(base, adapter_path=ckpt)
    await backend.load()

    for title, lines in contexts():
        print("=" * 72)
        print(f"--- {title} ---")
        for line in lines:
            print(f"    {line[:100]}")
        print("-" * 72)
        for persona in personas:
            reply = await backend.chat(lines, persona)
            print(f"  [{persona}] {reply}")
        print()

    backend.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
