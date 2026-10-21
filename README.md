# FriendBot2

A Discord bot with two GPU personalities, one at a time (they don't fit in a
12 GB RTX 3080 together):

- **Image mode** (default) — generates images from text prompts with a local
  diffusion pipeline: [Stable Diffusion 3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)
  by default, or [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell).
- **Chat mode** — chats in the style of the people on your server, using a
  small local LLM QLoRA-fine-tuned on the server's own message history.

```bash
python -m friendbot2          # image mode
python -m friendbot2 chat     # chat mode
```

This is a ground-up rewrite of the original FriendBot:

- **Modern Discord API** — discord.py 2.x with slash commands and classic
  prefix commands via hybrid commands, gateway intents, an async `setup_hook`,
  and `await bot.add_cog(...)`.
- **Local diffusion instead of the old Stable Diffusion subprocess** — image
  generation calls the local [`flux/generate.py`](../flux/generate.py) script's
  `load_pipeline` / `generate` functions directly, keeping the model resident in
  process for fast turnaround instead of shelling out and reloading weights per
  prompt. The model is selected with `FRIENDBOT_MODEL` (`sd35` default, or
  `flux`), the same choice as generate.py's `--model` flag.

## Image mode

`/artistic <prompt>` puts your prompt on a bounded queue. A single background
worker pulls prompts one at a time and runs the model on a dedicated thread (so
the event loop is never blocked and only one generation touches the GPU at once),
then replies with the finished PNG. Per-user caps and a queue-full check keep any
one person from hogging the bot.

The pipeline is loaded once in the background at startup; until it's ready the
bot stays online and `/artistic` politely reports that it's still warming up.

## Requirements

- Python 3.10+
- A CUDA GPU with the FLUX deps installed (see the [flux](../flux) repo)
- A Discord application + bot token

Because the bot runs FLUX **in process**, it needs the flux dependencies (torch,
diffusers, transformers, bitsandbytes, …) available in the same environment. The
simplest setup is to reuse the flux virtualenv:

```bash
# 1. Build the flux environment + (optionally) pre-quantize the models
cd ../flux
bash setup.sh
python quantize.py --model sd35   # optional but recommended: faster loads + generation
# (use plain `python quantize.py` instead if you set FRIENDBOT_MODEL=flux)

# 2. Add FriendBot2's own deps into that same venv
source .venv/bin/activate
cd ../FriendBot2
pip install -r requirements.txt
```

## Chat mode

Chat mode runs a small local LLM fine-tuned on your server's own chat history so
it can talk like the people in it. @Mention the bot (or reply to it) and it
answers in character; `/persona <name>` switches whose style it mimics
(`/persona randomized` picks a different listed persona for every message) and
`/personas` lists who it knows.

On top of the persona you can layer a **role**: `/role lawyer` keeps the same
person's voice but has them answering as a lawyer, `/role none` clears it, and
`/roles` lists what's available. Roles live in `roles.json` (or wherever
`FRIENDBOT_ROLES_FILE` points), so add your own:

```json
{
  "lawyer": {
    "description": "Answers everything like a (questionably qualified) lawyer.",
    "lines": [
      "{persona}: update: i passed the bar. i'm a lawyer now (not legal advice)",
      "Sam: my boss says i'm not allowed to talk about my salary with coworkers",
      "{persona}: that's straight up illegal in the US, the NLRA protects talking about pay"
    ]
  }
}
```

The model is a base model tuned on raw transcripts, so it doesn't take
instructions. A role works by placing those `lines` (with `{persona}` filled
in) at the top of the conversation. The lines are always kept, even when the
transcript has to be trimmed to fit the context. Tips for writing them:

- **Show, don't announce.** A short exchange where someone asks something and
  the persona answers in role does more than the persona saying "i'm a lawyer
  now". Base models copy what they're shown.
- **Use made-up names** for the other speakers, not real server members, so the
  examples don't read like something a real person said.
- **Write in the server's own chat style.** The model imitates lines that sound
  like its training data more closely.

### Model and runtime choice

- **Base model:** `meta-llama/Llama-3.2-3B` (the *base* checkpoint, not
  Instruct). Mimicry from raw chat logs is a text-continuation task, so the bot
  is trained and prompted completion-style on plain `Name: message` transcripts —
  a base model fits that naturally, with no chat-template gymnastics. 3B
  QLoRA-trains comfortably on a 12 GB RTX 3080 (~7 GB peak) and leaves headroom
  for a desktop. `meta-llama/Llama-3.1-8B` also works (set `--base` when
  training and `FRIENDBOT_LLM_BASE` when running; expect to drop
  `--batch-size` to 1–2). GPT-OSS was ruled out: its smallest release (20B)
  wants ~16 GB. Llama/Gemma repos are gated on Hugging Face — accept the
  license on the model page once and run `hf auth login`.
- **Runtime:** transformers + bitsandbytes NF4 + PEFT, in process, same pattern
  (and same venv) as the image backend: the model loads once onto a dedicated
  worker thread and stays resident (~2.5 GB VRAM for 3B). At Discord-chat
  message lengths this generates a reply in a couple of seconds; if you ever
  want more speed, merging the adapter and exporting to GGUF for llama.cpp is
  the upgrade path, at the cost of an extra conversion step in the pipeline.

### Fine-tuning pipeline

```bash
# 0. one-time: training deps + HF login for the gated base model
pip install -r requirements-train.txt
hf auth login

# 1. pull the server's message history -> data/raw/<guild>/<channel>.jsonl
#    (re-running resumes incrementally; needs the Message Content intent)
python tools/collect_history.py --guild <your_guild_id>

# 2. filter + chunk into transcript training samples -> data/sft/
python tools/build_dataset.py

# 3. QLoRA fine-tune -> models/adapters/friendbot-lora/
#    (stop the image bot first; training wants the GPU to itself)
python tools/train_lora.py

# 4. run it
python -m friendbot2 chat
```

Each tool is standalone with `--help`. The dataset format is deliberately
simple — one `{"text": "<transcript chunk>"}` per line — sessions are split on
4-hour silences, commands/bots/attachment-only messages are filtered out, and
`personas.json` records who appears often enough to imitate. `data/` and
`models/` are gitignored: the collected history is your friends' private chat,
so keep it on this machine and don't commit or upload it.

## Configuration

Copy `.env.example` to `.env` and fill it in (at minimum `FRIENDBOT_TOKEN`):

```bash
cp .env.example .env
$EDITOR .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `FRIENDBOT_TOKEN` | — | **Required.** Discord bot token. |
| `FRIENDBOT_MODE` | `image` | `image` or `chat` (CLI arg overrides). |
| `FLUX_REPO_PATH` | `../flux` | Path to the flux checkout with `generate.py` + `models/`. |
| `FRIENDBOT_MODEL` | `sd35` | Diffusion model: `sd35` or `flux`. |
| `FRIENDBOT_LLM_BASE` | `meta-llama/Llama-3.2-3B` | Chat base model. |
| `FRIENDBOT_LLM_ADAPTER` | `models/adapters/friendbot-lora` | LoRA adapter path. |
| `FRIENDBOT_PERSONA` | most active user | Whose style to mimic. |
| `FRIENDBOT_ROLE` | — | Role from the roles file to start in (e.g. `lawyer`). |
| `FRIENDBOT_ROLES_FILE` | `roles.json` | Role definitions (see Chat mode). |
| `FRIENDBOT_LLM_4BIT` | `true` | NF4-quantize the LLM (disable for CPU tests). |
| `FRIENDBOT_LLM_MAX_NEW_TOKENS` / `_TEMPERATURE` / `_TOP_P` / `_CONTEXT_TOKENS` | `120` / `0.9` / `0.95` / `1536` | Sampling and context budget. |
| `FRIENDBOT_LLM_REPETITION_PENALTY` | `1.05` | Repetition penalty; 1.0 disables. Penalizes prompt tokens too, so keep gentle. |
| `FRIENDBOT_LLM_MIN_REPLY_TOKENS` | `12` | Newline/EOS blocked for the first N tokens of each reply; 0 disables. |
| `FRIENDBOT_CHAT_CONTEXT` | `25` | Recent messages kept as transcript context. |
| `FRIENDBOT_PREFIX` | `!` | Prefix for classic text commands. |
| `FRIENDBOT_GUILD_ID` | — | Guild id for instant slash-command sync (dev). |
| `FRIENDBOT_CHANNELS` | (all) | Allowed channel ids (comma/space separated). |
| `FRIENDBOT_PRIVILEGED_USERS` | — | Admin user ids (reserved for future use). |
| `FRIENDBOT_MAX_QUEUE` | `16` | Max prompts in the queue. |
| `FRIENDBOT_USER_CAP` | `5` | Max simultaneous prompts per user. |
| `FRIENDBOT_DELETE_IMAGES` | `false` | Delete PNGs after upload. |
| `FRIENDBOT_SNARKY` | `true` | Snarky cap/slowdown replies. |

### Discord developer portal

- Invite the bot with the **applications.commands** and **bot** scopes.
- Under **Bot → Privileged Gateway Intents**, enable **Message Content Intent**
  (required for `!` prefix commands, chat-mode listening, and history
  collection; slash commands work without it).

## Running

```bash
# from inside the venv that has both discord.py and the flux deps
python -m friendbot2          # image mode
python -m friendbot2 chat     # chat mode
```

Then in Discord:

```
/artistic a photorealistic sunset over misty mountains
!gen a macro photo of a dragonfly        (image mode)

@FriendBot2 what do you think?           (chat mode)
/persona Alice                           (chat mode)
/role lawyer                             (chat mode)
```

(`prompt`, `txt2img`, `generate`, `gen`, and `ig` are aliases for `!artistic`.)

## Repo layout

```
friendbot2/            the bot
  __main__.py          entry point: python -m friendbot2 [image|chat]
  bot.py               FriendBot: intents, mode wiring, slash sync
  image_generation.py  image mode cog (prompt queue, caps)
  flux_backend.py      resident diffusion pipeline via ../flux/generate.py
  chat.py              chat mode cog (transcript context, /persona, /role)
  llm_backend.py       resident NF4 LLM + LoRA adapter
  config.py            env-based configuration
tools/                 offline fine-tuning pipeline (standalone scripts)
  collect_history.py   Discord -> data/raw/*.jsonl (incremental)
  build_dataset.py     data/raw -> data/sft (transcript chunks + personas)
  train_lora.py        QLoRA fine-tune -> models/adapters/
```

## License

MIT — see [LICENSE](LICENSE).
