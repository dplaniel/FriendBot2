# FriendBot2

A Discord bot that generates images from text prompts using a local diffusion
pipeline — [Stable Diffusion 3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)
by default, or [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell).

This is a ground-up rewrite of the original FriendBot:

- **Modern Discord API** — discord.py 2.x with slash commands (`/artistic`) and
  classic prefix commands (`!artistic`) via hybrid commands, gateway intents, an
  async `setup_hook`, and `await bot.add_cog(...)`.
- **Local diffusion instead of the old Stable Diffusion subprocess** — image
  generation calls the local [`flux/generate.py`](../flux/generate.py) script's
  `load_pipeline` / `generate` functions directly, keeping the model resident in
  process for fast turnaround instead of shelling out and reloading weights per
  prompt. The model is selected with `FRIENDBOT_MODEL` (`sd35` default, or
  `flux`), the same choice as generate.py's `--model` flag.
- **No text-model machinery** — all chat-history collection and text-model
  training/fine-tuning code (and its data files) from the original were dropped.

## How it works

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

## Configuration

Copy `.env.example` to `.env` and fill it in (at minimum `FRIENDBOT_TOKEN`):

```bash
cp .env.example .env
$EDITOR .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `FRIENDBOT_TOKEN` | — | **Required.** Discord bot token. |
| `FLUX_REPO_PATH` | `../flux` | Path to the flux checkout with `generate.py` + `models/`. |
| `FRIENDBOT_MODEL` | `sd35` | Model to load: `sd35` or `flux`. |
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
  (required for the `!` prefix commands; slash commands work without it).

## Running

```bash
# from inside the venv that has both discord.py and the flux deps
python -m friendbot2
```

Then in Discord:

```
/artistic a photorealistic sunset over misty mountains
!gen a macro photo of a dragonfly
```

(`prompt`, `txt2img`, `generate`, `gen`, and `ig` are aliases for the prefix form.)

## License

MIT — see [LICENSE](LICENSE).
