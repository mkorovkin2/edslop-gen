# edslop-gen

Educational video content generator. This repo contains the full implementation under `claude-work/`, including a LangGraph-based workflow that researches a topic, writes a script, finds images, and generates narration audio.

## What's inside

- `claude-work/`: Main project (code, scripts, docs, outputs)
- `setup.txt`: Notes used during initial setup

For detailed usage and configuration, see `claude-work/README.md`.

## Quickstart

```bash
cd claude-work
./setup.sh
source .venv/bin/activate
cp .env.example .env
```

Edit `.env` with your API keys, then run:

```bash
python -m src.main "neural networks"
```

## Requirements

- Python 3.11+
- OpenAI API key (model + TTS access)
- Tavily API key
- ffmpeg (required for audio post-processing)

## Project layout

```
claude-work/
├── src/        # application code
├── scripts/    # utilities and one-off scripts
├── tests/      # test suite
├── docs/       # reference material
└── output/     # generated runs
```

## Tests

```bash
cd claude-work
source .venv/bin/activate
pytest tests/ -v
```
