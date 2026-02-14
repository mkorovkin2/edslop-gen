# video-gen

End-to-end pipeline that turns a topic into short-form educational TikTok videos — scripts, audio, visuals, and AI-generated video clips.

Built on [LangGraph](https://github.com/langchain-ai/langgraph) with OpenAI (or Anthropic) for text, ElevenLabs for TTS, and Sora 2 for video generation.

## Pipeline

```
get_topic                          User provides a topic
    |
generate_variants                  LLM + web search produces 8 variant ideas
    |
user_select_variants               User picks up to 4 (or requests regen)
    |
generate_scripts                   LLM + web search writes 100-200 word scripts
    |
judge_scripts                      LLM-as-a-judge scores on 6-point rubric (up to 5 rounds)
    |
user_approve_scripts               User approves or requests revisions
    |
generate_audio                     ElevenLabs TTS generates .mp3 files
    |
generate_visual_scripts            LLM + web search creates per-segment visual cue scripts
    |
judge_visual_scripts               LLM-as-a-judge scores visual flow/sync/variety (up to 5 rounds)
    |
user_approve_visuals               User approves or requests revisions
    |
generate_video_breakdown           LLM + web search produces exact Sora prompts per segment
    |
user_approve_breakdown             User reviews prompts before generation
    |
generate_videos                    Sora 2 API generates .mp4 clips per segment
```

Every step saves a state snapshot to `output/<session_id>/thoughts/` for crash recovery.

## Setup

Requires Python 3.13.

```bash
chmod +x setup.sh && ./setup.sh
source .venv/bin/activate
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | yes | `openai` or `anthropic` |
| `OPENAI_API_KEY` | yes | Used for LLM, web search, and Sora |
| `ANTHROPIC_API_KEY` | if anthropic | Anthropic API key |
| `ELEVENLABS_API_KEY` | yes | ElevenLabs TTS |
| `ELEVENLABS_VOICE_ID` | yes | Voice to use for narration |
| `ELEVENLABS_MODEL_ID` | no | Defaults to `eleven_multilingual_v2` |
| `SORA_MODEL` | no | `sora-2` (default, faster) or `sora-2-pro` (higher quality) |

## Usage

```bash
# New session
python main.py

# Resume a crashed/interrupted session
python main.py --resume <session_id>
```

## Output structure

```
output/<session_id>/
    thoughts/          State snapshots (JSON) for each step
    scripts/           .txt script files
    audio/             .mp3 audio files (ElevenLabs)
    videos/
        variant_<id>/
            part_1.mp4
            part_2.mp4
            ...
    agent.log          Full debug log
```

## Project structure

```
main.py              CLI entry point + crash recovery
graph.py             LangGraph node/edge wiring
state.py             AgentState TypedDict
llm.py               LLM client factory (OpenAI/Anthropic/Sora)
prompts.py           All prompt templates
persistence.py       Save/load state snapshots
logger.py            Dual console + file logging
nodes/
    get_topic.py
    generate_variants.py
    user_select_variants.py
    generate_scripts.py
    judge_scripts.py
    user_approve_scripts.py
    generate_audio.py
    generate_visual_scripts.py
    judge_visual_scripts.py
    user_approve_visuals.py
    generate_video_breakdown.py
    user_approve_breakdown.py
    generate_videos.py
```

## Sora constraints

- Allowed durations: `4`, `8`, or `12` seconds per clip
- Vertical TikTok size: `720x1280`
- Models: `sora-2` (fast iteration) or `sora-2-pro` (production quality)
- No real people, copyrighted characters, or NSFW content
- Download URLs expire 1 hour after generation
