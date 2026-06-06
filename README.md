# Kiwami Capture

Private Edge AI for Capturing Expert Skill

熟練者の一瞬を、初心者が学べるPractice Memoryへ変える。

熟練者の動きに宿る知を、次の世代のPractice Memoryへ。

Kiwami Capture turns expert motion differences into Practice Memory.

師匠と初心者の差分を、次に練習できる稽古記憶へ変える。

「見て盗む」だけでは残しきれない知を、次の世代へ。

## Positioning

Kiwami Capture is not a Tebiki clone.
It is not a general video summarizer.
It is not only a SOP generator.

Kiwami Capture compares a master clip and an optional practice clip, then turns the learning difference into reusable Practice Memory.

Tebiki creates video manuals.
Video-to-SOP tools create procedures.
Kiwami Capture captures what successors fail to see.

YouTube shows the demonstration.
Kiwami Capture preserves what successors should notice and practice.

The product flow is intentionally different:

Master Clip
+ optional Practice Clip
+ optional Context Layers
→ Key Moments / Difference Capture
→ Practice Memory
→ Local-first Skill Archive

## Problem

Workplace and craft training often fails at the exact point where a learner or successor needs the hidden difference that an expert makes look effortless. Manuals can describe steps. Videos can show motion. The critical cue still gets missed.

## Solution

Kiwami Capture starts from the expert demonstration itself. Practice Clip, Audio / Spoken Hint, and Context Note are optional layers that enrich the capture when they are available.

This MVP is local-first and uses Liquid LFM first, with Rule and Mock available as fallback modes.
Liquid LFM uses a local model server for Practice Memory generation.

## Why This Is Different

- It centers the expert-vs-learner gap instead of generic summarization.
- It turns that gap into a reusable Practice Memory card.
- It keeps raw clips local while making only the learning essence shareable.
- It uses a real local Liquid text generation path for Practice Memory.

## Why Local-First / Edge AI

Raw skill videos may contain private workplace knowledge:

- workshop techniques
- factory processes
- safety-sensitive operations
- training mistakes
- internal know-how
- customer or site information
- faces, hands, tools, machines, and workspace layout

Kiwami Capture keeps raw clips local and turns only the learning essence into Practice Memory.

## Why Liquid LFMs Fit

Liquid LFM is used in the text generation layer.
It generates Practice Memory from structured expert and practice observations and the master hint through a local model server.

Switch models by restarting `llama-server` with another Liquid GGUF model.
Use `LIQUID_LFM_BASE_URL` and `LIQUID_LFM_MODEL` when needed.

## Practice Memory Optimization Loop

Kiwami Capture does not stop at AI generation. Practice Memory can be reviewed by an expert, corrected, approved, and exported as a tuning record for skill-specific Liquid model optimization.

## Training JSONL Export

Approved sessions can be exported as JSONL. Raw videos are not included by default. The export contains structured observations, Practice Memory, and expert-reviewed corrections.

## Runtime / Deployment Path

Current demo:

- Flask local app
- OpenCV keyframe extraction
- Liquid LFM through llama.cpp / GGUF

AMD path:

- Ryzen AI PC for local on-site inference
- ROCm-ready acceleration path where available

Device path:

- The architecture is designed for mobile / edge deployment paths such as LEAP, and the AMD / ROCm path can accelerate local inference where available.

Goal: capture on site, process on edge, keep raw demonstrations local, share only Practice Memory.

## Reference Pattern Extraction

The reference repositories informed the architecture, but they were not copied.

What was extracted:

- from `Video-to-SOP-Generator`: the upload → frame extraction → structured output workflow
- from `SOP-LVM-ICL-Ensemble`: temporal reasoning and multimodal workflow framing
- from `Label Studio`: human review and correction thinking
- from `WhisperX`: timestamped speech and cue-alignment direction
- from `sop-generator-llm-rag`: grounded document generation and versioned outputs

How Kiwami Capture redesigns those patterns:

Master Clip + optional Practice Clip + optional Context Layers
→ Difference Capture / Key Moments
→ Practice Memory
→ Local-first Skill Archive

That is the original product flow.
The app uses familiar OSS patterns only as architectural reference points.

## Default Sample

Pottery Centering Technique

Japanese theme: 陶芸の中心取り

Why pottery works:

- hand pressure
- center wobble
- wheel timing
- water amount
- finger angle
- pause timing
- clay surface change
- rhythm
- material response

## Other Possible Domains

- factory training
- tool handling
- repair work
- food preparation
- craft education
- machine operation
- safety training
- nursing care
- construction
- cleaning technique
- sports technique
- vocational schools
- internal workplace education

## Core Workflow

1. Upload Master Clip
2. Optionally add Practice Clip, Audio / Spoken Hint, and Context Note
3. Enter craft name and optional Master Hint
4. Select analysis mode
5. Process locally
6. Extract key frames
7. Generate Practice Memory
8. Review side-by-side
9. Edit Practice Memory
10. Export Markdown or JSON
11. Open Project Explanation / Judge Mode

## MVP Features

- Master Clip upload
- Optional Practice Clip upload
- Optional Audio / Spoken Hint
- Optional Context Note / Master Hint
- Craft name input
- AI mode selector: Mock, Rule, Liquid LFM
- Local key frame extraction
- Master-only or side-by-side comparison
- Practice Memory generation
- Editable Practice Memory card
- Evidence notes
- Local-first privacy badge
- Markdown export
- JSON export
- Project Explanation / Judge Mode page
- Local session archive

## Screens / Pages

- `GET /` home
- `POST /process` process upload
- `GET /compare/<session_id>` compare
- `GET /memory/<session_id>` practice memory
- `POST /memory/<session_id>/update` update memory
- `GET /judge/<session_id>` project explanation
- `GET /export/<session_id>/markdown` export markdown
- `GET /export/<session_id>/json` export json

## Screenshots

Add screenshot reference paths here:

- Home: `docs/screenshots/home.png`
- Compare: `docs/screenshots/compare.png`
- Practice Memory: `docs/screenshots/practice-memory.png`
- Project Explanation: `docs/screenshots/project-explanation.png`

These references are intentionally optional for the stable MVP checkpoint. The image files do not need to exist yet.

## Stable MVP Checkpoint

This version includes:

- local upload flow
- Master-only capture with optional comparison
- key frame extraction with safe fallback
- Practice Memory generation
- editable Practice Memory
- Markdown / JSON export
- Project Explanation page
- Mock / Rule / Liquid LFM modes
- local-first product positioning

## Architecture

### Stack

- Python
- Flask
- OpenCV
- Jinja templates
- local file storage
- Markdown export
- JSON export
- CSS only

### File Layout

```text
kiwami-capture/
  README.md
  requirements.txt
  app.py
  src/
    __init__.py
    video_processor.py
    analyzer.py
    schema.py
    exporter.py
    storage.py
  templates/
    index.html
    compare.html
    practice_memory.html
    judge.html
  static/
    style.css
  uploads/
    .gitkeep
  outputs/
    .gitkeep
  samples/
    README.md
```

### Session Storage

Each session lives under:

`outputs/<session_id>/session.json`

Uploaded clips are stored locally under `uploads/<session_id>/`.
Extracted frames and exports are stored under `outputs/<session_id>/`.

## Model Modes

Kiwami Capture supports three model modes:

- Mock
- Rule
- Liquid LFM

### Mock

Stable sample demo output for Pottery Centering Technique.

### Rule

Offline fallback for generic skill-memory generation from craft name and hint.

### Liquid LFM

Uses a local Liquid model server to generate Practice Memory from structured expert and practice observations.

Local setup:

1. Install llama.cpp
2. Start the local Liquid model server with:
   - `llama-server -hf LiquidAI/LFM2.5-1.2B-JP-GGUF`
   - or `llama-server -hf LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M`
3. Run Kiwami Capture
4. Select Liquid LFM mode

## Practice Memory Schema

```json
{
  "craft": "Pottery Centering Technique",
  "skill_focus": "Stabilize the clay before shaping",
  "watch_points": [
    "hand pressure",
    "center wobble",
    "water amount",
    "pause timing"
  ],
  "timing_cue": "the first 5 seconds after the wheel starts",
  "motion_cue": "hands stay close and apply light, even pressure",
  "material_cue": "the clay should rise smoothly without leaning",
  "sound_cue": "steady wheel sound with minimal scraping",
  "common_mistake": "pressing too hard before the clay is centered",
  "master_hint": "Do not force the clay. Keep your hands steady, use light pressure, and wait until the center becomes stable.",
  "practice_task": "Repeat a 20-second centering drill. Stop if the clay starts leaning, then reset your hand pressure.",
  "evidence": [
    "Master clip shows stable hand pressure",
    "Practice clip shows visible center wobble",
    "Practice clip applies pressure before the clay stabilizes"
  ],
  "privacy_mode": "local_only",
  "model_mode": "mock",
  "shareable": true
}
```

## Reference Repositories

Primary reference:

https://github.com/Shezan57/Video-to-SOP-Generator

Use as implementation inspiration for:

- video upload
- frame extraction
- audio/transcription pipeline
- AI analysis pipeline
- document generation

Research reference:

https://github.com/moucheng2017/SOP-LVM-ICL-Ensemble

Use for:

- video-language workflow understanding
- SOP generation research context
- temporal step reasoning

Human review reference:

https://github.com/HumanSignal/label-studio

Use for:

- human-in-the-loop correction idea
- expert review workflow
- annotation UI inspiration

Speech reference:

https://github.com/m-bain/whisperX

Use for future:

- timestamped transcription
- speaker diarization
- sound cue alignment

SOP / RAG reference:

https://github.com/praveen0777/sop-generator-llm-rag

Use later for:

- expanding Practice Memory into SOP
- company manual grounding
- versioned training documents

Liquid AI references:

https://www.liquid.ai/models

https://huggingface.co/LiquidAI

## How to Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open:

`http://127.0.0.1:5000`

## Honest Implementation Status

Current MVP:

- local upload flow
- key frame extraction
- Mock, Rule, and Liquid LFM Practice Memory generation
- side-by-side comparison UI
- editable Practice Memory card
- Markdown/JSON export
- Project Explanation page
- local-first product positioning

Not added yet:

- local speech transcription
- audio cue extraction
- vision model comparison
- private workplace deployment

## Future Roadmap

### Phase 1: Local Practice Memory

- Master-only capture
- optional Practice Clip comparison
- Practice Memory generation
- local-first upload
- editable learning card
- export

### Phase 2: Multi-domain Skill Capture

- craft
- factory training
- food preparation
- repair work
- sports training
- tool handling
- safety training

### Phase 3: Kiwami LFA Expansion

- approved Practice Memory archive
- SOP generation
- learner curriculum
- expert review history
- team training dashboard
- company knowledge base

### Phase 4: Enterprise / Edge Deployment

- local inference
- private workshop deployment
- on-prem training archive
- Liquid LFM text layer improvements
- model evaluation loop
- secure sharing of only learning cards

## Honest Limitations

- Liquid Vision selected keyframe analysis is available; audio understanding is not added yet.
- The app is intentionally small and local-first, not a full enterprise platform.
- It is a foundation for a serious product, not the final product.

## Why This Is Not Just a Mock

The current product includes Mock and Rule fallbacks, but Liquid LFM also runs locally for Practice Memory generation.

The important design is the local-first skill memory pipeline.
