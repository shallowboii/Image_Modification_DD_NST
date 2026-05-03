# Image Modification with DeepDream and Neural Style Transfer

Bachelor's thesis project — **Szöllős Marcell**, Óbuda University, 2026.
Supervisor: dr. Vámossy Zoltán.

A web-based image modification system that combines **DeepDream** and
**Neural Style Transfer (NST)** in a cyclic pipeline to produce
psychedelic-style imagery, with the research goal of comparing the
output to visual experiences reported under the influence of
hallucinogenic substances (LSD, psilocybin, DMT).

The thesis itself (Hungarian, LaTeX) is not part of this repository.
Section references in this README (e.g. "§9.3") refer to chapters of
the bound thesis document.

---

## Table of contents

1. [Abstract](#abstract)
2. [System architecture](#system-architecture)
3. [Installation](#installation)
4. [Running the application](#running-the-application)
5. [Presets](#presets)
6. [Reproducing the thesis experiments](#reproducing-the-thesis-experiments)
7. [Analysis & figure scripts](#analysis--figure-scripts)
8. [Configuration files reference](#configuration-files-reference)
9. [Repository layout](#repository-layout)
10. [Limitations & future work](#limitations--future-work)
11. [License](#license)

---

## Abstract

The system maximises activations of selected layers of a pre-trained
**ResNet-50** network (DeepDream, gradient ascent on an image pyramid)
and combines them with Gatys-style NST (Gram matrix style loss on
ResNet-50 `layer1`/`layer2`/`layer3`, Adam optimiser, TV regularisation).
The two algorithms are applied in **alternating cycles** (DD → NST → DD → …),
optionally driven by an **adaptive controller** that adjusts per-cycle
parameters based on per-image metrics (high-frequency ratio, edge
density, saturation, contrast).

The research question is whether the resulting imagery matches the
canonical descriptions of visually altered states — Klüver's
*form constants* (grids, cobwebs, tunnels, spirals), the
Ermentrout–Cowan cortical model of periodic V1 activity, and the
Dittrich 5D-ASC scale's *visionary restructuralization* dimension
(more vivid colours, sharper contours). The answer, supported by the
quantitative analysis in §9 of the thesis, is that the **hybrid**
output reproduces both vivid colours (Dittrich) and periodic
kaleidoscope-like patterns (Ermentrout–Cowan) more strongly than
either DeepDream or NST alone.

---

## System architecture

Three independently developable layers:

| Layer            | Tech stack                                          | Path |
|------------------|-----------------------------------------------------|------|
| Frontend         | React 19 + TypeScript, Vite, native `fetch`         | [frontend/react-app/](frontend/react-app/) |
| Backend          | FastAPI + Uvicorn, SQLAlchemy + SQLite, Pydantic    | [backend/app/](backend/app/) |
| Processing core  | Python 3.11, PyTorch 2.10 + CUDA 13.0, NumPy, OpenCV, Pillow, Matplotlib | [utils/](utils/) |

### REST endpoints

- `GET  /health` — liveness check.
- `POST /generate` — multipart upload (content image + optional style image + preset name + parameter overrides). Returns a `job_id` immediately and runs the pipeline in a `BackgroundTasks` worker.
- `GET  /job/{job_id}` — current job status (`pending` → `running` → `done`/`failed`) and output path. The frontend polls every 3 s.

### Pipeline

```
input image → resize to 1080 px →  (DeepDream → NST) × N cycles  → Lanczos upsample → output
```

- **Fixed pipeline** ([run_cyclic_pipeline](backend/app/services/orchestrator.py)) — every cycle uses the same preset parameters.
- **Adaptive pipeline** ([run_adaptive_cyclic_pipeline](backend/app/services/orchestrator.py)) — each cycle calls [image_analyzer.analyze_image](backend/app/services/image_analyzer.py); the [AdaptiveController](backend/app/services/adaptive_controller.py) decays `dream_weight` (0.85^c-1), boosts `tv_weight` if HF ratio > 0.35, shrinks `pyramid_size` if edge density > 0.18, and clamps `lr ∈ [0.02, 0.15]`.

The DeepDream core ([utils/main_deepdream.py](utils/main_deepdream.py))
is adapted from
[gordicaleksa/pytorch-deepdream](https://github.com/gordicaleksa/pytorch-deepdream).
The NST core ([utils/neural_style_transfer.py](utils/neural_style_transfer.py))
is a custom Adam-based reimplementation of the Gatys et al. method on
ResNet-50.

---

## Installation

### Prerequisites

- Python **3.11**
- Node.js **18+**
- An NVIDIA GPU with **CUDA 13.0** drivers (CPU fallback works but a
  `HEAVY` preset takes 15–30 minutes instead of seconds)
- Windows 10/11, Linux, or macOS

### Backend / processing core

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

pip install -r backend/requirements.txt
```

The pre-trained ResNet-50 weights live under [utils/model/](utils/model/) and are loaded automatically on first run.

### Frontend

```bash
cd frontend/react-app
npm install
```

---

## Running the application

Open two terminals.

**Terminal 1 — backend:**

```bash
uvicorn backend.app.main:app --reload --port 8000
```

**Terminal 2 — frontend:**

```bash
cd frontend/react-app
npm run dev
```

Then open the URL Vite prints (usually <http://localhost:5173>).
Upload a content image, optionally a style image, pick a preset (or leave
on `Default`), and hit generate. The frontend polls the backend every 3 s
and renders the result when ready.

---

## Presets

Defined in [utils/data/presets/presets.json](utils/data/presets/presets.json).
Resolution precedence at runtime: **user override → preset value → hard default**.

| Preset      | Cycles | DD layers              | Pyramid | DD iters | DD `lr` | `dream_blend` | NST steps | Style weight | Character |
|-------------|--------|------------------------|---------|----------|---------|---------------|-----------|--------------|-----------|
| `GENTLE`    | 1      | layer1                 | 4       | 3        | 0.02    | 0.40          | 80        | 5e5          | Barely visible, pastel |
| `FAST`      | 2      | layer1, layer2         | 5       | 8        | 0.05    | 0.70          | 100       | 1e6          | Quick, mid-intensity |
| `Default`   | 2      | layer1                 | 3       | 10       | 0.03    | 0.80          | 100       | 1e6          | Balanced default |
| `PORTRAIT`  | 2      | layer1                 | 5       | 4        | 0.02    | 0.50          | 150       | 8e5          | Conservative, face-friendly |
| `PAINTERLY` | 1      | layer1                 | 4       | 3        | 0.015   | 0.25          | 400       | 2e6          | NST-dominated, painterly |
| `LSD`       | 3      | layer1, layer3         | 6       | 12       | 0.06    | 0.80          | 200       | 1e6          | Spirals, kaleidoscope bursts |
| `DEEP`      | 3      | layer1, layer4         | 7       | 15       | 0.07    | 0.90          | 200       | 1.5e6        | Cobweb-like, fine DD detail |
| `HEAVY`     | 4      | layer1, layer2, layer3 | 7       | 20       | 0.06    | 0.85          | 300       | 1e6          | Most intense, heavy distortion |

---

## Reproducing the thesis experiments

All experiments are driven by JSON configs in [thesis_configs/](thesis_configs/)
and the [analysis/run_experiments.py](analysis/run_experiments.py) runner.
It reads a config, dispatches each `(input, style, preset, mode)` tuple
to the orchestrator, writes outputs to a structured tree under
`analysis/experiments/`, and records wall-clock timings to
`analysis/experiments/runtime_report.json`.

```bash
# Full validation matrix (multi-input + multi-style + adaptive vs fixed)
python analysis/run_experiments.py thesis_configs/experiments_config.json

# Per-preset wall-clock timing chart data (all 8 presets)
python analysis/run_experiments.py thesis_configs/experiments_preset_timing.json

# Adaptive showcase (HEAVY fixed vs adaptive)
python analysis/run_experiments.py thesis_configs/experiments_adaptive_showcase.json
```

---

## Analysis & figure scripts

All under [analysis/](analysis/). Each consumes one or more of the JSON
configs in [thesis_configs/](thesis_configs/).

| Script | Produces | Reads |
|---|---|---|
| [psychedelic_feature_analysis.py](analysis/psychedelic_feature_analysis.py) | Per-image metrics: spectral slope α, fractal dimension D, mean saturation, autocorrelation peak ratio (§9.1, §9.4 figures). | Image folders directly. |
| [make_collage.py](analysis/make_collage.py) | Labelled n-up image collages used throughout the Results chapter. | Any `thesis_configs/collage_*.json`. |
| [make_nst_triplet.py](analysis/make_nst_triplet.py) | Content / style / NST-output triplet figure (§7 NST figure). | Hard-coded paths. |
| [make_blend_figure.py](analysis/make_blend_figure.py) | `dream_blend` sweep figure (§7 DeepDream service figure). | Hard-coded paths. |
| [make_runtime_chart.py](analysis/make_runtime_chart.py) | Per-mode runtime bar chart (§9.3 *runtime_chart*). | `analysis/experiments/runtime_report.json` from the runner. |
| [make_preset_timing_chart.py](analysis/make_preset_timing_chart.py) | DD-vs-NST stacked bar per preset (§9.3 *preset_timing_chart*). | `runtime_report_preset_timing.json`. |
| [plot_experiment_log.py](analysis/plot_experiment_log.py) | Per-cycle metric & parameter trajectories of an adaptive run (§9.2 *adaptive_metrics_heavy*). | `experiment_log.json` written by the adaptive orchestrator. |

---

## Configuration files reference

The thesis figures are reproducible directly from the JSON configs
checked into [thesis_configs/](thesis_configs/). They fall into three groups.

### 1. Experiment runner configs (`run_experiments.py`)

Each entry is `{category, preset, mode, style, input}`. The runner uses
`category` to decide the output folder layout.

| Config | Category coverage | Used for |
|---|---|---|
| [experiments_config.json](thesis_configs/experiments_config.json) | `multi_input`, `multi_style`, `adaptive_vs_fixed` | Main thesis matrix: 4 inputs × LSD preset, 3 styles × church input, fixed vs adaptive on yosemite. |
| [experiments_preset_timing.json](thesis_configs/experiments_preset_timing.json) | `preset_timing` | All 8 presets on the same yosemite + kaleidoscope pair → §9.3 timing analysis. |
| [experiments_adaptive_showcase.json](thesis_configs/experiments_adaptive_showcase.json) | `adaptive_vs_fixed` | HEAVY preset, fixed vs adaptive (§9.2 *adaptive_vs_fixed_heavy*). |

### 2. Collage configs (`make_collage.py`)

Each entry is `{path, label}` (or `{input, output, label}` for before/after pairs). Used to build the labelled grids in the Results chapter.

| Config | Figure |
|---|---|
| [collage_church.json](thesis_configs/collage_church.json) | All 8 presets on `chruch.jpg` (§9.1 preset spectrum grid). |
| [collage_multi_input.json](thesis_configs/collage_multi_input.json) | LSD preset across 4 different inputs (church, mountain, dahlia, waterfall). |
| [collage_multi_style.json](thesis_configs/collage_multi_style.json) | LSD preset, church input, three style images (kaleidoscope, cyberpunk, Starry Night). |
| [collage_portrait.json](thesis_configs/collage_portrait.json) | Portrait input with the `PORTRAIT` preset before/after. |
| [collage_adaptive_vs_fixed.json](thesis_configs/collage_adaptive_vs_fixed.json) | LSD fixed vs adaptive side-by-side. |
| [collage_adaptive_heavy.json](thesis_configs/collage_adaptive_heavy.json) | HEAVY fixed vs adaptive (§9.2 *adaptive_vs_fixed*). |
| [collage_pure_comparison.json](thesis_configs/collage_pure_comparison.json) | Original / DD-only / NST-only / hybrid (§9.4 *pure_comparison_collage*). |

### 3. Metrics input

| File | Purpose |
|---|---|
| [pure_comparison_metrics.json](thesis_configs/pure_comparison_metrics.json) | Lists the original / DD-only / NST-only / hybrid image triplet that `psychedelic_feature_analysis.py` consumes for §9.4 *pure_comparison_metrics*. |

> The `analysis/batch_config*.json` files are ad-hoc dev artefacts from
> manual frontend/API testing — they reference job-ID-stamped paths
> under `backend/results/` and are **not** required to reproduce any
> thesis figure.

---

## Repository layout

```
backend/
  app/
    main.py                  FastAPI entry point, REST endpoints
    db.py                    SQLAlchemy Job model, init_db
    services/
      orchestrator.py        fixed + adaptive pipeline driver
      deepdream_service.py   run_deepdream_once (with dream_blend)
      nst_service.py         run_nst (NumPy ↔ tensor bridge)
      adaptive_controller.py AdaptiveController rules
      image_analyzer.py      4-metric per-image analyser
  database.db                SQLite job table
  uploads/                   incoming images
  results/                   per-job outputs (gitignored)
utils/
  main_deepdream.py          DeepDream core (pyramid + octave detail)
  neural_style_transfer.py   NST core (Gram-matrix, Adam)
  utilities/                 net helpers, constants
  model/                     ResNet-50 + custom .pth weights
  data/                      in-images/, out-images/, styles/, presets/presets.json
frontend/react-app/src/      React + TypeScript components
analysis/                    experiment runner, figure scripts, ad-hoc batch configs
thesis_configs/              JSON configs that drive the thesis figures
```

---

## Limitations & future work

Discussed in detail in §9.5 of the thesis:

- Manual preset tuning — the interaction between `style_weight`, `lr`, `dream_blend` and cycle count is non-linear; presets were found empirically.
- 1080 px processing width caps the visible hallucinatory detail; VRAM and runtime scale quadratically.
- No structured perceptual user study — psychedelic faithfulness is judged on quantitative metrics and the author's own visual analysis only.
- Adaptive controller thresholds (HF ratio 0.35, edge density 0.18) did not fire even on the most aggressive `HEAVY` preset; they protect against extreme inputs but should be re-tuned on a wider corpus.
- Single backbone (ResNet-50) — adding VGG-16, GoogLeNet or EfficientNet would let users compare hallucination character across architectures.
- Spiral- and cobweb-form constants are visible but not directly measured; polar-coordinate FFT and radial-gradient detectors would close that gap.
- An After Effects plugin would extend the pipeline to frame-level video processing.

---

## License

See [LICENSE](LICENSE).
