# Resume Shortlister — Intelligent Candidate Discovery & Ranking

> **India Runs Data & AI Challenge**
> Full Pipeline: Web Frontend → Hybrid Semantic Scoring → Penalty & Bonus Engine → Final Ranking

---

## 🚀 Quick Start

### Option A: Web UI (recommended)

```bash
pip install flask pandas numpy rank-bm25 faiss-cpu sentence-transformers
python app.py
# Open http://localhost:5000
```

1. Select **Default (100k candidates)** or upload your own `.csv` / `.json` dataset via the UI
2. Click **"Load Default JD"** to test with the Redrob AI Engineer posting
3. Adjust **tier weights** and **penalty multipliers** via sliders
4. Click **"Run Pipeline"** → see ranked results in ~30-60 seconds

### Option B: CLI

```bash
# Full pipeline with default JD
python run_pipeline.py --verbose

# Custom JD from JSON
python run_pipeline.py --jd-json my_jd.json --verbose
```

> **First run** downloads transformer models (~80 MB) and loads pre-computed embeddings. Subsequent runs start in ~5 seconds.

---

## 📋 What This Does

Given **100,000+ candidate resumes** and **any job description**, this system finds the best candidates using:

| Stage | What | How |
|---|---|---|
| **Preprocessing** | Cleans raw JSON data | Text aggregation + numeric signal extraction |
| **Hybrid Search** | Finds text matches | BM25 (20%) + HNSW Vector Search (80%) → Cross-Encoder reranking |
| **Penalty Engine** | Applies business rules | 9 multiplicative penalties + 5 additive bonuses |

### Works on ANY Job Description & ANY Candidate Dataset

The system is not hardcoded to one JD or candidate pool. Through the web UI or CLI, you can:
- **Upload Custom Datasets**: Upload a `.csv` or `.json` file, and the app automatically parses the text, calculates dense vector embeddings, builds FAISS indices, and saves it as a new searchable candidate pool.
- Enter **must-haves**, **good-to-haves**, **bonuses**, and **disqualifiers** as focused sub-queries
- Adjust **tier weights** (how much each tier matters)
- Tune **penalty multipliers** (how harsh each penalty is)

The **default Redrob AI Engineer JD** and **100k candidate pool** are included for judges to test instantly.

### Why it's fast for 100k candidates

Pre-computed BM25 + FAISS HNSW indices ship with the app. Only the **Cross-Encoder reranking** (on the top 50-100 candidates) runs fresh per JD. This means:
- Engine startup: ~20 seconds
- Each new JD: ~30-60 seconds

---

## 🏗️ Architecture

```
  Web UI (any JD)              CLI (--jd-json)
       │                            │
       ▼                            ▼
┌─────────────────────────────────────────┐
│         JD Tier Parser                  │
│  must_have | good_to_have | bonus | disq│
│  (weights configurable via sliders)     │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    STAGE 1: HYBRID RETRIEVAL            │
│                                         │
│  BM25 Index (20%) ─┐                   │
│                     ├→ RRF → Top 200    │
│  FAISS HNSW (80%) ──┘                   │
│  (pre-computed embeddings)              │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    STAGE 2: CROSS-ENCODER RERANKING     │
│                                         │
│  Sub-query decomposition per tier       │
│  "uses ChatGPT" → LOW score            │
│  "builds ML models" → HIGH score        │
│  Percentile-based score calibration     │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│    STAGE 3: PENALTY & BONUS ENGINE      │
│                                         │
│  9 Penalties (configurable via sliders):│
│    Ghost ×0.20 | Mismatch ×0.50        │
│    Hopper ×0.60 | No Code ×0.70        │
│    Consult ×0.65 | Low Profile ×0.80   │
│    CV-Only ×0.55 | Research ×0.40      │
│    LangChain ×0.45                      │
│  5 Bonuses: Notice +0.05 | City +0.05  │
│    GitHub +0.03 | Assessed +0.02       │
│    Engaged +0.03                        │
└────────────────┬────────────────────────┘
                 │
          final_ranking.csv
          (rank, candidate_id, final_score 0→1)
```

---

## 📁 Repository Structure

```
resume-shortlister/
│
├── app.py                ←  Web frontend (Flask server)
├── run_pipeline.py       ←  CLI pipeline runner
├── hybrid_search.py      ←  Hybrid semantic scorer
├── penalty.py            ←  Penalty & bonus engine
├── preprocessing.py      ←  Data preprocessing
├── default_jd.json       ←  Default JD for testing
│
├── templates/
│   └── index.html        ←  Web UI template
├── static/
│   ├── style.css         ←  Premium dark theme
│   └── app.js            ←  Frontend logic
│
├── candidates.jsonl      ←  Full dataset (~487 MB)
├── HOW_IT_WORKS.md       ←  Plain-English walkthrough
├── README.md             ←  This file
├── DEPLOYMENT.md         ←  Production deployment guide
├── STATUS.md             ←  Current capabilities & future roadmap
│
├── datasets/             ←  Custom user-uploaded datasets
└── processed/            ←  Default 100k pre-computed indices
    ├── text_corpus.pkl
    ├── numeric_signals.pkl
    ├── bm25_index.pkl
    ├── faiss_hnsw.index
    └── embeddings.npy
```

---

## ⚙️ Prerequisites

**Python:** 3.9+ (3.10+ recommended)

```bash
pip install flask pandas numpy rank-bm25 faiss-cpu sentence-transformers python-docx
```

| Package | Purpose |
|---|---|
| `flask` | Web frontend server |
| `pandas` | DataFrames and data manipulation |
| `numpy` | Numerical operations |
| `rank-bm25` | BM25Okapi keyword search |
| `faiss-cpu` | HNSW vector index |
| `sentence-transformers` | Bi-Encoder + Cross-Encoder models |
| `python-docx` | Parse `.docx` JD files (optional) |

---

## 🎛️ Web UI Features

### JD Input
- Enter requirements as **focused sub-queries** (10-30 words each)
- Separate tiers: Must Have, Good to Have, Bonus, Disqualifiers
- Add/remove queries dynamically
- **"Load Default JD"** button pre-fills the Redrob AI Engineer posting

### Tier Weight Sliders
| Slider | Default | Range | What it controls |
|---|---|---|---|
| Must Have Weight | 1.00 | 0–1 | How much must-have matches contribute |
| Good to Have Weight | 0.50 | 0–1 | How much nice-to-haves contribute |
| Bonus Weight | 0.25 | 0–1 | How much bonuses contribute |
| Disqualifier Penalty | 0.15 | 0–0.5 | How much to subtract for disqualifier matches |

### Penalty Multiplier Sliders
| Slider | Default | What it controls |
|---|---|---|
| Ghost Candidate | ×0.20 | Unresponsive/inactive candidates |
| Role Mismatch | ×0.50 | Non-tech titles with high AI scores |
| Job Hopper | ×0.60 | Frequent job switching |
| Coding Recency | ×0.70 | Senior with no coding proof |
| Consulting Lifer | ×0.65 | Entire career at service firms |
| Low Profile | ×0.80 | Incomplete profiles |
| CV/Speech Only | ×0.55 | Wrong domain specialization |
| Pure Research | ×0.40 | No production experience |
| LangChain Only | ×0.45 | API wrappers without ML foundations |

### Results View
- **Stats bar:** Total indexed, scored, top score, mean, penalized/bonused counts
- **Ranked table:** Score bars, penalty flags, bonus badges
- **Gold/silver/bronze** ranking for top 3

---

## 🔧 CLI Reference

### `app.py`
```bash
python app.py                  # default: http://localhost:5000
python app.py --port 8080      # custom port
```

### `run_pipeline.py`
```bash
python run_pipeline.py                          # default JD
python run_pipeline.py --jd-json my_jd.json     # custom JD from JSON
python run_pipeline.py --jd job.docx            # custom JD from .docx
python run_pipeline.py --top-k 50 --verbose     # control output
python run_pipeline.py --rebuild                # force index rebuild
```

### `preprocessing.py`
```bash
python preprocessing.py --sample    # 50 candidates (fast)
python preprocessing.py             # full dataset
```

---

## 📊 Custom JD JSON Format

```json
{
  "must_have": [
    "Built and deployed embeddings-based retrieval systems in production",
    "Production experience with vector databases FAISS Elasticsearch"
  ],
  "good_to_have": [
    "LLM fine-tuning experience LoRA QLoRA PEFT"
  ],
  "bonus": [
    "Shipped ranking system to real users at scale"
  ],
  "disqualifiers": [
    "Career entirely in pure research without production deployment"
  ]
}
```

Each entry should be a **short, focused query** (10-30 words). The Cross-Encoder works best with short queries — it was trained on Google-search-length inputs.

---

## 📈 Output Files

| File | Description |
|---|---|
| `processed/final_ranking.csv` | Final ranked output with scores, penalties, bonuses |
| `processed/semantic_scores.csv` | Intermediate semantic scores (before penalties) |
| `processed/text_corpus.pkl` | Cleaned candidate text corpus |
| `processed/numeric_signals.pkl` | Numeric signals per candidate |

---

## 🧠 How It Works (detailed)

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for a complete plain-English walkthrough of every algorithm, formula, and design decision.
