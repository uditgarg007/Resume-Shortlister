---
title: Resume Shortlister
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Resume Shortlister

### India Runs Data & AI Hackathon — Intelligent Candidate Discovery & Ranking

> Given **100,000+ candidate resumes** and **any job description**, find the best matches in ~30 seconds.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web app
python app.py

# 3. Open http://localhost:5000
```

On first launch the app loads pre-computed embeddings (~20s). Every subsequent JD query takes **30–60 seconds** — no re-embedding needed.

---

## What It Does

| Stage | Module | Algorithm |
|---|---|---|
| **1. Preprocessing** | `preprocessing.py` | Text aggregation + numeric signal extraction |
| **2. Hybrid Retrieval** | `hybrid_search.py` | BM25 (20%) + FAISS HNSW (80%) → RRF fusion |
| **3. Cross-Encoder Reranking** | `hybrid_search.py` | `ms-marco-MiniLM-L-6-v2` with chunk-and-max-pool |
| **4. Penalty & Bonus Engine** | `penalty.py` | 9 multiplicative penalties + 5 additive bonuses |
| **5. Web UI** | `app.py` | Flask app with real-time slider controls |

---

## Architecture

```
  Web UI / CLI
       │
       ▼
  JD Tier Parser  (must_have · good_to_have · bonus · disqualifiers)
       │
       ▼
  Hybrid Retrieval  ──  BM25 Index (20%)
                    └─  FAISS HNSW  (80%)   →  RRF Top-200
       │
       ▼
  Cross-Encoder Reranking  (sub-query decomposition per tier)
       │
       ▼
  Penalty & Bonus Engine  (9 rules, configurable via sliders)
       │
       ▼
  Final Ranked Candidates  (score ∈ [0, 1])
```

---

## Branch Structure

| Branch | Focus |
|---|---|
| **Main** ← you are here | Complete production app (web UI + full pipeline) |
| **finalRanker** | Penalty & Bonus Engine + CLI pipeline runner |
| **SemanticSearch** | Hybrid BM25 + HNSW + Cross-Encoder scoring |
| **preprocessing** | Data flattening & text corpus builder |

---

## Running the CLI Pipeline

```bash
# Default JD (Redrob AI Engineer)
python run_pipeline.py --verbose

# Custom JD from JSON
python run_pipeline.py --jd-json my_jd.json --top-k 50

# Force rebuild indices
python run_pipeline.py --rebuild
```

## Custom Dataset Upload

The web UI supports uploading your own candidate pool (`.csv`, `.json`, `.jsonl`). The app builds FAISS + BM25 indices on the fly.

## JD Format (JSON)

```json
{
  "must_have":    ["Built production embedding/retrieval systems"],
  "good_to_have": ["LLM fine-tuning experience LoRA QLoRA"],
  "bonus":        ["Shipped ranking system to real users at scale"],
  "disqualifiers":["Career entirely in academic research"]
}
```

Each entry should be a **short, focused query** (10–30 words) — Cross-Encoders work best on search-length inputs.

---

## Scoring Formula

```
final_score = semantic_score × penalty_multiplier + bonus_total
```

- `semantic_score` — calibrated Cross-Encoder score ∈ [0.30, 0.95]
- `penalty_multiplier` — product of all applicable penalty weights ∈ (0, 1]
- `bonus_total` — sum of applicable bonus points

## File Reference

```
app.py              ← Flask web server
hybrid_search.py    ← BM25 + HNSW + Cross-Encoder
penalty.py          ← 9 penalties + 5 bonuses
preprocessing.py    ← Text & numeric feature extraction
run_pipeline.py     ← CLI end-to-end orchestrator
dataset_manager.py  ← Custom dataset upload handler
default_jd.json     ← Default JD (Redrob AI Engineer)
candidate.json      ← 50-candidate sample for testing
requirements.txt    ← Python dependencies
```
