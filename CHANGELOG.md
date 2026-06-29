# Changelog

All notable changes to **Resume Shortlister** are documented here.

---

## [v1.0.0] — 2026-06-29 — Final Hackathon Submission

### Added
- **Premium UI redesign** — Deep Navy & Coral glassmorphism theme, Plus Jakarta Sans typography, animated mesh background, spring hover effects, tier pip indicators
- **`requirements.txt`** — pinned Python dependencies for reproducible installs
- **`DEPLOYMENT.md`** — step-by-step guide for Hugging Face Spaces, Render, Railway, and Cloud Run; explains why Vercel is not compatible
- **`HOW_IT_WORKS.md`** — plain-English walkthrough of every algorithm, formula, and design decision
- **`STATUS.md`** — current capabilities and future roadmap
- **`candidate.json`** — 50-candidate sample for local testing without the full 487 MB dataset

### Changed
- Renamed `sample_candidates.json` → `candidate.json` to match submission spec
- Untracked large binary artifacts from git (`processed/` data files, embeddings, indices) — they are generated locally via `python preprocessing.py`

---

## [0.5.0] — 2026-06-20 — Semantic-Score Branch

### Added
- **Custom dataset upload** via the web UI (`.csv`, `.json`, `.jsonl`)
- **DatasetManager** (`dataset_manager.py`) — builds FAISS + BM25 indices on the fly for any uploaded dataset
- **Result caching** — default JD results cached to disk so first-load is instant for judges
- **Score calibration fix** — top candidates no longer tied at 1.0; floor set to 0.30, ceiling to 0.95

### Changed
- Hybrid RRF weights tuned: BM25 20%, HNSW Vector 80%
- Cross-Encoder batch size increased for throughput

---

## [0.4.0] — 2026-06-18 — Final-Ranker Branch

### Added
- **9-rule Penalty Engine** (`penalty.py`): Ghosting, Job-Hopping, Role Mismatch, No-Code, Overqualified, Relocation Risk, Notice Period, Consulting Background, Unverified Contact
- **5 Additive Bonuses**: Short Notice, Tier-1 City, GitHub Activity, Skill Assessment, Engagement Score
- Exponential diminishing returns on bonuses — prevents artificial score capping

### Changed
- Final score formula: `semantic_score × penalty_multiplier + bonus_total`

---

## [0.3.0] — 2026-06-17 — Hybrid Search Branch

### Added
- **Hybrid Semantic Scoring** (`hybrid_search.py`): BM25 + FAISS HNSW via Reciprocal Rank Fusion
- **Cross-Encoder reranking** (`ms-marco-MiniLM-L-6-v2`) with chunk-and-max-pool strategy for long resumes
- **Tiered JD scoring** — each tier (`must_have`, `good_to_have`, `bonus`, `disqualifiers`) scored independently via focused sub-queries
- **FAISS HNSW index** built from `all-MiniLM-L6-v2` embeddings

---

## [0.2.0] — 2026-06-03 — Preprocessing Branch

### Added
- **Text corpus builder** (`preprocessing.py`) — aggregates headline, summary, career history, education, skills, certifications, languages into clean text per candidate
- **Numeric signal extractor** — 31 numeric features from `redrob_signals`
- **Flask web app** (`app.py`) with sliders for real-time penalty and weight tuning

---

## [0.1.0] — 2026-06-01 — Initial Commit

- Repository initialized with raw dataset analysis notebook and schema documentation
