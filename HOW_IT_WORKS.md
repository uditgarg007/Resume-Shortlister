# How This Resume Shortlister Actually Works

> A plain-English walkthrough — no jargon, no BS, just what each piece does and why.

---

## The Big Picture

You have **100,000+ candidate resumes** and **any job description**. You need to find the best 10-50 people. This system does it in 3 steps, exposed through a **web frontend** where you can configure everything:

```
Step 1: PREPROCESSING    →  Clean up the messy data (run once per dataset)
Step 2: HYBRID SEARCH    →  Find candidates who match YOUR JD text
Step 3: PENALTY ENGINE   →  Apply business rules with configurable penalties
```

**Web UI:** `python app.py` → open http://localhost:5000

**CLI:** `python run_pipeline.py --verbose`

---

## The Web Frontend

The frontend lets you do everything the pipeline does, but interactively:

1. **Select Dataset** — use the default 100k candidates, or upload your own `.csv` or `.json` file
2. **Paste any JD** — enter must-haves, good-to-haves, bonuses, and disqualifiers as focused sub-queries
3. **Adjust weights** — sliders control how much each tier contributes (must_have=1.0×, good_to_have=0.5×, bonus=0.25×)
4. **Tune penalties** — 9 penalty multipliers with sliders (ghost=×0.20, mismatch=×0.50, etc.)
5. **Run & view results** — ranked table with scores, penalty flags, and bonus badges
6. **Default JD included** — click "Load Default JD" to instantly load the Redrob AI Engineer posting for testing

### Why it's fast

The 100k candidates are pre-indexed with BM25 + FAISS HNSW embeddings (built once). Only the **Cross-Encoder reranking** (the expensive, intelligent step) runs fresh against your new JD. This means:
- First startup: ~20 seconds (loads models + indices)
- Each new JD run: ~30-60 seconds (cross-encoder on top 50-100 candidates)

---

## Step 1: Preprocessing (`preprocessing.py`)

### What it does
Takes the raw candidate data (a big JSON file) and produces two clean tables:

1. **Text table** — one row per candidate, containing everything about them mashed into a single text string (their title, summary, work history, skills, education, certifications)
2. **Numbers table** — one row per candidate, containing all their numeric signals (years of experience, GitHub score, notice period, recruiter response rate, etc.)

### Why two tables?
The search engine needs **text** to match against the JD. The penalty engine needs **numbers** to apply business rules. Keeping them separate makes each piece clean and fast.

---

## Step 2: Hybrid Search (`hybrid_search.py`)

This is the brain of the system. It figures out **how well each candidate's resume text matches the job description**.

### Stage 2a: Finding candidates (Retrieval)

We use **two search methods** and combine them:

#### Method 1: Keyword Search (BM25) — 20% weight
Think of this like Google in the early 2000s. It looks for **exact word matches**:
- JD says "FAISS" → does the resume contain "FAISS"? → higher score
- JD says "embeddings" → does the resume say "embeddings"? → higher score

**Good at:** Finding candidates who use the exact right terminology.
**Bad at:** Understanding meaning. It thinks "I use ChatGPT daily" is a great match for "build ML systems."

#### Method 2: Vector Search (HNSW) — 80% weight
This converts both the JD and every resume into **384-dimensional vectors** using a neural network (`all-MiniLM-L6-v2`). Similar meanings → similar vectors.

**Good at:** Understanding meaning, synonyms, related concepts.
**Bad at:** Exact technical terms. It might not distinguish "FAISS" from "Milvus."

#### Combining them: Reciprocal Rank Fusion (RRF)
```
score = 0.20 × (1/rank_in_BM25) + 0.80 × (1/rank_in_vector_search)
```

This gives us the **top 200 candidates** to examine more deeply.

---

### Stage 2b: Deep scoring (Cross-Encoder Reranking)

The Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) reads the JD query and the candidate's text **together** through all transformer layers. This is where "uses ChatGPT" vs "builds ML models" gets differentiated.

### Sub-Query Decomposition

Cross-encoders were **trained on short queries**. When we feed them a 100-word paragraph, they get confused. **The fix:** break each tier into **3-5 short, focused queries**:

```
OLD: "Built and deployed embeddings-based retrieval systems..."  ← 100+ words
NEW: 5 focused sub-queries of 10-20 words each
```

For each candidate, we score against ALL sub-queries and take the **MAX** (best matching aspect wins).

### Score Calibration

The cross-encoder's raw scores cluster in a narrow range (like 0.30-0.50). We apply **percentile-based calibration** to stretch them to [0, 0.95] while preserving relative ordering.

### JD Tier Weights (configurable via sliders)

| Tier | Default Weight | What it is |
|---|---|---|
| **must_have** | 1.00× | Non-negotiable requirements |
| **good_to_have** | 0.50× | Preferred but won't reject for |
| **bonus** | 0.25× | Cherry on top |
| **disqualifiers** | −0.15× | Red flags to subtract from score |

---

## Step 3: Penalty & Bonus Engine (`penalty.py`)

### How penalties work: Multiplicative stacking (configurable via sliders)

Each penalty is a **multiplier**. They stack by multiplication:
- One penalty of ×0.50 → score halved
- Two penalties of ×0.50 and ×0.70 → score × 0.50 × 0.70 = ×0.35

### The 9 Penalties

| # | Penalty | Default ×  | Trigger |
|---|---|---|---|
| P1 | Ghost Candidate | ×0.20 | recruiter_response_rate < 10% OR inactive |
| P2 | Role Mismatch | ×0.50 | Marketing/Sales title with high AI score |
| P3 | Job Hopper | ×0.60 | 5+ yrs exp but < 18 months per role |
| P4 | Coding Recency | ×0.70 | Senior with no GitHub AND no assessments |
| P5 | Consulting Lifer | ×0.65 | Entire career at TCS/Infosys/Wipro/etc. |
| P6 | Low Profile | ×0.80 | Profile completeness < 35% |
| P7 | CV/Speech Only | ×0.55 | CV/speech/robotics without NLP/IR |
| P8 | Pure Research | ×0.40 | Research keywords without production keywords |
| P9 | LangChain Only | ×0.45 | LangChain/OpenAI without pre-LLM ML |

### The 5 Bonuses

| Bonus | Value | Why |
|---|---|---|
| **Short Notice** | +0.05 | Notice period ≤ 30 days |
| **Tier-1 City** | +0.05 | Hyderabad, Pune, Mumbai, Delhi NCR, Bangalore |
| **GitHub Active** | +0.03 | Active GitHub profile |
| **Assessments Passed** | +0.02 | Platform-verified skills |
| **High Engagement** | +0.03 | Response rate ≥70% AND interview completion ≥70% |

### Final Score Formula
```
base_score  = semantic_score                        # from Step 2 (0 to 0.95)
penalized   = base_score × penalty1 × penalty2 ... # multiply all applicable penalties
final_score = clamp(penalized + bonus1 + bonus2 ..., 0, 0.99)
```

---

## Working on Any JD

The system is designed to work with **any job description**, not just the default one:

1. **Through the web UI:** Type or paste your requirements into the Must Have, Good to Have, Bonus, and Disqualifier fields. Each requirement should be a short, focused query (10-30 words works best).

2. **Through CLI with JSON:**
```bash
python run_pipeline.py --jd-json my_custom_jd.json
```

3. **Through CLI with .docx:**
```bash
python run_pipeline.py --jd my_job_description.docx
```

The pre-computed candidate embeddings are reused regardless of which JD you test. Only the Cross-Encoder reranking runs fresh.

---

## Working on Any Candidate Dataset

The system supports **Custom Datasets**. Instead of only ranking the default 100k candidates, you can bring your own data:

1. **Upload via Web UI**: Click "Upload" and provide a `.csv` (needs a `resume_text` column) or `.jsonl` file.
2. **Automatic Pipeline**: The `dataset_manager.py` kicks in. It parses the data, runs the Step 1 Preprocessing logic, calculates the neural embeddings for every candidate, and builds custom FAISS/BM25 indices on the fly.
3. **Dynamic Loading**: The `HybridSearchEngine` dynamically switches its context folder. It unloads the default indices and loads your custom team's indices into RAM.
4. **Score**: You can then run any JD against your private candidate pool!

---

## File Structure

```
resume-shortlister/
│
├── app.py                ←  Web frontend (Flask) — USE THIS
├── run_pipeline.py       ←  CLI pipeline runner
├── hybrid_search.py      ←  Semantic scoring engine (the brain)
├── penalty.py            ←  Business rule penalties (the filter)
├── dataset_manager.py    ←  Handles custom file uploads & indexing
├── default_jd.json       ←  Pre-loaded JD for testing
│
├── templates/index.html  ←  Web UI template
├── static/style.css      ←  Premium dark theme
├── static/app.js         ←  Frontend logic
│
├── candidates.jsonl      ←  Full dataset (~487 MB)
│
└── processed/            ←  Pre-computed indices & embeddings
    ├── text_corpus.pkl   ←  Cleaned candidate text
    ├── numeric_signals.pkl ← Numeric signals
    ├── bm25_index.pkl    ←  Cached BM25 index
    ├── faiss_hnsw.index  ←  Cached FAISS HNSW index
    ├── embeddings.npy    ←  Pre-computed embeddings
    └── final_ranking.csv ←  Last pipeline output
```

### Running it

```bash
# Web UI (recommended)
python app.py                     # starts at http://localhost:5000

# CLI
python run_pipeline.py --verbose  # uses default JD
python run_pipeline.py --jd-json my_jd.json  # custom JD
```
