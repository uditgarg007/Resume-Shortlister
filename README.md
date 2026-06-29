# Stage 2 — Hybrid Semantic Search & Cross-Encoder Reranking
### Resume Shortlister · India Runs Data & AI Hackathon

This branch contains **Stage 2**: finding the best-matching candidates from 100,000 profiles using a two-stage retrieval + reranking pipeline.

---

## What This Stage Does

Given preprocessed candidate embeddings + any job description, produces `semantic_score` ∈ [0.30, 0.95] per candidate.

### Stage 2a — Hybrid Retrieval (fast, broad)
- **BM25** keyword search (20% weight) over the full text corpus
- **FAISS HNSW** vector search (80% weight) using `all-MiniLM-L6-v2` embeddings
- Fused via **Reciprocal Rank Fusion (RRF)** → Top 200 candidates

### Stage 2b — Cross-Encoder Reranking (deep, narrow)
- `ms-marco-MiniLM-L-6-v2` reads (query, candidate_text) pairs jointly through all transformer layers
- **Sub-query decomposition**: each JD tier split into 3–5 focused 10–30-word queries for maximum Cross-Encoder accuracy
- **Chunk-and-max-pool**: long resumes split into overlapping 200-token chunks; final score = MAX across chunks

---

## How to Run

```bash
pip install -r requirements.txt

# 1. First run preprocessing to generate processed/ artifacts
python preprocessing.py

# 2. Run hybrid search with default JD
python hybrid_search.py --verbose

# 3. Custom JD from JSON
python hybrid_search.py --jd-json my_jd.json --top-k 50
```

---

## JD Tier Format

```json
{
  "must_have":    ["Built production embedding retrieval systems using FAISS"],
  "good_to_have": ["LLM fine-tuning experience LoRA QLoRA PEFT"],
  "bonus":        ["Shipped ranking system to real users at scale"],
  "disqualifiers":["Career entirely in academic research without production"]
}
```

**Tier weights:** `must_have × 1.0` + `good_to_have × 0.5` + `bonus × 0.25` − `disqualifier penalty`

---

## Output

`processed/semantic_scores.csv`

| candidate_id | semantic_score | rrf_score |
|---|---|---|
| CAND_0057701 | 0.9500 | 0.0121 |
| CAND_0053695 | 0.8799 | 0.0098 |
| ... | ... | ... |

---

## Files in This Branch

```
hybrid_search.py      ← BM25 + HNSW + Cross-Encoder (⭐ main module)
preprocessing.py      ← Required: generates processed/ directory
default_jd.json       ← Default JD (Redrob Senior AI Engineer)
candidate.json        ← 50-candidate sample for quick testing
candidate_schema.json ← JSON schema reference
requirements.txt
.gitignore
```

---

## Previous Stage → [preprocessing](../tree/preprocessing)  
## Next Stage → [finalRanker](../tree/finalRanker)
