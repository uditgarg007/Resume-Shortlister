# Stage 1 — Data Preprocessing
### Resume Shortlister · India Runs Data & AI Hackathon

This branch contains **Subproblem 1**: converting 100,000 raw nested JSON candidate profiles into clean, algorithm-ready artifacts.

---

## What This Stage Produces

| Output File | Description |
|---|---|
| `processed/text_corpus.pkl` | One clean text blob per candidate (for BM25 + vector search) |
| `processed/numeric_signals.pkl` | 31 numeric/boolean features per candidate (for penalty scoring) |

---

## How to Run

```bash
pip install pandas numpy

# Fast dev mode — 50 candidates from candidate.json
python preprocessing.py --sample --verbose

# Full dataset — 100,000 candidates from candidates.jsonl
python preprocessing.py
```

---

## Text Aggregation (Task 1.1)

Each candidate's text is built by concatenating (in order):

1. `profile.headline` + `profile.summary`
2. `profile.current_title`, `current_company`, `current_industry`, `location`
3. `career_history[*].title` + `.company` + `.description`
4. `education[*].degree` + `.field_of_study` + `.institution`
5. `skills[*].name` + `.proficiency`
6. `certifications[*].name` + `.issuer`
7. `languages[*].language` + `.proficiency`

Then cleaned: unicode-normalized → punctuation stripped → whitespace collapsed.

---

## Numeric Signal Extraction (Task 1.2)

Extracts **31 features** from `redrob_signals`:

- Core metrics: `profile_completeness_score`, `github_activity_score`, `recruiter_response_rate`, etc.
- Sentinel handling: `github_activity_score == -1` → `has_github_linked = 0`
- Derived: `salary_mid_lpa`, `skill_assessment_mean`, work mode one-hot encoding
- Boolean flags: `open_to_work_flag`, `verified_email`, `linkedin_connected`

---

## Files in This Branch

```
preprocessing.py      ← Core preprocessing logic
candidate.json        ← 50-candidate sample dataset
candidate_schema.json ← Full JSON schema reference
requirements.txt      ← Dependencies
.gitignore
```

---

## Next Stage → [SemanticSearch](../tree/SemanticSearch)

The output `processed/text_corpus.pkl` feeds directly into the hybrid BM25 + HNSW vector search engine.
