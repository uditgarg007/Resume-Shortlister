# Resume Shortlister — Intelligent Candidate Discovery & Ranking

> **India Runs Data & AI Challenge**
> Subproblem 1: Data Flattening & Preprocessing Pipeline

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset Files](#dataset-files)
4. [Prerequisites](#prerequisites)
5. [How to Run](#how-to-run)
6. [What the Pipeline Does](#what-the-pipeline-does)
   - [Task 1.1 — Text Aggregation](#task-11--text-aggregation-stage-1-search)
   - [Task 1.2 — Numeric Signal Extraction](#task-12--numeric-signal-extraction-stage-2-ranking)
7. [Output Files](#output-files)
8. [Column Reference](#column-reference)
9. [Sentinel / Default Flag Handling](#sentinel--default-flag-handling)
10. [Using the Outputs in Downstream Code](#using-the-outputs-in-downstream-code)

---

## Project Overview

This project processes a large dataset of candidate profiles (sourced from the **Redrob** platform) and converts the raw, deeply nested JSON records into two clean, algorithm-ready tables:

| Output                      | Used For |
|---                          |                    ---|
| `processed/text_corpus.csv` | Stage 1 — BM25 / TF-IDF full-text search |
| `processed/numeric_signals.csv` | Stage 2 — Numeric scoring & ranking model |

The preprocessing handles messy real-world data problems: escape characters, Unicode symbols, sentinel `-1` flags, nested salary objects, variable-length skill assessment dicts, and boolean-to-integer conversions.

---

## Repository Structure

```
Resume-Shortlister/
│
├── preprocessing.py          ← Main preprocessing pipeline (Subproblem 1)
├── data_analysis.py          ← Exploratory data analysis helpers
├── data_analysis.ipynb       ← Jupyter notebook for EDA
│
├── candidates.jsonl          ← Full dataset (~487 MB, one JSON object per line)
├── sample_candidates.json    ← 50-candidate sample (fast dev/testing)
├── candidate_schema.json     ← JSON Schema defining all fields
├── sample_submission.csv     ← Expected submission format
│
├── README.md                 ← This file
│
└── processed/                ← Auto-created by preprocessing.py
    ├── text_corpus.csv
    ├── text_corpus.pkl
    ├── numeric_signals.csv
    └── numeric_signals.pkl
```

---

## Dataset Files

| File | Description |
|---|---|
| `candidates.jsonl` | Full candidate dataset. Each line is a JSON object with fields: `candidate_id`, `profile`, `career_history`, `education`, `skills`, `certifications`, `languages`, `redrob_signals` |
| `sample_candidates.json` | A JSON array of 50 candidates. Identical schema to the full dataset. Use this for fast iteration. |
| `candidate_schema.json` | Full JSON Schema (`draft-07`) documenting every field, type, enum, and constraint. |
| `sample_submission.csv` | Shows the expected output format for the final ranking submission. |

---

## Prerequisites

**Python version:** 3.9 or higher (3.10+ recommended)

Install dependencies:

```bash
pip install pandas
```

> `json`, `re`, `unicodedata`, `argparse`, `pathlib` are all part of the Python standard library — no extra install needed.

---

## How to Run

All commands should be run from the **root of the repository** (the `Resume-Shortlister/` folder).

### 1. Fast dev mode — 50 sample candidates

```bash
python preprocessing.py --sample
```

Runs in ~2 seconds. Good for testing and iteration.

### 2. Fast dev mode + detailed output

```bash
python preprocessing.py --sample --verbose
```

Prints:
- Shape of each output DataFrame
- 2 sample rows of the text corpus
- All 31 column names of the numeric signals table
- Full `describe()` stats for every numeric column
- Value counts for `has_github_linked` and `has_offer_history`

### 3. Full dataset (production run)

```bash
python preprocessing.py
```

Streams `candidates.jsonl` line-by-line (memory-efficient). Depending on your machine, this may take a few minutes for the full ~487 MB file.

### 4. Full dataset + verbose output

```bash
python preprocessing.py --verbose
```

### CLI flags summary

| Flag | Description |
|---|---|
| `--sample` | Use `sample_candidates.json` instead of `candidates.jsonl` |
| `--verbose` | Print detailed stats, column names, and sample rows to stdout |

---

## What the Pipeline Does

### Task 1.1 — Text Aggregation (Stage 1 Search)

**Goal:** Create a single, clean text string per candidate for full-text search (BM25, TF-IDF, or embeddings).

**Fields joined (in order):**
1. `profile.headline` — one-line professional title
2. `profile.summary` — multi-sentence professional bio
3. `career_history[*].description` — all job role descriptions (concatenated)

**Cleaning steps applied:**
1. **Unicode normalisation (NFKD → ASCII)** — strips escape characters like `\u2014` (em-dash), `\u2019` (curly apostrophe), `\u00e9` (é), etc.
2. **Punctuation removal** — regex strips everything that isn't alphanumeric or whitespace
3. **Whitespace collapse** — multiple spaces, tabs, and newlines are reduced to a single space
4. **Strip** — removes any leading/trailing whitespace

**Example transformation:**

| Before | After |
|---|---|
| `"Senior Engineer — ML & AI\u2014 building..."` | `"Senior Engineer ML AI building"` |
| `"Python, TensorFlow, & PyTorch"` | `"Python TensorFlow PyTorch"` |

**Output:** `processed/text_corpus.csv` — 2 columns: `candidate_id`, `clean_text`

---

### Task 1.2 — Numeric Signal Extraction (Stage 2 Ranking)

**Goal:** Flatten the `redrob_signals` nested object into a wide, numeric DataFrame ready for scoring algorithms.

**Source fields processed:**

| Source | Fields |
|---|---|
| `profile` | `years_of_experience` |
| `redrob_signals` | 13 core numeric fields (see Column Reference) |
| `redrob_signals.expected_salary_range_inr_lpa` | Expanded to `salary_min_lpa`, `salary_max_lpa`, `salary_mid_lpa` |
| `redrob_signals.skill_assessment_scores` | Collapsed to `skill_assessment_mean`, `num_assessments_taken` |
| `redrob_signals` (booleans) | 5 boolean flags cast to int (0/1) |
| `redrob_signals.preferred_work_mode` | One-hot encoded into 4 columns |

**Output:** `processed/numeric_signals.csv` — 31 columns, one row per candidate

---

## Output Files

All outputs are saved to the `processed/` directory (created automatically).

| File | Format | Description |
|---|---|---|
| `text_corpus.csv` | CSV | `candidate_id`, `clean_text` |
| `text_corpus.pkl` | Pickle | Same content, faster to load in Python |
| `numeric_signals.csv` | CSV | 31 numeric/binary columns per candidate |
| `numeric_signals.pkl` | Pickle | Same content, preserves `NaN` correctly (recommended) |

> **Tip:** Load `.pkl` files for downstream Python work — they preserve `float('nan')` exactly, whereas CSV will encode NaN as empty string.

---

## Column Reference

### `text_corpus.csv`

| Column | Type | Description |
|---|---|---|
| `candidate_id` | str | Unique ID, format `CAND_XXXXXXX` |
| `clean_text` | str | Cleaned, concatenated text blob for search indexing |

---

### `numeric_signals.csv` (31 columns)

#### Identity
| Column | Type | Range | Description |
|---|---|---|---|
| `candidate_id` | str | — | Unique candidate identifier |

#### From `profile`
| Column | Type | Range | Description |
|---|---|---|---|
| `years_of_experience` | float | 0–50 | Total years of professional experience |

#### Core Redrob Signals
| Column | Type | Range | Description |
|---|---|---|---|
| `profile_completeness_score` | float | 0–100 | % of profile fields filled |
| `profile_views_received_30d` | float | ≥ 0 | Profile views in last 30 days |
| `applications_submitted_30d` | float | ≥ 0 | Applications submitted in last 30 days |
| `recruiter_response_rate` | float | 0.0–1.0 | Fraction of recruiter messages responded to |
| `avg_response_time_hours` | float | ≥ 0 | Average response time to recruiter messages |
| `connection_count` | float | ≥ 0 | Total network connections |
| `endorsements_received` | float | ≥ 0 | Total endorsements received on skills |
| `notice_period_days` | float | 0–180 | Notice period required before joining |
| `github_activity_score` | float | 0–100 | Commits/PRs/stars score (0 if not linked) |
| `search_appearance_30d` | float | ≥ 0 | Times profile appeared in recruiter searches |
| `saved_by_recruiters_30d` | float | ≥ 0 | Times profile was saved by recruiters |
| `interview_completion_rate` | float | 0.0–1.0 | Fraction of scheduled interviews attended |
| `offer_acceptance_rate` | float | 0.0–1.0 or NaN | Historical offer acceptance rate |

#### Sentinel-Derived Flags
| Column | Type | Values | Description |
|---|---|---|---|
| `has_github_linked` | int | 0 / 1 | 0 = no GitHub account linked |
| `has_offer_history` | int | 0 / 1 | 0 = candidate has never received an offer |

#### Salary (expanded from nested object)
| Column | Type | Description |
|---|---|---|
| `salary_min_lpa` | float | Minimum expected salary (INR Lakhs Per Annum) |
| `salary_max_lpa` | float | Maximum expected salary (INR Lakhs Per Annum) |
| `salary_mid_lpa` | float | Midpoint = (min + max) / 2 |

#### Skill Assessments (collapsed from dict)
| Column | Type | Description |
|---|---|---|
| `skill_assessment_mean` | float | Average score across all completed assessments (NaN if none taken) |
| `num_assessments_taken` | int | Count of distinct skill assessments completed |

#### Boolean Flags (cast to int)
| Column | Type | Values | Description |
|---|---|---|---|
| `open_to_work_flag` | int | 0 / 1 | Candidate is actively open to opportunities |
| `willing_to_relocate` | int | 0 / 1 | Willing to relocate for a role |
| `verified_email` | int | 0 / 1 | Email address verified on platform |
| `verified_phone` | int | 0 / 1 | Phone number verified on platform |
| `linkedin_connected` | int | 0 / 1 | LinkedIn account linked to profile |

#### Work Mode (one-hot encoded)
| Column | Type | Description |
|---|---|---|
| `work_mode_remote` | int | Prefers fully remote work |
| `work_mode_hybrid` | int | Prefers hybrid work |
| `work_mode_onsite` | int | Prefers onsite/in-office work |
| `work_mode_flexible` | int | No preference / flexible |

---

## Sentinel / Default Flag Handling

Two fields in `redrob_signals` use `-1` as a sentinel meaning "data not available":

| Field | Sentinel Value | Meaning | Treatment in output |
|---|---|---|---|
| `github_activity_score` | `-1` | No GitHub account linked | Score set to `0.0`; `has_github_linked = 0` |
| `offer_acceptance_rate` | `-1` | No offer history exists | Rate set to `NaN`; `has_offer_history = 0` |

This design separates the **signal value** from the **availability flag**, allowing downstream models to handle these two pieces of information independently (e.g., impute differently, or use the flag as a feature itself).

---

## Using the Outputs in Downstream Code

```python
import pandas as pd

# Load text corpus (for BM25 / TF-IDF indexing)
text_df = pd.read_pickle("processed/text_corpus.pkl")
# text_df.columns → ['candidate_id', 'clean_text']

# Load numeric signals (for scoring / ranking model)
num_df = pd.read_pickle("processed/numeric_signals.pkl")
# num_df.shape → (N_candidates, 31)

# Example: filter candidates who have GitHub linked
github_active = num_df[num_df["has_github_linked"] == 1]

# Example: impute missing offer_acceptance_rate with column mean
mean_oar = num_df["offer_acceptance_rate"].mean()
num_df["offer_acceptance_rate"].fillna(mean_oar, inplace=True)

# Example: merge both tables on candidate_id
merged = text_df.merge(num_df, on="candidate_id")
```
