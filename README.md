# Stage 3 — Final Ranking: Penalty & Bonus Engine
### Resume Shortlister · India Runs Data & AI Hackathon

This branch contains the **complete CLI pipeline**: preprocessing → hybrid semantic scoring → business-rule penalty/bonus engine → final ranked output.

---

## What This Stage Does

Takes the semantic scores from Stage 2 and applies **9 multiplicative penalties** and **5 additive bonuses** based on real recruiting signals, producing a final calibrated score per candidate.

```
final_score = semantic_score × penalty_multiplier + bonus_total
```

---

## How to Run

```bash
pip install -r requirements.txt

# Full pipeline with default JD (Redrob AI Engineer)
python run_pipeline.py --verbose

# Custom JD
python run_pipeline.py --jd-json my_jd.json --top-k 50

# Force rebuild indices
python run_pipeline.py --rebuild
```

> **First run:** downloads transformer models (~80 MB) and preprocesses candidates.  
> **Subsequent runs:** loads cached indices and completes in ~30 seconds.

---

## Penalty Rules

| Code | Condition | Multiplier |
|---|---|---|
| `GHOST` | Recruiter response rate < 20% | ×0.20 |
| `MISMATCH` | Current title unrelated to JD | ×0.50 |
| `HOPPER` | Avg tenure < 12 months | ×0.60 |
| `NO_CODE` | No GitHub + zero assessments | ×0.70 |
| `OVERQUAL` | YOE > 15 and applying for junior | ×0.75 |
| `RELO_RISK` | Unwilling to relocate, remote-only | ×0.80 |
| `NOTICE` | Notice period > 90 days | ×0.85 |
| `CONSULTING` | Entire career at service firms | ×0.85 |
| `UNVERIFIED` | No verified email or phone | ×0.90 |

Penalties **compound multiplicatively**: a candidate matching two rules gets both applied.

---

## Bonus Rules

| Code | Condition | Points |
|---|---|---|
| `SHORT_NOTICE` | Notice period ≤ 15 days | +0.05 |
| `TIER1_CITY` | Located in Tier-1 city | +0.05 |
| `GITHUB` | GitHub activity score > 70 | +0.03 |
| `ASSESSED` | Skill assessment mean > 75% | +0.02 |
| `ENGAGED` | High platform engagement score | +0.03 |

Bonuses use **exponential diminishing returns** to prevent artificial score caps.

---

## Output

`processed/final_ranking.csv`

```
rank  candidate_id   final_score  semantic_score  penalty_mult  bonus   flags
   1  CAND_0057701        0.9686          0.9500         ×1.00  +0.05  GITHUB(+0.03) ASSESSED(+0.02)
   2  CAND_0053695        0.9683          0.8799         ×1.00  +0.13  SHORT_NOTICE TIER1_CITY GITHUB
   3  CAND_0037944        0.9648          0.8255         ×1.00  +0.15  ...
```

---

## Files in This Branch

```
run_pipeline.py       ← End-to-end orchestrator (⭐ start here)
penalty.py            ← 9 penalties + 5 bonuses engine
hybrid_search.py      ← BM25 + HNSW + Cross-Encoder
preprocessing.py      ← Text & numeric feature extraction
default_jd.json       ← Default JD config
candidate.json        ← 50-candidate sample
candidate_schema.json ← JSON schema reference
requirements.txt
.gitignore
```

---

## Complete App → [Main](../tree/Main)

For the full web interface with real-time sliders, see the **Main** branch.
