"""
preprocessing.py
================
Subproblem 1 – Data Flattening & Preprocessing
================================================
Converts the nested, messy JSON/JSONL candidate data into two clean,
algorithm-ready artefacts:

  Task 1.1 – text_corpus.pkl / text_corpus.csv
      candidate_id  →  clean_text   (for Stage 1 BM25 / TF-IDF search)

  Task 1.2 – numeric_signals.pkl / numeric_signals.csv
      candidate_id  →  numeric features  (for Stage 2 ranking / scoring)

Run directly:
    python preprocessing.py                         # uses full candidates.jsonl
    python preprocessing.py --sample                # uses candidate.json (fast dev mode)
    python preprocessing.py --sample --verbose      # also prints intermediate info

Outputs land in ./processed/ by default (configurable via OUTPUT_DIR).
"""

from __future__ import annotations

import argparse
import json
import sys

# Ensure UTF-8 output even on Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "processed"

FULL_DATA_PATH   = ROOT_DIR / "candidates.jsonl"
SAMPLE_DATA_PATH = ROOT_DIR / "candidate.json"

# Numeric columns we want to pull directly from redrob_signals
# (excludes dates, booleans, nested dicts/strings – handled separately)
NUMERIC_SIGNAL_COLUMNS: list[str] = [
    "profile_completeness_score",       # 0–100
    "profile_views_received_30d",       # int ≥ 0
    "applications_submitted_30d",       # int ≥ 0
    "recruiter_response_rate",          # 0.0–1.0
    "avg_response_time_hours",          # float ≥ 0
    "connection_count",                 # int ≥ 0
    "endorsements_received",            # int ≥ 0
    "notice_period_days",               # int 0–180
    "github_activity_score",            # 0–100, or -1 → "not linked"
    "search_appearance_30d",            # int ≥ 0
    "saved_by_recruiters_30d",          # int ≥ 0
    "interview_completion_rate",        # 0.0–1.0
    "offer_acceptance_rate",            # 0.0–1.0, or -1 → "no history"
]

# Sentinel values that indicate "not available / not linked"
SENTINEL_VALUES: dict[str, float] = {
    "github_activity_score": -1.0,
    "offer_acceptance_rate": -1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(*, use_sample: bool = False) -> list[dict[str, Any]]:
    """
    Load candidates from either:
      • sample_candidates.json  (JSON array, fast for dev)
      • candidates.jsonl        (one JSON object per line, full dataset)
    Returns a plain Python list of dicts.
    """
    if use_sample:
        path = SAMPLE_DATA_PATH
        print(f"[load] Reading sample data from {path.name} …")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        path = FULL_DATA_PATH
        print(f"[load] Streaming full JSONL from {path.name} …")
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


# ─────────────────────────────────────────────────────────────────────────────
# Task 1.1 – Text Aggregation
# ─────────────────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]")   # strip punctuation (keeps alphanumerics + spaces)


def _clean_text(raw: str) -> str:
    """
    Clean a raw text string:
      1. Unicode normalise to ASCII (handles \u2014 em-dash, smart quotes, etc.)
      2. Strip residual punctuation (keeps alphanumerics + whitespace)
      3. Collapse multiple spaces / newlines into a single space
      4. Strip leading/trailing whitespace
    """
    # Normalise unicode → closest ASCII (e.g. \u2014 → -, \u2019 → ')
    text = unicodedata.normalize("NFKD", raw)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Remove punctuation
    text = _PUNCT_RE.sub(" ", text)

    # Collapse whitespace
    text = " ".join(text.split())

    return text.strip()


def build_text_corpus(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Task 1.1 – Text Aggregation.

    For each candidate concatenates (in order):
        1. profile.headline
        2. profile.summary
        3. profile.current_title, current_company, current_industry, location
        4. career_history[*].title + company + description
        5. education[*].degree + field_of_study + institution
        6. skills[*].name (+ proficiency level)
        7. certifications[*].name + issuer
        8. languages[*].language (+ proficiency level)

    Returns a DataFrame with columns:
        candidate_id | clean_text
    """
    rows = []
    for c in candidates:
        cid     = c["candidate_id"]
        profile = c.get("profile", {})

        # Collect text pieces (guard against None / missing keys)
        parts: list[str] = []

        # ── Profile core text ─────────────────────────────────────────────
        headline = profile.get("headline") or ""
        summary  = profile.get("summary")  or ""
        parts.append(headline)
        parts.append(summary)

        # Profile metadata (current role context)
        for field in ("current_title", "current_company", "current_industry", "location"):
            val = profile.get(field) or ""
            if val:
                parts.append(val)

        # ── Career history (title + company + description) ────────────────
        for job in c.get("career_history", []):
            title = job.get("title") or ""
            company = job.get("company") or ""
            desc = job.get("description") or ""
            if title:
                parts.append(title)
            if company:
                parts.append(company)
            if desc:
                parts.append(desc)

        # ── Education (degree + field_of_study + institution) ─────────────
        for edu in c.get("education", []):
            degree = edu.get("degree") or ""
            field  = edu.get("field_of_study") or ""
            inst   = edu.get("institution") or ""
            if degree:
                parts.append(degree)
            if field:
                parts.append(field)
            if inst:
                parts.append(inst)

        # ── Skills (name + proficiency) ───────────────────────────────────
        for skill in c.get("skills", []):
            name = skill.get("name") or ""
            prof = skill.get("proficiency") or ""
            if name:
                parts.append(f"{name} {prof}" if prof else name)

        # ── Certifications (name + issuer) ────────────────────────────────
        for cert in c.get("certifications", []):
            cert_name = cert.get("name") or ""
            issuer    = cert.get("issuer") or ""
            if cert_name:
                parts.append(f"{cert_name} {issuer}" if issuer else cert_name)

        # ── Languages (language + proficiency) ────────────────────────────
        for lang in c.get("languages", []):
            lang_name = lang.get("language") or ""
            lang_prof = lang.get("proficiency") or ""
            if lang_name:
                parts.append(f"{lang_name} {lang_prof}" if lang_prof else lang_name)

        # Join, then clean
        raw_blob  = " ".join(parts)
        clean_blob = _clean_text(raw_blob)

        rows.append({"candidate_id": cid, "clean_text": clean_blob})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1.2 – Numeric Signal Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_numeric_signals(signals: dict[str, Any]) -> dict[str, Any]:
    """
    Pull numeric signals from a single candidate's redrob_signals dict.

    Special handling:
      • github_activity_score == -1  → set to 0, flag has_github_linked = 0
      • offer_acceptance_rate  == -1 → set to NaN (no offer history), flag has_offer_history = 0
      • expected_salary_range  → expand into salary_min_lpa, salary_max_lpa, salary_mid_lpa
      • skill_assessment_scores → mean score across all assessments (NaN if none)
      • boolean flags: open_to_work, willing_to_relocate, verified_email,
                       verified_phone, linkedin_connected  → cast to int (0/1)
    """
    row: dict[str, Any] = {}

    # ── Core numeric columns ──────────────────────────────────────────────────
    for col in NUMERIC_SIGNAL_COLUMNS:
        val = signals.get(col)

        # Handle sentinel -1 values
        if col in SENTINEL_VALUES and val == SENTINEL_VALUES[col]:
            if col == "github_activity_score":
                row[col]                 = 0.0
                row["has_github_linked"] = 0
            elif col == "offer_acceptance_rate":
                row[col]                  = float("nan")
                row["has_offer_history"]  = 0
        else:
            row[col] = float(val) if val is not None else float("nan")
            if col == "github_activity_score":
                row["has_github_linked"] = 1
            if col == "offer_acceptance_rate":
                row["has_offer_history"] = 1

    # ── Salary range ─────────────────────────────────────────────────────────
    salary = signals.get("expected_salary_range_inr_lpa") or {}
    sal_min = salary.get("min", float("nan"))
    sal_max = salary.get("max", float("nan"))
    row["salary_min_lpa"] = float(sal_min) if sal_min is not None else float("nan")
    row["salary_max_lpa"] = float(sal_max) if sal_max is not None else float("nan")
    row["salary_mid_lpa"] = (row["salary_min_lpa"] + row["salary_max_lpa"]) / 2.0

    # ── Skill assessment mean ─────────────────────────────────────────────────
    skill_scores = signals.get("skill_assessment_scores") or {}
    if skill_scores:
        scores = [v for v in skill_scores.values() if isinstance(v, (int, float))]
        row["skill_assessment_mean"] = sum(scores) / len(scores) if scores else float("nan")
        row["num_assessments_taken"] = len(scores)
    else:
        row["skill_assessment_mean"] = float("nan")
        row["num_assessments_taken"] = 0

    # ── Boolean flags → int ───────────────────────────────────────────────────
    for bool_col in (
        "open_to_work_flag",
        "willing_to_relocate",
        "verified_email",
        "verified_phone",
        "linkedin_connected",
    ):
        val = signals.get(bool_col)
        row[bool_col] = int(bool(val)) if val is not None else 0

    # ── Preferred work mode (one-hot) ─────────────────────────────────────────
    work_mode = (signals.get("preferred_work_mode") or "").lower().strip()
    for mode in ("remote", "hybrid", "onsite", "flexible"):
        row[f"work_mode_{mode}"] = 1 if work_mode == mode else 0

    return row


def build_numeric_signals(
    candidates: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Task 1.2 – Numeric Signal Extraction.

    Returns a tidy DataFrame where each row = one candidate with all
    numeric / boolean / derived numeric columns ready for scoring models.

    Columns include:
        candidate_id
        years_of_experience       ← from profile (not signals)
        <all NUMERIC_SIGNAL_COLUMNS with sentinel handling>
        has_github_linked         ← boolean flag derived from github_activity_score
        has_offer_history         ← boolean flag derived from offer_acceptance_rate
        salary_min_lpa, salary_max_lpa, salary_mid_lpa
        skill_assessment_mean, num_assessments_taken
        open_to_work_flag, willing_to_relocate, verified_email,
        verified_phone, linkedin_connected
        work_mode_remote, work_mode_hybrid, work_mode_onsite, work_mode_flexible
    """
    rows = []
    for c in candidates:
        cid     = c["candidate_id"]
        signals = c.get("redrob_signals", {})
        profile = c.get("profile", {})

        row = {"candidate_id": cid}

        # years_of_experience lives in profile, not signals
        yoe = profile.get("years_of_experience")
        row["years_of_experience"] = float(yoe) if yoe is not None else float("nan")

        row.update(_extract_numeric_signals(signals))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Enforce candidate_id as first column
    cols = ["candidate_id"] + [c for c in df.columns if c != "candidate_id"]
    df = df[cols].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(*, use_sample: bool = False, verbose: bool = False) -> None:
    """End-to-end preprocessing pipeline. Saves outputs to OUTPUT_DIR."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    candidates = load_candidates(use_sample=use_sample)
    print(f"[load] {len(candidates):,} candidates loaded.\n")

    # ── 2. Task 1.1 – Text corpus ─────────────────────────────────────────────
    print("── Task 1.1: Building text corpus …")
    text_df = build_text_corpus(candidates)

    if verbose:
        print(f"   Shape: {text_df.shape}")
        print("   Sample (first 2 rows):")
        for _, row in text_df.head(2).iterrows():
            print(f"   [{row['candidate_id']}]  {row['clean_text'][:120]} …")

    text_df.to_csv(OUTPUT_DIR / "text_corpus.csv", index=False)
    text_df.to_pickle(OUTPUT_DIR / "text_corpus.pkl")
    print(f"   ✓ Saved → {OUTPUT_DIR / 'text_corpus.csv'}")
    print(f"   ✓ Saved → {OUTPUT_DIR / 'text_corpus.pkl'}\n")

    # ── 3. Task 1.2 – Numeric signals ─────────────────────────────────────────
    print("── Task 1.2: Extracting numeric signals …")
    num_df = build_numeric_signals(candidates)

    if verbose:
        print(f"   Shape: {num_df.shape}")
        print(f"   Columns ({len(num_df.columns)}): {num_df.columns.tolist()}")
        print("\n   Descriptive stats (numeric columns):")
        numeric_cols = num_df.select_dtypes(include="number").columns.tolist()
        print(num_df[numeric_cols].describe().round(3).to_string())

        # Show github flag distribution
        print("\n   has_github_linked value counts:")
        print(num_df["has_github_linked"].value_counts().to_string())

        print("\n   has_offer_history value counts:")
        print(num_df["has_offer_history"].value_counts().to_string())

    num_df.to_csv(OUTPUT_DIR / "numeric_signals.csv", index=False)
    num_df.to_pickle(OUTPUT_DIR / "numeric_signals.pkl")
    print(f"   ✓ Saved → {OUTPUT_DIR / 'numeric_signals.csv'}")
    print(f"   ✓ Saved → {OUTPUT_DIR / 'numeric_signals.pkl'}\n")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Preprocessing complete.")
    print(f"  Text corpus shape   : {text_df.shape}")
    print(f"  Numeric signals shape: {num_df.shape}")
    print(f"  Output directory     : {OUTPUT_DIR}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subproblem 1 – Data Flattening & Preprocessing for Resume Shortlister"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample_candidates.json instead of the full candidates.jsonl (fast dev mode).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed intermediate output (stats, sample rows, etc.).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(use_sample=args.sample, verbose=args.verbose)
