"""
penalty.py — Subproblem 3: Business-Rule Penalty & Bonus Engine

Takes the semantic_scores (from hybrid_search.py) and numeric_signals (from
preprocessing.py), applies JD-driven business rules, and produces the final
ranked output.

Penalties (multiplicative — stack via multiplication):
  1. Ghost Candidate     ×0.20  — recruiter_response_rate < 10% OR inactive proxy
  2. Role Mismatch       ×0.50  — non-tech title with high semantic score
  3. Job Hopper          ×0.60  — 5+ yrs exp but avg < 18 months per role
  4. Coding Recency      ×0.70  — senior (6+ yrs) with no GitHub AND no assessments
  5. Consulting Lifer    ×0.65  — entire career at TCS/Infosys/Wipro/etc.
  6. Low Engagement      ×0.80  — very low profile completeness (<35%)

Bonuses (additive — capped so final stays in [0, 1]):
  1. Short Notice        +0.05  — notice_period_days <= 30
  2. Tier-1 City         +0.05  — Hyderabad, Pune, Mumbai, Delhi, Noida, Gurgaon, Bangalore
  3. GitHub Active       +0.03  — github_activity_score >= 30
  4. Assessment Proven   +0.02  — skill_assessment_mean >= 60 AND num_assessments >= 3
  5. High Engagement     +0.03  — recruiter_response_rate >= 0.70 AND interview_completion >= 0.70

Final formula:
  base_score  = semantic_score  (already [0.15, 0.95])
  penalized   = base_score × Π(applicable penalties)
  effective_bonus = room × (1 - e^(-bonus_total / 0.08))  [diminishing returns]
  final_score = clamp(penalized + effective_bonus, 0, 0.99)

Usage:
  python penalty.py                   # default: uses processed/ outputs
  python penalty.py --verbose         # print per-candidate penalty breakdown
  python penalty.py --output results/final_ranking.csv
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

SEMANTIC_SCORES_PATH = PROCESSED_DIR / "semantic_scores.pkl"
NUMERIC_SIGNALS_PATH = PROCESSED_DIR / "numeric_signals.pkl"
TEXT_CORPUS_PATH      = PROCESSED_DIR / "text_corpus.pkl"

OUTPUT_CSV_PATH = PROCESSED_DIR / "final_ranking.csv"
OUTPUT_PKL_PATH = PROCESSED_DIR / "final_ranking.pkl"

# ---------------------------------------------------------------------------
# Penalty & Bonus Constants
# ---------------------------------------------------------------------------

# Penalty 1: Ghost candidate
GHOST_RESPONSE_THRESHOLD    = 0.10   # recruiter_response_rate below this
GHOST_ACTIVITY_THRESHOLD    = 5.0    # combined 30d activity proxy below this
GHOST_PENALTY               = 0.20   # multiply score by this

# Penalty 2: Role mismatch (non-tech title + high semantic score)
MISMATCH_TITLES = {
    "marketing", "sales", "support", "operations", "accountant",
    "hr manager", "graphic designer", "content writer", "civil engineer",
    "mechanical engineer", "project manager",
}
MISMATCH_SEMANTIC_THRESHOLD = 0.40   # penalize if score is suspiciously high (raised for new calibration floor)
MISMATCH_PENALTY            = 0.50   # halve their score

# Penalty 3: Job hopper
HOPPER_EXP_THRESHOLD        = 5.0    # years_of_experience > this
HOPPER_MONTHS_THRESHOLD     = 18     # avg_months_per_role < this
HOPPER_PENALTY              = 0.60

# Penalty 4: Coding recency (senior + no proof of hands-on)
CODING_EXP_THRESHOLD        = 6.0    # "senior" = 6+ years
CODING_PENALTY              = 0.70

# Penalty 5: Consulting lifer — entire career at services firms
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "ltimindtree",
}
CONSULTING_PENALTY = 0.65

# Penalty 6: Low profile engagement
LOW_COMPLETENESS_THRESHOLD  = 35.0
LOW_COMPLETENESS_PENALTY    = 0.80

# Penalty 7: CV/Speech/Robotics specialist without NLP/IR background
# JD: "Primary expertise is in computer vision, speech, or robotics
#      without meaningful NLP or information retrieval exposure"
CV_SPEECH_KEYWORDS = {
    "computer vision", "image recognition", "object detection", "opencv",
    "speech processing", "speech recognition", "robotics", "autonomous",
    "image segmentation", "yolo", "cnn for images",
}
NLP_IR_KEYWORDS = {
    "nlp", "natural language", "information retrieval", "search ranking",
    "text mining", "embeddings", "transformer", "bert", "retrieval",
    "recommendation", "ranking system",
}
CV_SPEECH_PENALTY = 0.55

# Penalty 8: Pure research background — no production deployment signals
# JD: "Career spent entirely in pure research or academic lab environments"
RESEARCH_KEYWORDS = {
    "research scientist", "research fellow", "postdoc", "phd researcher",
    "academic", "research assistant", "lab assistant", "research intern",
    "published papers", "research lab",
}
PRODUCTION_KEYWORDS = {
    "production", "deployed", "shipped", "api", "microservice", "devops",
    "ci cd", "docker", "kubernetes", "aws", "gcp", "azure", "scaling",
    "load balancing", "monitoring", "sre",
}
RESEARCH_ONLY_PENALTY = 0.40

# Penalty 9: LangChain-only / API-wrapper experience
# JD: "AI experience consists primarily of calling OpenAI APIs via LangChain"
LANGCHAIN_KEYWORDS = {
    "langchain", "llamaindex", "openai api", "chatgpt api", "gpt wrapper",
    "prompt engineering", "ai chatbot using",
}
PRE_LLM_KEYWORDS = {
    "scikit", "sklearn", "xgboost", "tensorflow", "pytorch", "keras",
    "feature engineering", "model training", "hyperparameter",
    "gradient descent", "random forest", "neural network",
    "deep learning", "machine learning pipeline",
}
LANGCHAIN_ONLY_PENALTY = 0.45


# Bonus 1: Short notice period
SHORT_NOTICE_DAYS   = 30
SHORT_NOTICE_BONUS  = 0.05

# Bonus 2: Tier-1 cities (normalized for matching against clean_text)
TIER1_CITIES = {
    "hyderabad", "pune", "mumbai", "delhi", "noida", "gurgaon", "gurugram",
    "bangalore", "bengaluru",
}
CITY_BONUS = 0.05

# Bonus 3: GitHub active
GITHUB_ACTIVE_THRESHOLD = 30.0
GITHUB_BONUS            = 0.03

# Bonus 4: Assessment proven
ASSESSMENT_SCORE_THRESHOLD = 60.0
ASSESSMENT_COUNT_THRESHOLD = 3
ASSESSMENT_BONUS           = 0.02

# Bonus 5: High engagement (responsive + reliable)
HIGH_RESPONSE_THRESHOLD    = 0.70
HIGH_INTERVIEW_THRESHOLD   = 0.70
ENGAGEMENT_BONUS           = 0.03

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("penalty")


# ===========================================================================
# Helper: Extract title and location from clean_text
# ===========================================================================

def extract_title_from_text(text: str) -> str:
    """
    The clean_text starts with the headline (current_title).
    Extract everything before the first number or the word 'Professional/Software/Machine'.
    """
    # The pattern in our data: "Title [optional subtitle] X.X yrs..."
    # or "Title [subtitle] Professional with..."
    # or "Title [subtitle] Software engineer..."
    # or "Title [subtitle] Machine learning..."
    match = re.match(
        r'^(.*?)(?:\d+\s+\d+\s+yrs|Professional with|Software (?:data |engineer)|Machine learning|Project Manager with)',
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().lower()
    # Fallback: take first 5 words
    words = text.split()[:5]
    return " ".join(words).lower()


def extract_location_from_text(text: str) -> str:
    """
    Location appears in clean_text after the company name, e.g.:
    '... Swiggy Food Delivery Hyderabad Telangana ...'
    We search for known city names anywhere in the text.
    """
    return text.lower()


def count_roles_in_text(text: str) -> int:
    """
    Count distinct job roles by looking for company-name patterns.
    In our clean_text, each role block starts with a title + company combo.
    We use a heuristic: count occurrences of known company patterns or
    repeated title patterns.
    """
    # Common companies that appear in the data
    company_markers = [
        "wipro", "tcs", "infosys", "accenture", "cognizant", "capgemini",
        "hcl", "tech mahindra", "mindtree", "dunder mifflin", "acme corp",
        "globex inc", "initech", "pied piper", "hooli", "stark industries",
        "wayne enterprises", "swiggy", "zomato", "flipkart", "ola",
        "razorpay", "cred", "uber", "mad street den",
    ]
    text_lower = text.lower()
    count = 0
    for company in company_markers:
        # Each mention of a company typically = one role
        count += text_lower.count(company)

    # Each company appears twice in our data format (once in header, once in description)
    # so divide by 2, minimum 1
    return max(1, count // 2)


# ===========================================================================
# Penalty Engine
# ===========================================================================

def apply_penalties_and_bonuses(
    sem_df: pd.DataFrame,
    num_df: pd.DataFrame,
    text_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Merge all data sources, apply multiplicative penalties and additive bonuses,
    produce final_score in [0, 1].

    Returns a DataFrame with all original columns + penalty flags + final_score.
    """

    # --- 1. Merge all three data sources ---
    df = sem_df.merge(num_df, on="candidate_id", how="left")
    df = df.merge(text_df[["candidate_id", "clean_text"]], on="candidate_id", how="left")

    n = len(df)
    log.info("Merged %d candidates across semantic scores, numeric signals, and text corpus", n)

    # --- 2. Derive missing columns ---

    # avg_months_per_role: years_of_experience * 12 / num_roles
    df["num_roles"] = df["clean_text"].fillna("").apply(count_roles_in_text)
    df["avg_months_per_role"] = np.where(
        df["num_roles"] > 0,
        (df["years_of_experience"] * 12) / df["num_roles"],
        df["years_of_experience"] * 12,  # single role = all experience
    )

    # Activity proxy (no last_active_date in dataset, so use 30d engagement signals)
    # Higher = more recently active on the platform
    df["activity_proxy"] = (
        df["profile_views_received_30d"].fillna(0)
        + df["applications_submitted_30d"].fillna(0)
        + df["search_appearance_30d"].fillna(0) * 0.1  # downweight passive appearances
        + df["saved_by_recruiters_30d"].fillna(0)
    )

    # Extract title for mismatch check
    df["extracted_title"] = df["clean_text"].fillna("").apply(extract_title_from_text)

    # --- 3. Initialize penalty/bonus tracking ---
    df["penalty_multiplier"] = 1.0
    df["bonus_total"]        = 0.0
    df["penalties_applied"]  = ""
    df["bonuses_applied"]    = ""

    # =====================================================================
    # PENALTIES (multiplicative — stack)
    # =====================================================================

    # P1: Ghost Candidate — low response rate OR near-zero platform activity
    ghost_mask = (
        (df["recruiter_response_rate"].fillna(0) < GHOST_RESPONSE_THRESHOLD) |
        (df["activity_proxy"] < GHOST_ACTIVITY_THRESHOLD)
    )
    df.loc[ghost_mask, "penalty_multiplier"] *= GHOST_PENALTY
    df.loc[ghost_mask, "penalties_applied"] += "GHOST(×0.20) "
    log.info("P1 Ghost Candidate:    %d / %d flagged", ghost_mask.sum(), n)

    # P2: Role Mismatch — non-tech title with suspiciously high semantic score
    mismatch_mask = pd.Series(False, index=df.index)
    for idx, row in df.iterrows():
        title = row["extracted_title"]
        if any(mt in title for mt in MISMATCH_TITLES):
            if row["semantic_score"] > MISMATCH_SEMANTIC_THRESHOLD:
                mismatch_mask.at[idx] = True

    df.loc[mismatch_mask, "penalty_multiplier"] *= MISMATCH_PENALTY
    df.loc[mismatch_mask, "penalties_applied"] += "MISMATCH(×0.50) "
    log.info("P2 Role Mismatch:      %d / %d flagged", mismatch_mask.sum(), n)

    # P3: Job Hopper — experienced but switching too frequently
    hopper_mask = (
        (df["years_of_experience"] > HOPPER_EXP_THRESHOLD) &
        (df["avg_months_per_role"] < HOPPER_MONTHS_THRESHOLD)
    )
    df.loc[hopper_mask, "penalty_multiplier"] *= HOPPER_PENALTY
    df.loc[hopper_mask, "penalties_applied"] += "HOPPER(×0.60) "
    log.info("P3 Job Hopper:         %d / %d flagged", hopper_mask.sum(), n)

    # P4: Coding Recency — senior with no proof of hands-on work
    coding_fail_mask = (
        (df["years_of_experience"] > CODING_EXP_THRESHOLD) &
        (df["github_activity_score"].fillna(0) == 0) &
        (df["num_assessments_taken"].fillna(0) == 0)
    )
    df.loc[coding_fail_mask, "penalty_multiplier"] *= CODING_PENALTY
    df.loc[coding_fail_mask, "penalties_applied"] += "NO_CODE(×0.70) "
    log.info("P4 Coding Recency:     %d / %d flagged", coding_fail_mask.sum(), n)

    # P5: Consulting Lifer — every company is a services firm
    consulting_mask = pd.Series(False, index=df.index)
    for idx, row in df.iterrows():
        text_lower = row["clean_text"].lower() if pd.notna(row["clean_text"]) else ""
        # Find all companies mentioned
        companies_found = []
        for company in CONSULTING_FIRMS:
            if company in text_lower:
                companies_found.append(company)

        # If they have companies AND all of them are consulting firms
        all_companies_in_text = []
        for company in [
            "wipro", "tcs", "infosys", "accenture", "cognizant", "capgemini",
            "hcl", "tech mahindra", "mindtree", "dunder mifflin", "acme corp",
            "globex inc", "initech", "pied piper", "hooli", "stark industries",
            "wayne enterprises", "swiggy", "zomato", "flipkart", "ola",
            "razorpay", "cred", "uber", "mad street den",
        ]:
            if company in text_lower:
                all_companies_in_text.append(company)

        if all_companies_in_text:
            # Check if ALL companies are consulting firms
            all_consulting = all(c in CONSULTING_FIRMS for c in all_companies_in_text)
            if all_consulting:
                consulting_mask.at[idx] = True

    df.loc[consulting_mask, "penalty_multiplier"] *= CONSULTING_PENALTY
    df.loc[consulting_mask, "penalties_applied"] += "CONSULT(×0.65) "
    log.info("P5 Consulting Lifer:   %d / %d flagged", consulting_mask.sum(), n)

    # P6: Low Profile Engagement — very incomplete profile
    low_profile_mask = df["profile_completeness_score"].fillna(0) < LOW_COMPLETENESS_THRESHOLD
    df.loc[low_profile_mask, "penalty_multiplier"] *= LOW_COMPLETENESS_PENALTY
    df.loc[low_profile_mask, "penalties_applied"] += "LOW_PROFILE(×0.80) "
    log.info("P6 Low Profile:        %d / %d flagged", low_profile_mask.sum(), n)

    # P7: CV/Speech/Robotics specialist without NLP/IR
    cv_speech_mask = pd.Series(False, index=df.index)
    for idx, row in df.iterrows():
        text_lower = row["clean_text"].lower() if pd.notna(row["clean_text"]) else ""
        has_cv_speech = any(kw in text_lower for kw in CV_SPEECH_KEYWORDS)
        has_nlp_ir    = any(kw in text_lower for kw in NLP_IR_KEYWORDS)
        if has_cv_speech and not has_nlp_ir:
            cv_speech_mask.at[idx] = True

    df.loc[cv_speech_mask, "penalty_multiplier"] *= CV_SPEECH_PENALTY
    df.loc[cv_speech_mask, "penalties_applied"] += "CV_ONLY(×0.55) "
    log.info("P7 CV/Speech Only:     %d / %d flagged", cv_speech_mask.sum(), n)

    # P8: Pure research — research keywords without production keywords
    research_mask = pd.Series(False, index=df.index)
    for idx, row in df.iterrows():
        text_lower = row["clean_text"].lower() if pd.notna(row["clean_text"]) else ""
        has_research   = any(kw in text_lower for kw in RESEARCH_KEYWORDS)
        has_production = any(kw in text_lower for kw in PRODUCTION_KEYWORDS)
        if has_research and not has_production:
            research_mask.at[idx] = True

    df.loc[research_mask, "penalty_multiplier"] *= RESEARCH_ONLY_PENALTY
    df.loc[research_mask, "penalties_applied"] += "RESEARCH(×0.40) "
    log.info("P8 Pure Research:      %d / %d flagged", research_mask.sum(), n)

    # P9: LangChain-only — has LLM wrapper keywords but no pre-LLM ML
    langchain_mask = pd.Series(False, index=df.index)
    for idx, row in df.iterrows():
        text_lower = row["clean_text"].lower() if pd.notna(row["clean_text"]) else ""
        has_langchain = any(kw in text_lower for kw in LANGCHAIN_KEYWORDS)
        has_pre_llm   = any(kw in text_lower for kw in PRE_LLM_KEYWORDS)
        if has_langchain and not has_pre_llm:
            langchain_mask.at[idx] = True

    df.loc[langchain_mask, "penalty_multiplier"] *= LANGCHAIN_ONLY_PENALTY
    df.loc[langchain_mask, "penalties_applied"] += "LANGCHAIN(×0.45) "
    log.info("P9 LangChain Only:     %d / %d flagged", langchain_mask.sum(), n)

    # =====================================================================
    # BONUSES (additive — applied after penalties)
    # =====================================================================

    # B1: Short Notice Period
    short_notice_mask = df["notice_period_days"].fillna(999) <= SHORT_NOTICE_DAYS
    df.loc[short_notice_mask, "bonus_total"] += SHORT_NOTICE_BONUS
    df.loc[short_notice_mask, "bonuses_applied"] += "SHORT_NOTICE(+0.05) "
    log.info("B1 Short Notice:       %d / %d qualify", short_notice_mask.sum(), n)

    # B2: Tier-1 City
    city_mask = pd.Series(False, index=df.index)
    for idx, row in df.iterrows():
        text_lower = row["clean_text"].lower() if pd.notna(row["clean_text"]) else ""
        if any(city in text_lower for city in TIER1_CITIES):
            city_mask.at[idx] = True

    df.loc[city_mask, "bonus_total"] += CITY_BONUS
    df.loc[city_mask, "bonuses_applied"] += "TIER1_CITY(+0.05) "
    log.info("B2 Tier-1 City:        %d / %d qualify", city_mask.sum(), n)

    # B3: GitHub Active
    github_mask = df["github_activity_score"].fillna(0) >= GITHUB_ACTIVE_THRESHOLD
    df.loc[github_mask, "bonus_total"] += GITHUB_BONUS
    df.loc[github_mask, "bonuses_applied"] += "GITHUB(+0.03) "
    log.info("B3 GitHub Active:      %d / %d qualify", github_mask.sum(), n)

    # B4: Assessment Proven
    assessment_mask = (
        (df["skill_assessment_mean"].fillna(0) >= ASSESSMENT_SCORE_THRESHOLD) &
        (df["num_assessments_taken"].fillna(0) >= ASSESSMENT_COUNT_THRESHOLD)
    )
    df.loc[assessment_mask, "bonus_total"] += ASSESSMENT_BONUS
    df.loc[assessment_mask, "bonuses_applied"] += "ASSESSED(+0.02) "
    log.info("B4 Assessment Proven:  %d / %d qualify", assessment_mask.sum(), n)

    # B5: High Engagement (responsive + shows up to interviews)
    engagement_mask = (
        (df["recruiter_response_rate"].fillna(0) >= HIGH_RESPONSE_THRESHOLD) &
        (df["interview_completion_rate"].fillna(0) >= HIGH_INTERVIEW_THRESHOLD)
    )
    df.loc[engagement_mask, "bonus_total"] += ENGAGEMENT_BONUS
    df.loc[engagement_mask, "bonuses_applied"] += "ENGAGED(+0.03) "
    log.info("B5 High Engagement:    %d / %d qualify", engagement_mask.sum(), n)

    # =====================================================================
    # COMPUTE FINAL SCORE
    # =====================================================================

    # base_score = semantic_score (already [0, 1])
    df["base_score"] = df["semantic_score"]

    # Apply multiplicative penalties
    df["penalized_score"] = df["base_score"] * df["penalty_multiplier"]

    # Apply bonuses with exponential diminishing returns, capped at 0.99
    # ------------------------------------------------------------------
    # Problem: flat additive bonuses push many top candidates to exactly
    # 0.99, destroying differentiation where it matters most.
    #
    # Solution: exponential diminishing returns. Each bonus point fills
    # a decreasing fraction of the remaining room to the ceiling.
    # Formula: effective_bonus = room * (1 - e^(-bonus_total / scale))
    #   - First 0.05 bonus fills ~47% of room (significant)
    #   - Stacking 0.10 fills ~71% of room (less per additional bonus)
    #   - Stacking 0.18 fills ~89% of room (heavily diminished)
    # This preserves semantic ordering among top candidates while still
    # rewarding bonuses meaningfully.
    SCORE_CEILING = 0.99
    BONUS_SCALE = 0.08  # controls how fast diminishing returns kick in
    room = np.clip(SCORE_CEILING - df["penalized_score"], 0.0, None)
    # Exponential saturation: approaches room asymptotically
    effective_bonus = room * (1.0 - np.exp(-df["bonus_total"] / BONUS_SCALE))
    df["final_score"] = np.clip(df["penalized_score"] + effective_bonus, 0.0, SCORE_CEILING)

    # Sort by final_score descending
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # --- Stats ---
    log.info("\n--- Final Score Distribution ---")
    log.info("  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
             df["final_score"].mean(), df["final_score"].std(),
             df["final_score"].min(), df["final_score"].max())

    penalized_count = (df["penalties_applied"].str.strip() != "").sum()
    bonused_count   = (df["bonuses_applied"].str.strip() != "").sum()
    log.info("  %d candidates received at least one penalty", penalized_count)
    log.info("  %d candidates received at least one bonus", bonused_count)

    if verbose:
        log.info("\n--- Per-Candidate Breakdown ---")
        for _, row in df.iterrows():
            penalties = row["penalties_applied"].strip() or "none"
            bonuses   = row["bonuses_applied"].strip() or "none"
            log.info(
                "  #%02d  %s  base=%.4f  ×%.2f  +%.2f  → final=%.4f  | penalties: %s | bonuses: %s",
                row["rank"], row["candidate_id"],
                row["base_score"], row["penalty_multiplier"], row["bonus_total"],
                row["final_score"],
                penalties, bonuses,
            )

    return df


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Penalty & Bonus Engine — applies business rules to semantic scores"
    )
    parser.add_argument(
        "--semantic-scores", type=str, default=None,
        help=f"Path to semantic scores PKL (default: {SEMANTIC_SCORES_PATH})"
    )
    parser.add_argument(
        "--numeric-signals", type=str, default=None,
        help=f"Path to numeric signals PKL (default: {NUMERIC_SIGNALS_PATH})"
    )
    parser.add_argument(
        "--text-corpus", type=str, default=None,
        help=f"Path to text corpus PKL (default: {TEXT_CORPUS_PATH})"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"Output CSV path (default: {OUTPUT_CSV_PATH})"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-candidate penalty/bonus breakdown"
    )
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # --- Load data ---
    sem_path  = Path(args.semantic_scores) if args.semantic_scores else SEMANTIC_SCORES_PATH
    num_path  = Path(args.numeric_signals) if args.numeric_signals else NUMERIC_SIGNALS_PATH
    text_path = Path(args.text_corpus) if args.text_corpus else TEXT_CORPUS_PATH

    log.info("Loading semantic scores from %s ...", sem_path)
    if sem_path.exists():
        sem_df = pd.read_pickle(sem_path)
    else:
        # Fallback to CSV if PKL doesn't exist yet
        csv_fallback = sem_path.with_suffix(".csv")
        log.info("  PKL not found, falling back to %s", csv_fallback)
        sem_df = pd.read_csv(csv_fallback)

    log.info("Loading numeric signals from %s ...", num_path)
    num_df = pd.read_pickle(num_path)

    log.info("Loading text corpus from %s ...", text_path)
    text_df = pd.read_pickle(text_path)

    # --- Apply penalties & bonuses ---
    results = apply_penalties_and_bonuses(sem_df, num_df, text_df, verbose=args.verbose)

    # --- Save output ---
    output_cols = [
        "rank", "candidate_id", "final_score",
        "semantic_score", "rrf_score",
        "penalty_multiplier", "bonus_total",
        "penalties_applied", "bonuses_applied",
    ]

    if args.output:
        csv_path = Path(args.output)
        pkl_path = csv_path.with_suffix(".pkl")
    else:
        csv_path = OUTPUT_CSV_PATH
        pkl_path = OUTPUT_PKL_PATH

    results[output_cols].to_csv(csv_path, index=False)
    results[output_cols].to_pickle(pkl_path)
    log.info("Final ranking saved to %s  and  %s", csv_path, pkl_path)

    # --- Print top results ---
    print("\n" + "=" * 80)
    print(f"  FINAL RANKING — TOP {min(20, len(results))} CANDIDATES")
    print("=" * 80)
    print(f"  {'#':>3}  {'Candidate':<14}  {'Final':>6}  {'Base':>6}  {'Pen':>5}  {'Bon':>5}  {'Flags'}")
    print("-" * 80)
    for _, row in results.head(20).iterrows():
        penalties = row["penalties_applied"].strip() or "—"
        bonuses   = row["bonuses_applied"].strip() or "—"
        flags = f"{penalties} | {bonuses}"
        print(
            f"  {row['rank']:>3}  {row['candidate_id']:<14}  "
            f"{row['final_score']:.4f}  {row['base_score']:.4f}  "
            f"×{row['penalty_multiplier']:.2f}  +{row['bonus_total']:.2f}  "
            f"{flags}"
        )
    print()


if __name__ == "__main__":
    main()
