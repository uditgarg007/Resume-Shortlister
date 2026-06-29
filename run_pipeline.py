"""
run_pipeline.py — Full End-to-End Ranking Pipeline

Orchestrates all three subproblems in sequence:
  1. hybrid_search.py  → semantic_scores (BM25 + HNSW + Cross-Encoder)
  2. penalty.py        → final_ranking   (business-rule penalties & bonuses)

Produces a single final output: processed/final_ranking.csv + .pkl

Usage:
  python run_pipeline.py                        # default JD, default settings
  python run_pipeline.py --jd job_description.docx --verbose
  python run_pipeline.py --top-k 50 --rebuild   # force index rebuild
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import our pipeline modules
from hybrid_search import (
    HybridSearchEngine,
    compute_tiered_score,
    parse_jd_from_docx,
    parse_jd_from_json,
    get_default_jd,
    PROCESSED_DIR,
    OUTPUT_CSV_PATH as SEMANTIC_CSV_PATH,
    OUTPUT_PKL_PATH as SEMANTIC_PKL_PATH,
)
from penalty import (
    apply_penalties_and_bonuses,
    TEXT_CORPUS_PATH,
    NUMERIC_SIGNALS_PATH,
    OUTPUT_CSV_PATH as FINAL_CSV_PATH,
    OUTPUT_PKL_PATH as FINAL_PKL_PATH,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline — Hybrid Search → Penalty Engine → Final Ranking"
    )
    # JD options
    parser.add_argument(
        "--jd", type=str, default=None,
        help="Path to a .docx JD file (auto-parsed into tiers)"
    )
    parser.add_argument(
        "--jd-json", type=str, default=None,
        help="Path to a pre-structured .json JD file"
    )
    # Search options
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="Number of candidates to retrieve and score (default: 100)"
    )
    parser.add_argument(
        "--retrieval-depth", type=int, default=200,
        help="Candidates fetched per retriever before RRF fusion (default: 200)"
    )
    parser.add_argument(
        "--bm25-weight", type=float, default=0.20,
        help="BM25 weight in RRF fusion (default: 0.20)"
    )
    parser.add_argument(
        "--vector-weight", type=float, default=0.80,
        help="Vector search weight in RRF fusion (default: 0.80)"
    )
    # Output options
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"Output CSV path (default: {FINAL_CSV_PATH})"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild of BM25 and FAISS indexes"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed statistics and per-candidate breakdown"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ===================================================================
    # STAGE 1: Parse JD
    # ===================================================================
    print("\n" + "=" * 70)
    print("  STAGE 1 / 3 — JOB DESCRIPTION PARSING")
    print("=" * 70)

    if args.jd:
        log.info("Parsing JD from .docx: %s", args.jd)
        jd = parse_jd_from_docx(args.jd)
    elif args.jd_json:
        log.info("Loading JD from .json: %s", args.jd_json)
        jd = parse_jd_from_json(args.jd_json)
    else:
        log.info("No JD file provided — using hardcoded Redrob AI Engineer JD")
        jd = get_default_jd()

    # ===================================================================
    # STAGE 2: Hybrid Search — Semantic Scoring
    # ===================================================================
    print("\n" + "=" * 70)
    print("  STAGE 2 / 3 — HYBRID SEMANTIC SCORING")
    print("=" * 70)

    engine = HybridSearchEngine()
    engine.load_or_build_indices(force_rebuild=args.rebuild)

    # Build broad retrieval query from all tier sub-queries
    # Flatten lists into a single string for BM25/vector retrieval
    broad_parts = []
    for tier_key in ["must_have", "good_to_have", "bonus"]:
        tier_val = jd.get(tier_key, "")
        if isinstance(tier_val, list):
            broad_parts.extend(tier_val)
        elif isinstance(tier_val, str) and tier_val.strip():
            broad_parts.append(tier_val)
    broad_query = " ".join(broad_parts).strip()
    if not broad_query:
        broad_query = jd.get("full_text", "AI ML engineer")

    # Log tier sizes
    for tier_key in ["must_have", "good_to_have", "bonus", "disqualifiers"]:
        tier_val = jd.get(tier_key, "")
        if isinstance(tier_val, list):
            log.info("  %s: %d sub-queries", tier_key, len(tier_val))
        else:
            log.info("  %s: %d chars", tier_key, len(tier_val))

    log.info("[Stage 2a] Hybrid RRF retrieval (BM25=%.0f%%, Vector=%.0f%%) ...",
             args.bm25_weight * 100, args.vector_weight * 100)

    candidates = engine.hybrid_rrf(
        query=broad_query,
        bm25_weight=args.bm25_weight,
        vector_weight=args.vector_weight,
        retrieval_depth=args.retrieval_depth,
        top_k=args.top_k,
    )
    log.info("Retrieved %d candidates from hybrid search", len(candidates))

    # Tiered reranking
    log.info("[Stage 2b] Cross-Encoder reranking with JD tier weights ...")
    sem_results = compute_tiered_score(engine, candidates, jd,
                                       top_k=args.top_k, verbose=args.verbose)

    # Normalize rrf_score to [0, 1]
    rrf_min = sem_results["rrf_score"].min()
    rrf_max = sem_results["rrf_score"].max()
    if rrf_max > rrf_min:
        sem_results["rrf_score"] = (sem_results["rrf_score"] - rrf_min) / (rrf_max - rrf_min)
    else:
        sem_results["rrf_score"] = 1.0

    # Save semantic scores (intermediate output)
    sem_output = sem_results[["candidate_id", "semantic_score", "rrf_score"]].copy()
    sem_output.to_csv(SEMANTIC_CSV_PATH, index=False)
    sem_output.to_pickle(SEMANTIC_PKL_PATH)
    log.info("Semantic scores saved → %s", SEMANTIC_CSV_PATH)

    # ===================================================================
    # STAGE 3: Penalty & Bonus Engine
    # ===================================================================
    print("\n" + "=" * 70)
    print("  STAGE 3 / 3 — PENALTY & BONUS ENGINE")
    print("=" * 70)

    log.info("Loading numeric signals and text corpus ...")
    num_df  = pd.read_pickle(NUMERIC_SIGNALS_PATH)
    text_df = pd.read_pickle(TEXT_CORPUS_PATH)

    final_results = apply_penalties_and_bonuses(
        sem_df=sem_output,
        num_df=num_df,
        text_df=text_df,
        verbose=args.verbose,
    )

    # --- Save final output ---
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
        csv_path = FINAL_CSV_PATH
        pkl_path = FINAL_PKL_PATH

    final_results[output_cols].to_csv(csv_path, index=False)
    final_results[output_cols].to_pickle(pkl_path)
    log.info("Final ranking saved → %s", csv_path)

    # ===================================================================
    # FINAL RESULTS
    # ===================================================================
    n_show = min(20, len(final_results))
    print("\n" + "=" * 90)
    print(f"  FINAL RANKING — TOP {n_show} CANDIDATES")
    print("=" * 90)
    print(f"  {'#':>3}  {'Candidate':<14}  {'Final':>6}  {'Semantic':>8}  "
          f"{'Penalty':>7}  {'Bonus':>5}  {'Flags'}")
    print("-" * 90)

    for _, row in final_results.head(n_show).iterrows():
        penalties = row["penalties_applied"].strip() or "—"
        bonuses   = row["bonuses_applied"].strip() or "—"
        flags = f"{penalties} | {bonuses}"
        print(
            f"  {row['rank']:>3}  {row['candidate_id']:<14}  "
            f"{row['final_score']:.4f}  {row['semantic_score']:>8.4f}  "
            f"  ×{row['penalty_multiplier']:.2f}  +{row['bonus_total']:.2f}  "
            f"{flags}"
        )

    print()
    print(f"  Output: {csv_path}")
    print(f"  Total candidates scored: {len(final_results)}")
    print()


if __name__ == "__main__":
    main()
