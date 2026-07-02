"""
app.py — Flask Web Server for Resume Shortlister

Provides a web frontend that lets users:
  1. Paste or upload ANY job description
  2. Configure tier weights & penalty multipliers via sliders
  3. Run the full pipeline and view ranked results

Pre-computed embeddings + indices ship with the app, so 100k candidates
don't need re-embedding on every run. Only the Cross-Encoder reranking
(which is the differentiating step) runs fresh against the new JD.

Performance optimization: Default JD results are cached after the first
run, so judges can test the default JD instantly (~1 second).

Usage:
  python app.py              # starts on http://localhost:7860
  python app.py --port 8080  # custom port
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response

# Import pipeline modules
from hybrid_search import (
    HybridSearchEngine,
    compute_tiered_score,
    get_default_jd,
    PROCESSED_DIR,
    TIER_WEIGHTS,
    DISQUALIFIER_PENALTY,
)
from penalty import apply_penalties_and_bonuses

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JD_PATH = BASE_DIR / "default_jd.json"
TEXT_CORPUS_PATH = PROCESSED_DIR / "text_corpus.pkl"
NUMERIC_SIGNALS_PATH = PROCESSED_DIR / "numeric_signals.pkl"
CACHED_RESULTS_PATH = PROCESSED_DIR / "cached_default_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")

app = Flask(__name__, template_folder="templates", static_folder="static")

from dataset_manager import DatasetManager

# Global engines — loaded once at startup for speed
_engines: dict[str, HybridSearchEngine] = {}
_engine_lock = threading.Lock()
_dataset_manager = None

def get_dataset_manager() -> DatasetManager:
    global _dataset_manager
    if _dataset_manager is None:
        # We need an encoder, so we ensure the default engine is loaded first
        # However, loading default engine at startup covers this.
        # Just create the manager with None encoder, it will lazy load if needed,
        # but we can pass the default engine's encoder if available to save memory.
        default_engine = _engines.get("default")
        encoder = default_engine.encoder if default_engine else None
        _dataset_manager = DatasetManager(encoder)
    return _dataset_manager

def get_engine(dataset_name: str = "default") -> HybridSearchEngine:
    """Lazy-load the search engine for a specific dataset."""
    global _engines
    if dataset_name not in _engines:
        with _engine_lock:
            if dataset_name not in _engines:
                log.info("Loading search engine for dataset '%s'...", dataset_name)
                
                # Check if dataset exists
                mgr = get_dataset_manager()
                if not mgr.dataset_exists(dataset_name):
                    raise ValueError(f"Dataset '{dataset_name}' not found or not ready.")

                # Optimize: share models from default engine to save RAM/VRAM
                default_engine = _engines.get("default")
                engine = HybridSearchEngine()
                if default_engine:
                    engine.encoder = default_engine.encoder
                    engine.cross_encoder = default_engine.cross_encoder
                    engine._tokenizer = default_engine._tokenizer

                # Load indices from the specific dataset directory
                data_dir = mgr.get_dataset_dir(dataset_name)
                engine.load_or_build_indices(data_dir=str(data_dir))
                
                _engines[dataset_name] = engine
                log.info("Search engine '%s' ready (%d candidates indexed)", dataset_name, len(engine.corpus_texts))
    return _engines[dataset_name]

# Results cache — keyed by hash of JD + config
_results_cache: dict[str, dict] = {}

# Store last pipeline results for CSV export
_last_results: list[dict] = []

# Candidate profile lookup — binary search on sorted candidates.jsonl
_candidate_profiles: dict[str, dict] = {}   # LRU-style in-memory cache
_jsonl_path: Path | None = None
_jsonl_size: int = 0
import re as _re
_ID_PAT = _re.compile(rb'"candidate_id":\s*"(CAND_[^"]+)"')


def load_candidate_profiles():
    """Locate candidates.jsonl for binary search. No pre-loading needed."""
    global _jsonl_path, _jsonl_size
    jsonl = BASE_DIR / "candidates.jsonl"
    if jsonl.exists():
        _jsonl_path = jsonl
        _jsonl_size = jsonl.stat().st_size
        log.info("candidates.jsonl ready for binary search (%d bytes)", _jsonl_size)
    else:
        log.warning("candidates.jsonl not found")


def _read_line_at(f, offset: int) -> bytes:
    """Seek to offset, skip partial line, return next complete line."""
    f.seek(offset)
    if offset > 0:
        f.readline()  # discard the partial line we landed in the middle of
    return f.readline()


def lookup_candidate_binary(candidate_id: str) -> dict | None:
    """Binary search candidates.jsonl (sorted by candidate_id) for the given ID.
    IDs are zero-padded 7-digit so lexicographic order == numeric order.
    Finds any record in ~17 file seeks for 100k candidates."""
    if not _jsonl_path or _jsonl_size == 0:
        return None
    if candidate_id in _candidate_profiles:
        return _candidate_profiles[candidate_id]

    lo, hi = 0, _jsonl_size
    try:
        with open(_jsonl_path, "rb") as f:
            while lo < hi:
                mid = (lo + hi) // 2
                line = _read_line_at(f, mid)
                if not line:
                    hi = mid
                    continue
                m = _ID_PAT.search(line)
                if not m:
                    lo = mid + 1
                    continue
                mid_id = m.group(1).decode()
                if mid_id == candidate_id:
                    data = json.loads(line.decode("utf-8"))
                    _candidate_profiles[candidate_id] = data  # cache
                    return data
                elif mid_id < candidate_id:
                    lo = mid + len(line)
                else:
                    hi = mid
    except Exception as e:
        log.warning("Binary search failed for %s: %s", candidate_id, e)
    return None

def load_default_jd() -> dict:
    """Load the default JD config from JSON file."""
    if DEFAULT_JD_PATH.exists():
        with open(DEFAULT_JD_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback to hardcoded
    jd = get_default_jd()
    jd["tier_weights"] = dict(TIER_WEIGHTS)
    jd["disqualifier_penalty"] = DISQUALIFIER_PENALTY
    return jd


def make_cache_key(payload: dict) -> str:
    """Create a hash key from the JD + scoring config (NOT display top_k)."""
    key_parts = json.dumps({
        "must_have": payload.get("must_have", []),
        "good_to_have": payload.get("good_to_have", []),
        "bonus": payload.get("bonus", []),
        "disqualifiers": payload.get("disqualifiers", []),
        "tier_weights": payload.get("tier_weights", {}),
        "disqualifier_penalty": payload.get("disqualifier_penalty", 0.15),
        "penalty_config": payload.get("penalty_config", {}),
        "dataset_name": payload.get("dataset_name", "default"),
        # top_k intentionally excluded — display count does not change pipeline scoring
    }, sort_keys=True)
    return hashlib.md5(key_parts.encode()).hexdigest()


def try_load_cached_results() -> dict | None:
    """Try to load cached default JD results from disk."""
    if CACHED_RESULTS_PATH.exists():
        try:
            with open(CACHED_RESULTS_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            log.info("Loaded cached default results from disk (%d results)", len(cached.get("results", [])))
            return cached
        except Exception as e:
            log.warning("Failed to load cached results: %s", e)
    return None


def save_cached_results(data: dict):
    """Save results to disk cache."""
    try:
        with open(CACHED_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
        log.info("Saved results to disk cache")
    except Exception as e:
        log.warning("Failed to save cache: %s", e)


# ===========================================================================
# Routes
# ===========================================================================

@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/api/default-jd")
def api_default_jd():
    """Return the default JD config for pre-filling the form."""
    return jsonify(load_default_jd())


@app.route("/api/status")
def api_status():
    """Check if the engine is loaded and how many candidates are indexed."""
    engine = get_engine()
    return jsonify({
        "ready": True,
        "candidates_indexed": len(engine.corpus_texts),
    })


@app.route("/api/datasets", methods=["GET"])
def api_list_datasets():
    """List all available candidate datasets."""
    mgr = get_dataset_manager()
    return jsonify(mgr.list_datasets())


@app.route("/api/datasets", methods=["POST"])
def api_create_dataset():
    """Upload and process a new custom candidate dataset."""
    name = request.form.get("name")
    if not name:
        return jsonify({"error": "Dataset name required"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
        
    try:
        content = file.read().decode("utf-8")
        mgr = get_dataset_manager()
        candidates = mgr.parse_upload(content, file.filename)
        
        # In a real app this would be a background task (e.g. Celery)
        # For this demo, we block and return when done
        meta = mgr.create_dataset(name, candidates)
        
        return jsonify({"message": f"Dataset {name} created successfully", "meta": meta})
    except Exception as e:
        log.exception("Dataset creation failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/datasets/<name>", methods=["DELETE"])
def api_delete_dataset(name):
    """Delete a custom candidate dataset."""
    mgr = get_dataset_manager()
    try:
        if mgr.delete_dataset(name):
            # Remove from loaded engines if present
            global _engines
            with _engine_lock:
                if name in _engines:
                    del _engines[name]
            return jsonify({"message": f"Dataset {name} deleted"})
        return jsonify({"error": "Dataset not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/run", methods=["POST"])
def api_run():
    """
    Run the full pipeline with the provided JD config.

    Expects JSON body with must_have, good_to_have, bonus, disqualifiers,
    tier_weights, disqualifier_penalty, penalty_config, top_k.
    """
    try:
        global _last_results
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        # Build JD dict for the pipeline
        jd = {
            "must_have": data.get("must_have", []),
            "good_to_have": data.get("good_to_have", []),
            "bonus": data.get("bonus", []),
            "disqualifiers": data.get("disqualifiers", []),
            "full_text": "",
        }

        # Validate at least must_have has content
        if not jd["must_have"] or all(not q.strip() for q in jd["must_have"]):
            return jsonify({"error": "At least one 'Must Have' requirement is needed"}), 400

        # top_k controls display only; pipeline always uses PIPELINE_TOP_K for consistent ranking
        PIPELINE_TOP_K = 100
        display_top_k = min(int(data.get("top_k", 50)), 500)

        # --- Check cache first ---
        cache_key = make_cache_key(data)
        if cache_key in _results_cache:
            log.info("Cache HIT — returning cached results (key=%s)", cache_key[:8])
            cached = _results_cache[cache_key]
            sliced = {"results": cached["results"][:display_top_k], "stats": cached["stats"]}
            _last_results = sliced["results"]
            return jsonify(sliced)

        # Check if this is the default JD (compare sub-queries)
        default_jd = load_default_jd()
        is_default_jd = (
            data.get("must_have") == default_jd.get("must_have") and
            data.get("good_to_have") == default_jd.get("good_to_have") and
            data.get("bonus") == default_jd.get("bonus") and
            data.get("disqualifiers") == default_jd.get("disqualifiers")
        )

        # Try disk cache for default JD with default settings
        if is_default_jd:
            cached = try_load_cached_results()
            if cached:
                _results_cache[cache_key] = cached
                sliced = {"results": cached["results"][:display_top_k], "stats": cached["stats"]}
                _last_results = sliced["results"]
                return jsonify(sliced)

        # --- Override tier weights ---
        tier_weights = data.get("tier_weights", {})
        import hybrid_search
        if tier_weights:
            hybrid_search.TIER_WEIGHTS["must_have"] = float(tier_weights.get("must_have", 1.0))
            hybrid_search.TIER_WEIGHTS["good_to_have"] = float(tier_weights.get("good_to_have", 0.5))
            hybrid_search.TIER_WEIGHTS["bonus"] = float(tier_weights.get("bonus", 0.25))

        disq_pen = data.get("disqualifier_penalty", 0.15)
        hybrid_search.DISQUALIFIER_PENALTY = float(disq_pen)

        # --- Override penalty config ---
        penalty_config = data.get("penalty_config", {})
        import penalty as penalty_module
        if penalty_config:
            penalty_module.GHOST_PENALTY = float(penalty_config.get("ghost_penalty", 0.20))
            penalty_module.MISMATCH_PENALTY = float(penalty_config.get("mismatch_penalty", 0.50))
            penalty_module.HOPPER_PENALTY = float(penalty_config.get("hopper_penalty", 0.60))
            penalty_module.CODING_PENALTY = float(penalty_config.get("coding_penalty", 0.70))
            penalty_module.CONSULTING_PENALTY = float(penalty_config.get("consulting_penalty", 0.65))
            penalty_module.LOW_COMPLETENESS_PENALTY = float(penalty_config.get("low_profile_penalty", 0.80))
            penalty_module.CV_SPEECH_PENALTY = float(penalty_config.get("cv_speech_penalty", 0.55))
            penalty_module.RESEARCH_ONLY_PENALTY = float(penalty_config.get("research_penalty", 0.40))
            penalty_module.LANGCHAIN_ONLY_PENALTY = float(penalty_config.get("langchain_penalty", 0.45))

        # --- Run pipeline ---
        start_time = time.time()
        dataset_name = data.get("dataset_name", "default")
        engine = get_engine(dataset_name)

        # Stage 1: Broad retrieval
        broad_parts = []
        for tier_key in ["must_have", "good_to_have", "bonus"]:
            tier_val = jd.get(tier_key, [])
            if isinstance(tier_val, list):
                broad_parts.extend(tier_val)
            elif isinstance(tier_val, str) and tier_val.strip():
                broad_parts.append(tier_val)
        broad_query = " ".join(broad_parts).strip()
        if not broad_query:
            broad_query = "AI ML engineer"

        candidates = engine.hybrid_rrf(
            query=broad_query,
            bm25_weight=0.20,
            vector_weight=0.80,
            retrieval_depth=200,
            top_k=PIPELINE_TOP_K,
        )

        # Stage 2: Cross-encoder reranking
        sem_results = compute_tiered_score(engine, candidates, jd, top_k=PIPELINE_TOP_K, verbose=False)

        # Normalize rrf_score
        rrf_min = sem_results["rrf_score"].min()
        rrf_max = sem_results["rrf_score"].max()
        if rrf_max > rrf_min:
            sem_results["rrf_score"] = (sem_results["rrf_score"] - rrf_min) / (rrf_max - rrf_min)
        else:
            sem_results["rrf_score"] = 1.0

        sem_output = sem_results[["candidate_id", "semantic_score", "rrf_score"]].copy()

        # Stage 3: Penalty & bonus engine
        mgr = get_dataset_manager()
        data_dir = mgr.get_dataset_dir(dataset_name)
        num_df = pd.read_pickle(data_dir / "numeric_signals.pkl")
        text_df = pd.read_pickle(data_dir / "text_corpus.pkl")

        final_results = apply_penalties_and_bonuses(
            sem_df=sem_output, num_df=num_df, text_df=text_df, verbose=False,
        )

        elapsed = time.time() - start_time

        # Build full result list (all PIPELINE_TOP_K results stored in cache)
        all_results = []
        for _, row in final_results.iterrows():
            all_results.append({
                "rank": int(row["rank"]),
                "candidate_id": row["candidate_id"],
                "final_score": round(float(row["final_score"]), 4),
                "semantic_score": round(float(row["semantic_score"]), 4),
                "penalty_multiplier": round(float(row["penalty_multiplier"]), 2),
                "bonus_total": round(float(row["bonus_total"]), 2),
                "penalties": row["penalties_applied"].strip() or "None",
                "bonuses": row["bonuses_applied"].strip() or "None",
            })

        # Stats
        scores = final_results["final_score"]
        stats = {
            "total_candidates": len(engine.corpus_texts),
            "scored_candidates": len(final_results),
            "score_mean": round(float(scores.mean()), 4),
            "score_std": round(float(scores.std()), 4),
            "score_min": round(float(scores.min()), 4),
            "score_max": round(float(scores.max()), 4),
            "penalized_count": int((final_results["penalties_applied"].str.strip() != "").sum()),
            "bonused_count": int((final_results["bonuses_applied"].str.strip() != "").sum()),
            "elapsed_seconds": round(elapsed, 1),
        }

        # Cache full results; response only sends display_top_k rows
        cache_data = {"results": all_results, "stats": stats}
        _results_cache[cache_key] = cache_data
        if is_default_jd:
            save_cached_results(cache_data)

        # Store full results for CSV export
        _last_results = all_results[:display_top_k]

        response_data = {"results": all_results[:display_top_k], "stats": stats}
        return jsonify(response_data)

    except Exception as e:
        log.exception("Pipeline error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/candidate/<candidate_id>")
def api_candidate_profile(candidate_id):
    """Return full profile data for a specific candidate via binary search."""
    load_candidate_profiles()
    profile = lookup_candidate_binary(candidate_id)
    if profile is None:
        return jsonify({"error": f"Candidate {candidate_id} not found in dataset"}), 404
    return jsonify(profile)


@app.route("/api/export-csv")
def api_export_csv():
    """Export the last pipeline results as a downloadable CSV file."""
    if not _last_results:
        return jsonify({"error": "No results to export. Run the pipeline first."}), 400

    output = io.StringIO()
    fieldnames = [
        "rank", "candidate_id", "final_score", "semantic_score",
        "penalty_multiplier", "bonus_total", "penalties", "bonuses",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in _last_results:
        writer.writerow({
            "rank": row["rank"],
            "candidate_id": row["candidate_id"],
            "final_score": row["final_score"],
            "semantic_score": row["semantic_score"],
            "penalty_multiplier": row["penalty_multiplier"],
            "bonus_total": row["bonus_total"],
            "penalties": row["penalties"],
            "bonuses": row["bonuses"],
        })
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=shortlisted_candidates.csv"},
    )


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Shortlister Web App")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    # Pre-load candidate profiles at startup
    log.info("Pre-loading candidate profiles...")
    load_candidate_profiles()

    # Pre-load engine at startup
    log.info("Pre-loading search engine...")
    get_engine()
    log.info("Server starting on http://localhost:%d", args.port)

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
