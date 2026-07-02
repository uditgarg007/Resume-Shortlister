"""
hybrid_search.py — Hybrid Semantic Scorer for Resume Shortlisting

Combines BM25 keyword search (20% weight) with HNSW vector search (80% weight)
using Reciprocal Rank Fusion (RRF), followed by Cross-Encoder reranking for
nuanced semantic understanding (e.g. "uses ChatGPT" vs "builds ML models").

JD requirements are split into 4 tiers with multiplicative weights:
  - must_have:    1.00x  (absolute requirements)
  - good_to_have: 0.50x  (preferred but not mandatory)
  - bonus:        0.25x  (appreciated extras)
  - disqualifiers: negative penalty applied to matching candidates

Final output: each candidate receives a semantic_score in [0.15, 0.95].

Usage:
  python hybrid_search.py                           # interactive / default JD
  python hybrid_search.py --jd path/to/jd.docx      # parse JD from .docx
  python hybrid_search.py --jd-json path/to/jd.json  # parse JD from pre-structured JSON
  python hybrid_search.py --top-k 100 --verbose      # control output size + logging
"""

import argparse
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# Deterministic Inference — ensures reproducible scores across machines
# Without this, Cross-Encoder produces slightly different floats on
# different hardware (Windows GPU vs Linux CPU on HF Spaces), causing
# score differences that cascade through calibration & ranking.
# ---------------------------------------------------------------------------
_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)
# Enable deterministic algorithms where possible (PyTorch ≥1.8)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
# Disable CuDNN benchmarking (non-deterministic kernel selection)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

TEXT_CORPUS_PATH  = PROCESSED_DIR / "text_corpus.pkl"
FAISS_INDEX_PATH  = PROCESSED_DIR / "faiss_hnsw.index"
BM25_INDEX_PATH   = PROCESSED_DIR / "bm25_index.pkl"
EMBEDDINGS_PATH   = PROCESSED_DIR / "embeddings.npy"
OUTPUT_CSV_PATH   = PROCESSED_DIR / "semantic_scores.csv"
OUTPUT_PKL_PATH   = PROCESSED_DIR / "semantic_scores.pkl"

# Model defaults — can be overridden via CLI
DEFAULT_BIENCODER    = "all-MiniLM-L6-v2"
DEFAULT_CROSSENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# HNSW parameters
HNSW_M              = 32     # neighbors per node (build-time)
HNSW_EF_CONSTRUCT   = 200    # accuracy during build (higher = slower but better index)
HNSW_EF_SEARCH      = 128    # accuracy during query (higher = slower but better recall)

# Chunking parameters — fixes silent truncation at model's 256-token limit
# Resumes avg ~400+ tokens; without chunking the model never sees skills/education/certs
CHUNK_TOKEN_LIMIT   = 200    # tokens per chunk (leave headroom below model's 256 max)
CHUNK_OVERLAP       = 50     # overlapping tokens to avoid splitting sentences

# RRF parameters
RRF_K = 60  # standard RRF constant

# JD tier weights
TIER_WEIGHTS = {
    "must_have":     1.00,
    "good_to_have":  0.50,
    "bonus":         0.25,
}
DISQUALIFIER_PENALTY = 0.15  # subtracted from final score per disqualifier match

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hybrid_search")


# ===========================================================================
# JD Parser
# ===========================================================================

def parse_jd_from_docx(docx_path: str) -> dict:
    """
    Reads a job description .docx and extracts the 4 tiers.

    The Redrob JD uses these section headers:
      - "Things you absolutely need"      → must_have
      - "Things we'd like you to have..." → good_to_have
      - "Things we explicitly do NOT want" → disqualifiers

    Everything else in the body is treated as general context (used in
    the broad retrieval query but not weighted separately).

    Returns:
        dict with keys: must_have, good_to_have, bonus, disqualifiers, full_text
    """
    from docx import Document

    doc = Document(docx_path)
    full_text = "\n".join(p.text for p in doc.paragraphs)

    # --- Section extraction via keyword markers ---
    sections = {
        "must_have": "",
        "good_to_have": "",
        "bonus": "",
        "disqualifiers": "",
        "full_text": full_text,
    }

    current_section = None
    buffer = []

    section_markers = {
        "things you absolutely need":                   "must_have",
        "things we'd like you to have":                 "good_to_have",
        "things we explicitly do not want":             "disqualifiers",
    }

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        lower = text.lower()

        # Check if this paragraph is a section header
        matched_section = None
        for marker, section_key in section_markers.items():
            if marker in lower:
                matched_section = section_key
                break

        if matched_section:
            # Flush previous buffer
            if current_section and buffer:
                sections[current_section] = " ".join(buffer)
            current_section = matched_section
            buffer = []
        elif current_section:
            # Check if we've hit a new top-level section (exits the current tier)
            if _is_new_section_header(text, para):
                if buffer:
                    sections[current_section] = " ".join(buffer)
                current_section = None
                buffer = []
            else:
                buffer.append(text)

    # Flush final buffer
    if current_section and buffer:
        sections[current_section] = " ".join(buffer)

    # If no explicit bonus section, leave it empty (will be skipped in scoring)
    log.info("JD parsed — must_have: %d chars, good_to_have: %d chars, disqualifiers: %d chars",
             len(sections["must_have"]), len(sections["good_to_have"]), len(sections["disqualifiers"]))

    return sections


def _is_new_section_header(text: str, para) -> bool:
    """Heuristic: a paragraph is a section header if it's short and bold/styled."""
    if len(text) > 120:
        return False
    lower = text.lower()
    header_keywords = [
        "on location", "the vibe check", "how to read between the lines",
        "final note", "what you'd actually be doing", "what we mean by",
        "let's be honest", "the skills inventory",
    ]
    return any(kw in lower for kw in header_keywords)


def parse_jd_from_json(json_path: str) -> dict:
    """
    Reads a pre-structured JD JSON with keys:
      must_have, good_to_have, bonus, disqualifiers
    Each value is a string of concatenated requirements.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {"must_have", "good_to_have", "bonus", "disqualifiers"}
    for key in required_keys:
        if key not in data:
            data[key] = ""

    if "full_text" not in data:
        data["full_text"] = " ".join(v for v in data.values() if isinstance(v, str))

    return data


def get_default_jd() -> dict:
    """
    Hardcoded fallback JD for the Redrob Senior AI Engineer posting.
    Used when no --jd or --jd-json flag is provided.

    DESIGN DECISION: Each tier uses a list of SHORT, focused sub-queries
    instead of one long paragraph. Cross-encoders (ms-marco family) were
    trained on (short query, passage) pairs and produce much higher-quality
    scores when the query is 10-30 words, not 100+.

    Each sub-query targets ONE specific signal the JD cares about,
    phrased with action verbs for Cross-Encoder discrimination.
    """
    return {
        # --- SUB-QUERY FORMAT ---
        # Each tier is a list of focused queries. The scoring engine
        # scores each sub-query independently and takes MAX across them
        # per tier (best-matching aspect wins).

        "must_have": [
            "Built and deployed embeddings-based retrieval systems in production using sentence-transformers BGE E5",
            "Production experience with vector databases hybrid search Pinecone Weaviate Qdrant Milvus FAISS Elasticsearch",
            "Managed embedding drift index refresh retrieval-quality regression in production",
            "Designed evaluation frameworks for ranking systems using NDCG MRR MAP A/B testing",
            "Strong Python engineering production-quality code for ML systems",
        ],

        "good_to_have": [
            "LLM fine-tuning experience LoRA QLoRA PEFT parameter-efficient methods",
            "Built learning-to-rank models XGBoost LambdaMART neural ranking",
            "Experience with HR-tech recruiting technology marketplace products",
            "Distributed systems large-scale ML inference optimization",
            "Open-source contributions AI ML with public repositories",
        ],

        "bonus": [
            "Shipped end-to-end ranking search recommendation system to real users at scale",
            "Strong opinions about retrieval evaluation hybrid vs dense and LLM integration",
            "6 to 8 years experience with 4 to 5 in applied ML AI at product companies",
        ],

        "disqualifiers": [
            "Career entirely in pure research academic lab without production deployment",
            "AI experience primarily calling OpenAI APIs LangChain without pre-LLM ML experience",
            "Has not written production code in 18 months moved into architecture or management",
            "Entire career at TCS Infosys Wipro Accenture Cognizant Capgemini consulting firms",
            "Primary expertise computer vision speech robotics without NLP information retrieval",
        ],

        "full_text": "",  # assembled at runtime from above
    }


# ===========================================================================
# Hybrid Search Engine
# ===========================================================================

class HybridSearchEngine:
    """
    Two-stage retrieval + reranking engine.

    Stage 1 (fast, broad):
        BM25 keyword search + FAISS HNSW vector search → fused via weighted RRF.

    Stage 2 (deep, narrow):
        Cross-Encoder reranking on the top-K from Stage 1.
        This is where "uses ChatGPT" vs "builds ML models" gets differentiated.
    """

    def __init__(
        self,
        biencoder_name: str = DEFAULT_BIENCODER,
        crossencoder_name: str = DEFAULT_CROSSENCODER,
    ):
        log.info("Loading Bi-Encoder: %s", biencoder_name)
        self.encoder = SentenceTransformer(biencoder_name)

        log.info("Loading Cross-Encoder: %s", crossencoder_name)
        self.cross_encoder = CrossEncoder(crossencoder_name)

        self.df: pd.DataFrame | None = None
        self.bm25: BM25Okapi | None = None
        self.faiss_index: faiss.Index | None = None
        self.corpus_texts: list[str] = []
        self.candidate_ids: list[str] = []
        self._tokenizer = self.encoder.tokenizer  # reuse for chunking

    # ------------------------------------------------------------------
    # Text Chunking (anti-truncation)
    # ------------------------------------------------------------------

    def chunk_text(self, text: str) -> list[str]:
        """
        Split a long text into overlapping chunks that fit within the
        Bi-Encoder's token limit.

        Uses the model's own tokenizer so chunk boundaries are exact
        (no guessing with word counts).

        Returns:
            List of text chunks. Minimum 1 chunk even for short texts.
        """
        tokens = self._tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= CHUNK_TOKEN_LIMIT:
            return [text]  # fits in one chunk, no splitting needed

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + CHUNK_TOKEN_LIMIT, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break
            start += CHUNK_TOKEN_LIMIT - CHUNK_OVERLAP  # slide with overlap

        return chunks

    # ------------------------------------------------------------------
    # Index Management
    # ------------------------------------------------------------------

    def load_or_build_indices(self, force_rebuild: bool = False, data_dir: str | None = None):
        """Load corpus, build/load BM25 + FAISS HNSW indexes.

        Args:
            force_rebuild: If True, rebuild indices even if cached versions exist.
            data_dir: Optional path to a dataset directory. If provided, loads
                      indices from this directory instead of the default processed/.
                      Used for custom uploaded datasets.
        """
        # Resolve paths — use custom data_dir if provided
        if data_dir:
            from pathlib import Path as P
            base = P(data_dir)
            text_path  = base / "text_corpus.pkl"
            bm25_path  = base / "bm25_index.pkl"
            faiss_path = base / "faiss_hnsw.index"
        else:
            text_path  = TEXT_CORPUS_PATH
            bm25_path  = BM25_INDEX_PATH
            faiss_path = FAISS_INDEX_PATH

        self._current_data_dir = str(data_dir) if data_dir else str(PROCESSED_DIR)

        log.info("Loading text corpus from %s ...", text_path)
        if not text_path.exists():
            raise FileNotFoundError(
                f"Corpus not found at {text_path}. "
                "Run preprocessing.py first to generate processed/text_corpus.pkl"
            )

        self.df = pd.read_pickle(text_path)
        self.corpus_texts = self.df["clean_text"].fillna("").tolist()
        self.candidate_ids = self.df["candidate_id"].tolist()
        log.info("Corpus loaded: %d candidates", len(self.corpus_texts))

        self._build_bm25(force_rebuild, bm25_path)
        self._build_faiss(force_rebuild, faiss_path)

    def _build_bm25(self, force: bool, bm25_path=None):
        bm25_path = bm25_path or BM25_INDEX_PATH
        if not force and bm25_path.exists():
            log.info("Loading cached BM25 index ...")
            with open(bm25_path, "rb") as f:
                loaded_bm25 = pickle.load(f)
            
            # Validation: ensure index matches corpus length
            if loaded_bm25.corpus_size == len(self.corpus_texts):
                self.bm25 = loaded_bm25
                return
            else:
                log.warning("BM25 index size (%d) mismatch with corpus (%d). Rebuilding...", loaded_bm25.corpus_size, len(self.corpus_texts))

        log.info("Building BM25 index ...")
        tokenized = [text.lower().split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)
        log.info("BM25 index saved to %s", bm25_path)

    def _build_faiss(self, force: bool, faiss_path=None):
        faiss_path = faiss_path or FAISS_INDEX_PATH
        if not force and faiss_path.exists():
            log.info("Loading cached FAISS HNSW index ...")
            loaded_faiss = faiss.read_index(str(faiss_path))
            
            # Validation: ensure index matches corpus length
            if loaded_faiss.ntotal == len(self.corpus_texts):
                self.faiss_index = loaded_faiss
                self.faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
                return
            else:
                log.warning("FAISS index size (%d) mismatch with corpus (%d). Rebuilding...", loaded_faiss.ntotal, len(self.corpus_texts))

        log.info("Building FAISS HNSW index (encoding %d documents) ...", len(self.corpus_texts))
        dim = self.encoder.get_sentence_embedding_dimension()

        # --- Chunk-and-Max-Pool Embedding Strategy ---
        # Each resume is split into overlapping chunks that fit the model's
        # 256-token window. Each chunk is embedded independently, then we
        # take the element-wise MAX across all chunk embeddings to produce
        # one vector per candidate. This ensures skills/education/certs
        # (which appear late in the text) are fully captured.
        embeddings = np.zeros((len(self.corpus_texts), dim), dtype=np.float32)
        total_chunks = 0

        for i, text in enumerate(self.corpus_texts):
            chunks = self.chunk_text(text)
            total_chunks += len(chunks)

            chunk_embs = self.encoder.encode(
                chunks,
                convert_to_numpy=True,
                normalize_embeddings=False,  # normalize AFTER pooling
            )

            # Max-pool across chunks: keeps the strongest signal per dimension
            pooled = np.max(chunk_embs, axis=0)

            # L2-normalize for cosine similarity via inner product
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm

            embeddings[i] = pooled

        log.info("Encoded %d candidates from %d chunks (avg %.1f chunks/candidate)",
                 len(self.corpus_texts), total_chunks, total_chunks / len(self.corpus_texts))

        # Save raw embeddings for potential reuse
        emb_path = faiss_path.parent / "embeddings.npy"
        np.save(str(emb_path), embeddings)

        # Build HNSW index with inner-product (= cosine sim on normalized vectors)
        self.faiss_index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self.faiss_index.hnsw.efConstruction = HNSW_EF_CONSTRUCT
        self.faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
        self.faiss_index.add(embeddings)

        faiss.write_index(self.faiss_index, str(faiss_path))
        log.info("FAISS index saved to %s (%d vectors, dim=%d)", faiss_path, embeddings.shape[0], dim)

    # ------------------------------------------------------------------
    # Stage 1: Retrieval
    # ------------------------------------------------------------------

    def search_bm25(self, query: str, top_k: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) of top-K BM25 matches."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return top_idx, scores[top_idx]

    def search_vector(self, query: str, top_k: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, similarities) of top-K HNSW matches."""
        query_emb = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        similarities, indices = self.faiss_index.search(query_emb, top_k)
        return indices[0], similarities[0]

    def hybrid_rrf(
        self,
        query: str,
        bm25_weight: float = 0.20,
        vector_weight: float = 0.80,
        retrieval_depth: int = 200,
        top_k: int = 100,
    ) -> pd.DataFrame:
        """
        Retrieves candidates using weighted Reciprocal Rank Fusion.

        RRF formula per candidate:
            score = bm25_w * 1/(k + rank_bm25) + vector_w * 1/(k + rank_vector)

        We retrieve `retrieval_depth` from each system to get good coverage,
        then return the top `top_k` after fusion.
        """
        bm25_idx, _   = self.search_bm25(query, top_k=retrieval_depth)
        vector_idx, _ = self.search_vector(query, top_k=retrieval_depth)

        rrf_scores: dict[int, float] = {}

        for rank, idx in enumerate(bm25_idx):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + bm25_weight / (RRF_K + rank + 1)

        for rank, idx in enumerate(vector_idx):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + vector_weight / (RRF_K + rank + 1)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        rows = [
            {
                "candidate_id":    self.candidate_ids[idx],
                "corpus_index":    idx,
                "clean_text":      self.corpus_texts[idx],
                "rrf_score":       rrf_scores[idx],
            }
            for idx in sorted_ids
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Stage 2: Cross-Encoder Reranking
    # ------------------------------------------------------------------

    def rerank(self, query: str, df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """
        Deep semantic reranking via Cross-Encoder with chunking.

        The Cross-Encoder processes (query, candidate_text) pairs jointly
        through all transformer layers, giving it the ability to understand
        contextual relationships that Bi-Encoders miss:
          - "proficiency in using ChatGPT and AI models"  → low score for ML Engineer JD
          - "contributed in building ML models end-to-end" → high score for ML Engineer JD

        To handle texts longer than the Cross-Encoder's 512-token limit,
        each candidate's text is split into chunks. The query is scored
        against every chunk, and the candidate's final score is the MAX
        across all chunks (the best-matching section wins).

        Returns df with an added 'rerank_score' column (sigmoid-normalized to 0-1).
        """
        texts = df["clean_text"].tolist()

        # Build all (query, chunk) pairs, tracking which candidate each belongs to
        all_pairs = []
        candidate_indices = []  # maps each pair back to its candidate row index

        for row_idx, text in enumerate(texts):
            chunks = self.chunk_text(text)
            for chunk in chunks:
                all_pairs.append([query, chunk])
                candidate_indices.append(row_idx)

        # Batch inference for speed — reset seeds for deterministic output
        torch.manual_seed(_SEED)
        np.random.seed(_SEED)
        raw_scores = self.cross_encoder.predict(all_pairs, batch_size=batch_size)

        # Aggregate: take MAX score across chunks for each candidate
        candidate_max_scores = np.full(len(texts), -np.inf)
        for pair_idx, score in enumerate(raw_scores):
            row_idx = candidate_indices[pair_idx]
            if score > candidate_max_scores[row_idx]:
                candidate_max_scores[row_idx] = score

        # Sigmoid normalization: raw logits → [0, 1]
        df = df.copy()
        df["rerank_score"] = 1.0 / (1.0 + np.exp(-candidate_max_scores))

        return df.sort_values("rerank_score", ascending=False).reset_index(drop=True)


# ===========================================================================
# Tiered JD Scoring (Sub-Query Decomposition + Calibration)
# ===========================================================================

def _score_sub_queries(
    engine: HybridSearchEngine,
    candidates_df: pd.DataFrame,
    queries: list | str,
    tier_name: str = "",
) -> np.ndarray:
    """
    Score candidates against one or more sub-queries for a single tier.

    If queries is a list: scores each sub-query independently, returns
    the MAX score across sub-queries for each candidate. This dramatically
    improves cross-encoder accuracy because each query is short and focused.

    If queries is a string: treats as a single query (backward compat).

    Returns: np.ndarray of shape (n_candidates,) with scores in [0, 1].
    """
    if isinstance(queries, str):
        queries = [queries]

    # Filter empty queries
    queries = [q for q in queries if q.strip()]
    if not queries:
        return np.zeros(len(candidates_df))

    n = len(candidates_df)
    cid_list = candidates_df["candidate_id"].tolist()

    # Score each sub-query independently
    sub_scores = np.zeros((len(queries), n))

    for qi, query in enumerate(queries):
        if tier_name:
            log.info("  Sub-query %d/%d: '%s'", qi + 1, len(queries), query[:60] + "..." if len(query) > 60 else query)
        reranked = engine.rerank(query, candidates_df)
        score_map = dict(zip(reranked["candidate_id"], reranked["rerank_score"]))
        for i, cid in enumerate(cid_list):
            sub_scores[qi, i] = score_map.get(cid, 0.0)

    # MAX across sub-queries: best-matching aspect wins per candidate
    return np.max(sub_scores, axis=0)


def calibrate_scores(scores: np.ndarray, floor: float = 0.30) -> np.ndarray:
    """
    Hybrid raw+rank score calibration for pre-selected candidate pools.

    Context: These 100 candidates were already selected from 100k via
    RRF retrieval — even the LAST-ranked candidate is in the top 0.1%.
    Pure rank-based mapping would incorrectly assign ~0.0 to #100.

    Approach: Blend raw cross-encoder signal with rank-based stretching.
    1. Raw-scaled component (70%): preserves the actual cross-encoder
       discrimination — if two candidates have similar raw scores,
       they get similar calibrated scores.
    2. Rank-stretched component (30%): provides additional separation
       when raw scores are tightly clustered.
    3. Floor = 0.30 reflects that the worst candidate in the pool
       is still a strong match (top 0.1% of 100k).
    4. Gentle softening (not zeroing) for truly low raw scores.
    """
    if len(scores) <= 1:
        return scores

    n = len(scores)
    ceiling = 0.95  # reserve 0.95-1.0 for penalty/bonus adjustments

    # --- Component 1: Raw-scaled (preserves cross-encoder signal) ---
    raw_min, raw_max = scores.min(), scores.max()
    raw_range = raw_max - raw_min
    if raw_range > 0:
        raw_scaled = (scores - raw_min) / raw_range  # [0, 1]
    else:
        raw_scaled = np.ones(n) * 0.5  # all tied

    # Map to [floor, ceiling]
    raw_component = floor + raw_scaled * (ceiling - floor)

    # --- Component 2: Rank-stretched (adds separation) ---
    ranks = np.argsort(np.argsort(scores))  # rank of each element (0-indexed)
    percentiles = ranks / (n - 1) if n > 1 else np.zeros(n)
    # Mild power curve for extra top-end separation
    rank_curved = np.power(percentiles, 0.8)
    rank_component = floor + rank_curved * (ceiling - floor)

    # --- Blend: 70% raw signal, 30% rank stretch ---
    calibrated = 0.70 * raw_component + 0.30 * rank_component

    # --- Gentle softening for truly poor raw scores ---
    # Only suppress candidates whose raw score is in the bottom 10%
    # of the raw range (genuinely weak matches). Use a soft sigmoid
    # ramp instead of a hard multiplier to avoid zeroing anything.
    if raw_range > 0:
        # Position in [0, 1] relative to pool
        raw_position = (scores - raw_min) / raw_range
        # Sigmoid ramp: 0.5 at position=0.05, ~1.0 above position=0.15
        softening = 1.0 / (1.0 + np.exp(-20 * (raw_position - 0.05)))
        # Floor the softening at 0.5 so even the worst candidate
        # keeps at least 50% of their calibrated score
        softening = np.clip(softening, 0.50, 1.0)
        calibrated = calibrated * softening

    return np.clip(calibrated, floor * 0.5, ceiling)


def compute_tiered_score(
    engine: HybridSearchEngine,
    candidates_df: pd.DataFrame,
    jd: dict,
    top_k: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Scores each candidate against each JD tier using focused sub-queries,
    then combines with weighted formula + score calibration.

    Architecture:
      1. For each tier, score all sub-queries independently via Cross-Encoder
      2. Per candidate: MAX across sub-queries per tier (best aspect wins)
      3. Weighted combination across tiers with must_have dominance
      4. Disqualifier penalty subtraction
      5. Hybrid raw+rank score calibration to [0.15, 0.95]

    The must_have tier is weighted heavily (1.0) because the JD is explicit:
    these are ABSOLUTE requirements, not "nice to haves."
    """
    n = len(candidates_df)
    weighted_sum = np.zeros(n)
    active_weight_sum = 0.0
    tier_scores = {}  # for verbose output

    for tier, weight in TIER_WEIGHTS.items():
        queries = jd.get(tier, "" if isinstance(jd.get(tier), str) else [])

        # Handle both string and list formats
        if isinstance(queries, str) and not queries.strip():
            log.info("Tier '%s' is empty — skipping", tier)
            continue
        if isinstance(queries, list) and not queries:
            log.info("Tier '%s' is empty — skipping", tier)
            continue

        log.info("Reranking %d candidates against tier '%s' (weight=%.2f, %d sub-queries) ...",
                 n, tier, weight, len(queries) if isinstance(queries, list) else 1)

        scores = _score_sub_queries(engine, candidates_df, queries, tier_name=tier)
        tier_scores[tier] = scores

        weighted_sum += weight * scores
        active_weight_sum += weight

    # Normalize by the sum of active weights
    if active_weight_sum > 0:
        raw_scores = weighted_sum / active_weight_sum
    else:
        raw_scores = weighted_sum

    # --- Disqualifier penalty ---
    disq_queries = jd.get("disqualifiers", "" if isinstance(jd.get("disqualifiers"), str) else [])
    if (isinstance(disq_queries, str) and disq_queries.strip()) or \
       (isinstance(disq_queries, list) and disq_queries):
        log.info("Applying disqualifier penalty (%.2f multiplier) ...", DISQUALIFIER_PENALTY)
        disq_scores = _score_sub_queries(engine, candidates_df, disq_queries, tier_name="disqualifiers")

        # Penalty proportional to disqualifier match strength
        for i in range(n):
            penalty = DISQUALIFIER_PENALTY * disq_scores[i]
            raw_scores[i] -= penalty

    # Clamp to [0, 1] before calibration
    raw_scores = np.clip(raw_scores, 0.0, 1.0)

    if verbose:
        log.info("\n--- Pre-calibration score distribution ---")
        log.info("  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                 raw_scores.mean(), raw_scores.std(), raw_scores.min(), raw_scores.max())
        for tier, ts in tier_scores.items():
            log.info("  tier '%s': mean=%.4f max=%.4f", tier, ts.mean(), ts.max())

    # --- Percentile-based calibration ---
    calibrated = calibrate_scores(raw_scores)

    if verbose:
        log.info("\n--- Post-calibration score distribution ---")
        log.info("  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                 calibrated.mean(), calibrated.std(), calibrated.min(), calibrated.max())

    result = candidates_df.copy()
    result["semantic_score"] = calibrated
    result["raw_semantic"]   = raw_scores  # keep raw for debugging
    result = result.sort_values("semantic_score", ascending=False).reset_index(drop=True)

    return result


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Semantic Scorer — scores candidates against a Job Description"
    )
    parser.add_argument(
        "--jd", type=str, default=None,
        help="Path to a .docx JD file (auto-parsed into tiers)"
    )
    parser.add_argument(
        "--jd-json", type=str, default=None,
        help="Path to a pre-structured .json JD file with keys: must_have, good_to_have, bonus, disqualifiers"
    )
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="Number of candidates to retrieve and score (default: 100)"
    )
    parser.add_argument(
        "--retrieval-depth", type=int, default=200,
        help="How many candidates each retriever (BM25, HNSW) fetches before RRF fusion (default: 200)"
    )
    parser.add_argument(
        "--bm25-weight", type=float, default=0.20,
        help="Weight for BM25 in RRF fusion (default: 0.20)"
    )
    parser.add_argument(
        "--vector-weight", type=float, default=0.80,
        help="Weight for vector search in RRF fusion (default: 0.80)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: processed/semantic_scores.csv)"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild of BM25 and FAISS indexes (ignore cache)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed statistics"
    )
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # 1. Parse JD
    if args.jd:
        log.info("Parsing JD from .docx: %s", args.jd)
        jd = parse_jd_from_docx(args.jd)
    elif args.jd_json:
        log.info("Loading JD from .json: %s", args.jd_json)
        jd = parse_jd_from_json(args.jd_json)
    else:
        log.info("No JD file provided — using hardcoded Redrob AI Engineer JD")
        jd = get_default_jd()

    # 2. Initialize engine and build/load indexes
    engine = HybridSearchEngine()
    engine.load_or_build_indices(force_rebuild=args.rebuild)

    # 3. Broad retrieval: flatten sub-queries into single string for RRF
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

    log.info("[Stage 1] Hybrid RRF retrieval (BM25=%.0f%%, Vector=%.0f%%) ...",
             args.bm25_weight * 100, args.vector_weight * 100)

    candidates = engine.hybrid_rrf(
        query=broad_query,
        bm25_weight=args.bm25_weight,
        vector_weight=args.vector_weight,
        retrieval_depth=args.retrieval_depth,
        top_k=args.top_k,
    )
    log.info("Retrieved %d candidates from hybrid search", len(candidates))

    # 4. Tiered reranking with Cross-Encoder
    log.info("[Stage 2] Cross-Encoder reranking with JD tier weights ...")
    results = compute_tiered_score(engine, candidates, jd, top_k=args.top_k, verbose=args.verbose)

    # 5. Normalize rrf_score to [0, 1] via min-max scaling
    #    so all scores are on the same scale for downstream combination
    rrf_min = results["rrf_score"].min()
    rrf_max = results["rrf_score"].max()
    if rrf_max > rrf_min:
        results["rrf_score"] = (results["rrf_score"] - rrf_min) / (rrf_max - rrf_min)
    else:
        results["rrf_score"] = 1.0  # all tied → all get max score

    # 6. Save output — both CSV (human-readable) and PKL (preserves dtypes)
    output_cols = ["candidate_id", "semantic_score", "rrf_score"]
    output_df = results[output_cols].copy()

    if args.output:
        csv_path = Path(args.output)
        pkl_path = csv_path.with_suffix(".pkl")
    else:
        csv_path = OUTPUT_CSV_PATH
        pkl_path = OUTPUT_PKL_PATH

    output_df.to_csv(csv_path, index=False)
    output_df.to_pickle(pkl_path)
    log.info("Scores saved to %s  and  %s", csv_path, pkl_path)
    log.info("  semantic_score range: [%.4f, %.4f]",
             output_df["semantic_score"].min(), output_df["semantic_score"].max())
    log.info("  rrf_score range:      [%.4f, %.4f]",
             output_df["rrf_score"].min(), output_df["rrf_score"].max())

    # 7. Print top results
    print("\n" + "=" * 60)
    print(f"  TOP {min(10, len(results))} CANDIDATES BY SEMANTIC SCORE")
    print("=" * 60)
    for i, row in results.head(10).iterrows():
        print(f"  #{i+1:>2}  {row['candidate_id']}  semantic={row['semantic_score']:.4f}  rrf={row['rrf_score']:.4f}")
        snippet = row["clean_text"][:120].replace("\n", " ")
        print(f"       {snippet}...")
        print()


if __name__ == "__main__":
    main()
