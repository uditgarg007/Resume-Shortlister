"""
dataset_manager.py — Custom Dataset Upload, Embedding & Management

Allows users to upload their own candidate JSONL/JSON files, preprocess
and embed them into named datasets, and later select which dataset to
rank against any JD.

Dataset directory layout:
  datasets/
    my_dataset/
      metadata.json          ← name, candidate_count, created_at, status
      text_corpus.pkl        ← preprocessed text
      numeric_signals.pkl    ← extracted numeric features
      faiss_hnsw.index       ← FAISS HNSW vector index
      bm25_index.pkl         ← BM25 keyword index
      embeddings.npy         ← raw embedding vectors

The "default" dataset (processed/) is always available and refers to the
built-in 100k candidate pool.

Usage (via API — called from app.py):
  manager = DatasetManager(engine)
  manager.create_dataset("my_team", candidates_jsonl_path)
  manager.list_datasets()
  manager.delete_dataset("my_team")
"""

import json
import logging
import os
import pickle
import shutil
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi

from preprocessing import build_text_corpus, build_numeric_signals

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
PROCESSED_DIR = BASE_DIR / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dataset_manager")

# HNSW parameters (same as hybrid_search.py)
HNSW_M = 32
HNSW_EF_CONSTRUCT = 200
HNSW_EF_SEARCH = 128

# Chunking parameters (same as hybrid_search.py)
CHUNK_TOKEN_LIMIT = 200
CHUNK_OVERLAP = 50


class DatasetManager:
    """
    Manages custom candidate datasets — preprocessing, embedding, and storage.

    Each dataset is a self-contained directory with all indices needed for
    the ranking pipeline. The HybridSearchEngine can load any dataset by
    pointing to its directory.
    """

    def __init__(self, encoder=None):
        """
        Args:
            encoder: A SentenceTransformer model instance (shared with
                     HybridSearchEngine to avoid loading the model twice).
                     If None, will be loaded on first use.
        """
        self._encoder = encoder
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading Bi-Encoder for dataset embedding...")
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._encoder

    # ------------------------------------------------------------------
    # List / Inspect
    # ------------------------------------------------------------------

    def list_datasets(self) -> list[dict]:
        """
        List all available datasets (including the built-in default).

        Returns list of dicts with: name, candidate_count, created_at, status
        """
        datasets = []

        # Built-in default dataset
        if (PROCESSED_DIR / "text_corpus.pkl").exists():
            try:
                default_df = pd.read_pickle(PROCESSED_DIR / "text_corpus.pkl")
                datasets.append({
                    "name": "default",
                    "candidate_count": len(default_df),
                    "created_at": "built-in",
                    "status": "ready",
                    "is_default": True,
                })
            except Exception:
                datasets.append({
                    "name": "default",
                    "candidate_count": 0,
                    "created_at": "built-in",
                    "status": "error",
                    "is_default": True,
                })

        # Custom datasets
        if DATASETS_DIR.exists():
            for entry in sorted(DATASETS_DIR.iterdir()):
                if entry.is_dir():
                    meta_path = entry / "metadata.json"
                    if meta_path.exists():
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            datasets.append({
                                "name": meta.get("name", entry.name),
                                "candidate_count": meta.get("candidate_count", 0),
                                "created_at": meta.get("created_at", "unknown"),
                                "status": meta.get("status", "unknown"),
                                "is_default": False,
                            })
                        except Exception:
                            datasets.append({
                                "name": entry.name,
                                "candidate_count": 0,
                                "created_at": "unknown",
                                "status": "error",
                                "is_default": False,
                            })

        return datasets

    def get_dataset_dir(self, name: str) -> Path:
        """Get the directory path for a dataset. 'default' maps to processed/."""
        if name == "default":
            return PROCESSED_DIR
        return DATASETS_DIR / name

    def dataset_exists(self, name: str) -> bool:
        """Check if a dataset exists and is ready."""
        if name == "default":
            return (PROCESSED_DIR / "text_corpus.pkl").exists()
        meta_path = DATASETS_DIR / name / "metadata.json"
        if not meta_path.exists():
            return False
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("status") == "ready"

    # ------------------------------------------------------------------
    # Create (preprocess + embed)
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        name: str,
        candidates: list[dict],
        progress_callback=None,
    ) -> dict:
        """
        Create a new dataset from a list of candidate dicts.

        Steps:
          1. Preprocess text (build_text_corpus)
          2. Extract numeric signals (build_numeric_signals)
          3. Embed all candidates (chunk + encode + max-pool)
          4. Build FAISS HNSW index
          5. Build BM25 index
          6. Save metadata

        Args:
            name: Dataset name (alphanumeric + hyphens/underscores)
            candidates: List of candidate dicts (same schema as candidates.jsonl)
            progress_callback: Optional fn(step, total, message) for progress updates

        Returns:
            dict with dataset metadata
        """
        # Validate name
        safe_name = "".join(c for c in name if c.isalnum() or c in "-_").strip()
        if not safe_name:
            raise ValueError("Dataset name must contain alphanumeric characters")
        if safe_name == "default":
            raise ValueError("Cannot overwrite the default dataset")

        dataset_dir = DATASETS_DIR / safe_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        def _progress(step, total, msg):
            if progress_callback:
                progress_callback(step, total, msg)
            log.info("[%d/%d] %s", step, total, msg)

        total_steps = 5

        # Save initial metadata
        meta = {
            "name": safe_name,
            "candidate_count": len(candidates),
            "created_at": datetime.now().isoformat(),
            "status": "processing",
        }
        self._save_metadata(dataset_dir, meta)

        try:
            # Step 1: Preprocess text
            _progress(1, total_steps, f"Preprocessing {len(candidates)} candidates...")
            text_df = build_text_corpus(candidates)
            text_df.to_pickle(dataset_dir / "text_corpus.pkl")

            # Step 2: Extract numeric signals
            _progress(2, total_steps, "Extracting numeric signals...")
            num_df = build_numeric_signals(candidates)
            num_df.to_pickle(dataset_dir / "numeric_signals.pkl")

            # Step 3: Embed all candidates
            _progress(3, total_steps, f"Embedding {len(text_df)} candidates (this may take a while)...")
            corpus_texts = text_df["clean_text"].fillna("").tolist()
            embeddings = self._embed_corpus(corpus_texts)
            np.save(str(dataset_dir / "embeddings.npy"), embeddings)

            # Step 4: Build FAISS HNSW index
            _progress(4, total_steps, "Building FAISS HNSW vector index...")
            dim = embeddings.shape[1]
            faiss_index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            faiss_index.hnsw.efConstruction = HNSW_EF_CONSTRUCT
            faiss_index.hnsw.efSearch = HNSW_EF_SEARCH
            faiss_index.add(embeddings)
            faiss.write_index(faiss_index, str(dataset_dir / "faiss_hnsw.index"))

            # Step 5: Build BM25 index
            _progress(5, total_steps, "Building BM25 keyword index...")
            tokenized = [text.lower().split() for text in corpus_texts]
            bm25 = BM25Okapi(tokenized)
            with open(dataset_dir / "bm25_index.pkl", "wb") as f:
                pickle.dump(bm25, f)

            # Update metadata
            meta["status"] = "ready"
            meta["embedding_dim"] = dim
            self._save_metadata(dataset_dir, meta)

            log.info("Dataset '%s' created successfully (%d candidates)", safe_name, len(candidates))
            return meta

        except Exception as e:
            meta["status"] = "error"
            meta["error"] = str(e)
            self._save_metadata(dataset_dir, meta)
            log.error("Failed to create dataset '%s': %s", safe_name, e)
            raise

    def _embed_corpus(self, corpus_texts: list[str]) -> np.ndarray:
        """Embed all texts using chunk-and-max-pool strategy."""
        encoder = self.encoder
        tokenizer = encoder.tokenizer
        dim = encoder.get_sentence_embedding_dimension()
        embeddings = np.zeros((len(corpus_texts), dim), dtype=np.float32)

        for i, text in enumerate(corpus_texts):
            chunks = self._chunk_text(text, tokenizer)
            chunk_embs = encoder.encode(
                chunks,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            # Max-pool across chunks
            pooled = np.max(chunk_embs, axis=0)
            # L2-normalize for cosine similarity
            norm = np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm
            embeddings[i] = pooled

            if (i + 1) % 500 == 0:
                log.info("  Embedded %d / %d candidates...", i + 1, len(corpus_texts))

        return embeddings

    @staticmethod
    def _chunk_text(text: str, tokenizer) -> list[str]:
        """Split text into overlapping chunks."""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= CHUNK_TOKEN_LIMIT:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + CHUNK_TOKEN_LIMIT, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            if end >= len(tokens):
                break
            start += CHUNK_TOKEN_LIMIT - CHUNK_OVERLAP
        return chunks

    @staticmethod
    def _save_metadata(dataset_dir: Path, meta: dict):
        with open(dataset_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_dataset(self, name: str) -> bool:
        """Delete a custom dataset. Cannot delete the default."""
        if name == "default":
            raise ValueError("Cannot delete the default dataset")

        dataset_dir = DATASETS_DIR / name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            log.info("Deleted dataset '%s'", name)
            return True
        return False

    # ------------------------------------------------------------------
    # Parse uploaded file
    # ------------------------------------------------------------------

    @staticmethod
    def parse_upload(file_content: str, filename: str) -> list[dict]:
        """
        Parse an uploaded file into a list of candidate dicts.

        Supports:
          - .jsonl  — one JSON object per line (same as candidates.jsonl)
          - .json   — JSON array of candidate objects
          - .csv    — must have 'candidate_id' and 'resume_text' columns
                      (simplified format for users without full schema)

        For CSV uploads with just resume_text, we synthesize a minimal
        candidate dict that's compatible with the preprocessing pipeline.
        """
        filename_lower = filename.lower()

        if filename_lower.endswith(".jsonl"):
            candidates = []
            for line in file_content.strip().split("\n"):
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
            return candidates

        elif filename_lower.endswith(".json"):
            data = json.loads(file_content)
            if isinstance(data, list):
                return data
            else:
                raise ValueError("JSON file must contain an array of candidate objects")

        elif filename_lower.endswith(".csv"):
            import io
            df = pd.read_csv(io.StringIO(file_content))

            if "candidate_id" not in df.columns:
                # Auto-generate IDs
                df["candidate_id"] = [f"CAND_{i:06d}" for i in range(len(df))]

            candidates = []
            for _, row in df.iterrows():
                cand = {"candidate_id": row["candidate_id"]}

                # If they have the full schema fields, use them
                if "headline" in df.columns or "summary" in df.columns:
                    cand["profile"] = {
                        "headline": row.get("headline", ""),
                        "summary": row.get("summary", ""),
                        "current_title": row.get("current_title", ""),
                        "current_company": row.get("current_company", ""),
                        "location": row.get("location", ""),
                        "years_of_experience": row.get("years_of_experience", 0),
                    }
                elif "resume_text" in df.columns:
                    # Simplified: just resume text → put it all in summary
                    cand["profile"] = {
                        "headline": row.get("name", ""),
                        "summary": row.get("resume_text", ""),
                        "years_of_experience": row.get("years_of_experience", 0),
                    }
                else:
                    # Try to concatenate all non-ID text columns
                    text_cols = [c for c in df.columns if c != "candidate_id"]
                    combined = " ".join(str(row[c]) for c in text_cols if pd.notna(row[c]))
                    cand["profile"] = {"summary": combined}

                cand["redrob_signals"] = {}
                cand["career_history"] = []
                cand["education"] = []
                cand["skills"] = []
                cand["certifications"] = []
                candidates.append(cand)

            return candidates

        else:
            raise ValueError(f"Unsupported file format: {filename}. Use .jsonl, .json, or .csv")
