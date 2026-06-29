# Project Status & Capabilities

This document outlines the current capabilities of the **Resume Shortlister** engine, as well as the areas that require future development for enterprise-scale readiness.

## What We Have Built (Current Capabilities)

1. **Intelligent Data Pipeline**
   - Automatically flattens complex nested JSON/JSONL candidate profiles into clean text and structured numeric signals.
   - Handles missing data gracefully with sentinel value replacement.

2. **Advanced Semantic Engine**
   - **Bi-Encoder Chunking**: Splits long resumes into overlapping chunks, embeds them via `all-MiniLM-L6-v2`, and uses Max-Pooling to capture all skills regardless of where they appear in the text.
   - **Hybrid Retrieval**: Combines FAISS HNSW vector search (Semantic) with BM25 Okapi (Keyword) using Reciprocal Rank Fusion (RRF).

3. **Cross-Encoder Reranking (Tiered Analysis)**
   - Doesn't just do generic matching; breaks down JDs into "Must Have", "Good to Have", and "Bonus" sub-queries.
   - Uses `ms-marco-MiniLM-L-6-v2` to deeply analyze the semantic relationship between the candidate's resume and the specific JD aspect.

4. **Calibrated Scoring & Penalty Engine**
   - **Raw+Rank Hybrid Calibration**: Ensures mathematically sound scoring curves. The bottom candidate in a top-1% pool receives a fair floor score (e.g., 0.30) instead of 0.0.
   - **Exponential Diminishing Returns**: Bonuses applied to top candidates don't artificially cap out at 0.99 causing tie-breaks. The math preserves semantic differentiation at the absolute top of the leaderboard.
   - 9 built-in, configurable business rule penalties (Ghosting, Job Hopping, Role Mismatch, etc.).

5. **Web Application & Custom Datasets**
   - Flask-based web interface with sliders for real-time penalty and weight tuning.
   - **Dynamic Custom Datasets**: Users can upload `.csv`, `.json`, or `.jsonl` files via the UI. The app instantly builds a customized text corpus, extracts numeric signals, embeds the data, and builds FAISS/BM25 indices on the fly.

---

## What It Still Needs (Future Roadmap)

While the core AI logic is highly robust, the infrastructure needs a few upgrades before it can be deployed as an enterprise SaaS product.

### 1. Asynchronous Task Queue (Critical)
**Current State**: When a user uploads a custom dataset, the Flask web thread blocks until the entire dataset is embedded and indexed. For 1,000+ candidates, this takes several minutes and will trigger a `504 Gateway Timeout` on production servers like Nginx/Gunicorn.
**Required Fix**: Implement **Celery** with **Redis** or RabbitMQ. When a user uploads a dataset, the API should return a `job_id` instantly, and the frontend should poll `/api/status/<job_id>` for progress updates while a background worker handles the heavy embedding calculations.

### 2. Relational Database Integration
**Current State**: Metadata and file paths are managed via the local filesystem (`datasets/my_team/metadata.json`).
**Required Fix**: Migrate state management to **PostgreSQL** or **SQLite**. This will allow for robust user authentication, associating specific datasets with specific users/companies, and persisting historical JD configurations.

### 3. PDF/DOCX Parsing Support
**Current State**: The Custom Dataset upload strictly accepts structured JSON/JSONL or a CSV with a `resume_text` column.
**Required Fix**: Integrate libraries like `PyMuPDF` or `pdfminer.six` to allow users to drag-and-drop raw `.pdf` or `.docx` resume files. The system would extract the text automatically and synthesize the candidate dict.

### 4. Advanced Frontend Features
**Current State**: The UI is a functional single-page application showing the Top K candidates.
**Required Fix**: 
- **Pagination**: Implement pagination or infinite scroll for browsing beyond the Top 100.
- **Explainability UI**: Click a candidate row to open a sidebar showing *why* they matched, highlighting the specific resume chunks that triggered the Cross-Encoder scores.

### 5. Multi-GPU Support for Inference
**Current State**: The embeddings and Cross-Encoder currently run on CPU. While optimized, this limits throughput.
**Required Fix**: Add PyTorch device-detection logic to push operations to `cuda` or `mps` automatically, significantly speeding up both initial dataset embedding and real-time candidate reranking.
