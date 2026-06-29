# 🚀 Deployment Guide — Resume Shortlister

> **TL;DR** — This is a CPU-heavy Python/Flask app that loads ~500 MB of pre-computed
> embeddings at startup. It cannot run on serverless/edge platforms (Vercel, Netlify).
> **Best options: Render (free tier), Railway, or a small VPS.**

---

## Why Not Vercel?

Vercel is built for Node.js serverless functions (max 50 MB bundle, 10–30 s cold start,
no persistent disk). This app:

- Loads a 500 MB+ FAISS HNSW index into RAM
- Requires `torch`, `sentence-transformers`, and `faiss-cpu` (not pip-installable in
  a Vercel lambda)
- Needs persistent `/processed/` files across requests

**Verdict: Vercel will not work.** Use one of the options below.

---

## ✅ Option 1 — Render (Recommended for Free Hosting)

[render.com](https://render.com) — Free tier: 512 MB RAM, spins down after 15 min idle.

### Steps

1. Push repo to GitHub (already done).

2. Go to **render.com → New → Web Service → Connect GitHub repo**.

3. Fill in:
   | Field | Value |
   |---|---|
   | **Runtime** | Python 3 |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `python app.py --port $PORT` |
   | **Instance Type** | Free (or Starter $7/mo for always-on) |

4. Set **Environment Variables** (if needed):
   ```
   PYTHONUNBUFFERED=1
   ```

5. **Important:** The `processed/` folder with embeddings must be present.
   Since `candidates.jsonl` is excluded from git, you have two options:
   - **Option A (Recommended):** Pre-generate and commit just the processed `.pkl`/`.npy`
     files (they're ~200 MB, use Git LFS — see below).
   - **Option B:** Add a build script that downloads the processed files from
     a cloud storage bucket (S3, GCS, etc.) before startup.

6. Click **Deploy** — Render builds and runs `app.py`.

### Git LFS for Large Processed Files

```bash
# Install Git LFS once
git lfs install

# Track processed files
git lfs track "processed/*.npy"
git lfs track "processed/*.pkl"
git lfs track "processed/*.index"

# Commit and push
git add .gitattributes
git add -f processed/
git commit -m "Add processed embeddings via LFS"
git push origin Main
```

---

## ✅ Option 2 — Railway (Easiest Setup)

[railway.app](https://railway.app) — $5 free credit/month, persistent disk, no spin-down.

### Steps

1. Go to **railway.app → New Project → Deploy from GitHub repo**.
2. Railway auto-detects Python. Set start command:
   ```
   python app.py --port $PORT
   ```
3. Add a **Volume** (persistent disk) at `/app/processed` so embeddings survive redeploys.
4. On first deploy, run the pipeline to generate files:
   ```bash
   # In Railway shell
   python run_pipeline.py --rebuild
   ```
5. Subsequent deploys reuse the volume — instant startup.

---

## ✅ Option 3 — Google Cloud Run (Scalable, Pay-per-use)

Best for demos that need real scale.

### Steps

1. **Build Docker image:**
   ```dockerfile
   # Dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8080
   CMD ["python", "app.py", "--port", "8080"]
   ```

2. **Build & push:**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT/resume-shortlister
   ```

3. **Deploy:**
   ```bash
   gcloud run deploy resume-shortlister \
     --image gcr.io/YOUR_PROJECT/resume-shortlister \
     --platform managed \
     --region us-central1 \
     --memory 2Gi \
     --allow-unauthenticated
   ```

4. Mount a GCS bucket for the `processed/` folder using **Cloud Storage FUSE** or
   pre-bake the processed files into the Docker image.

---

## ✅ Option 4 — Hugging Face Spaces (Zero Setup, Free)

[huggingface.co/spaces](https://huggingface.co/spaces) — Ideal for hackathon demos.

### Steps

1. Create a new Space → **Gradio** or **Docker** SDK.
2. Choose Docker, add a `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 7860
   CMD ["python", "app.py", "--port", "7860"]
   ```
3. Upload processed files to the Space repo (up to 10 GB supported via Git LFS).
4. HF Spaces keeps the service alive for free on CPU hardware.

> **This is the best option for a hackathon demo** — free, always-on, public URL,
> and supports large model files natively.

---

## Running Locally (Always Works)

```bash
# 1. Clone repo
git clone -b Main https://github.com/uditgarg007/Resume-Shortlister.git
cd Resume-Shortlister

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate processed files (first time only, ~5 min)
python run_pipeline.py --rebuild

# 5. Start the server
python app.py

# Open http://localhost:5000
```

---

## Summary Comparison

| Platform | Cost | Setup | Persistent Disk | Cold Start | Best For |
|---|---|---|---|---|---|
| **HF Spaces** | Free | ⭐ Easiest | ✅ | Fast | Hackathon demos |
| **Render** | Free / $7 | Easy | ⚠️ (LFS) | Slow (free) | Public demos |
| **Railway** | ~$5/mo | Easy | ✅ Volume | Fast | Production |
| **Cloud Run** | Pay-per-use | Medium | ✅ GCS | Medium | Scale |
| **Vercel** | — | — | ❌ | — | ❌ Not compatible |

**Recommendation for this hackathon:** Deploy on **Hugging Face Spaces** with Docker.
It's free, handles large files, has a public URL, and requires zero credit card.
