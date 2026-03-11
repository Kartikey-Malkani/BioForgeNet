# BioForgeNet — Scientific Image Integrity Platform

> Premium product website showcasing your trained image forgery detection model (CV Dice: 0.4654, 5-fold ensemble)

**Status:** ✅ Production Ready | **Deployment:** Render + Cloudflare | **Demo:** http://localhost:5500

---

## 🚀 Quick Start (Local Development)

### Terminal 1: Frontend
```bash
cd website
python -m http.server 5500
# Open http://localhost:5500 in browser
```

### Terminal 2: Backend API
```powershell
cd website
pip install -r requirements.txt
.\start_api.ps1            # recommended — handles Windows UTF-8 encoding
# or manually:
# $env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"; uvicorn api.app:app --port 8000
```

Both servers ready! Visit http://localhost:5500 to see the demo.

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **README.md** (this file) | Quick start & command reference |
| **DEPLOYMENT.md** | Complete deployment guide + architecture overview |
| **COMPLETION_CHECKLIST.md** | Next steps & production activation guide |

**→ Start with COMPLETION_CHECKLIST.md for immediate next steps**

---

## 🎯 Real Model Activation

The backend auto-detects and uses real trained checkpoints when available.

### How It Works
1. Searches for `best_fold*.pth` files in these locations (in order):
   - `VERISIGHT_CHECKPOINT_DIR` (env var, if set)
   - `../outputs/checkpoints/`
   - `../checkpoints/`
   - `/kaggle/working/checkpoints/` (Kaggle compatibility)

2. If found: Runs real ensemble + TTA inference (2-4s per image)
3. If not found: Falls back to Laplacian edge detection (50ms per image)

### To Activate Real Model
1. **Copy your trained checkpoints:**
   ```bash
   mkdir outputs\checkpoints
   # Copy best_fold0.pth ... best_fold4.pth to the new folder
   ```

2. **Restart API:**
   ```bash
   # Stop current API (Ctrl+C)
   uvicorn api.app:app --reload --port 8000
   ```

3. **Verify:**
   - Check: http://localhost:8000/model-status
   - Should show: `"mode": "real-checkpoint"` (not "proxy-fallback")
   - Frontend displays: "Backend: connected (real model)" in green

4. **Optional:** Set custom checkpoint path
   ```bash
   set VERISIGHT_CHECKPOINT_DIR=D:\path\to\checkpoints
   uvicorn api.app:app --reload --port 8000
   ```

5. **Optional:** limit loaded folds (useful for low-memory hosting)
   ```bash
   set VERISIGHT_MAX_FOLDS=1
   uvicorn api.app:app --reload --port 8000
   ```

---

## 🤗 Hugging Face Spaces (Free-Friendly Real Checkpoint Mode)

Use this when Render free RAM is not enough for local checkpoint storage.

### 1) Create repos on Hugging Face
- Create one **Docker Space** (for API), for example: `yourname/bioforgenet-api`
- Create one **Model or Dataset repo** (for checkpoints), for example: `yourname/bioforgenet-checkpoints`
- Upload `best_fold*.pth` files to that checkpoint repo

### 2) Space source folder
- Use `website/` as the Space project root (it already contains `Dockerfile` and `requirements.txt`)

### 3) Space variables/secrets
In Space **Settings → Variables and secrets**, set:
- `CORS_ORIGINS=https://bioforgenet.live,https://www.bioforgenet.live`
- `HF_REPO_ID=yourname/bioforgenet-checkpoints`
- `HF_REPO_TYPE=model` (or `dataset`)
- `VERISIGHT_CHECKPOINT_DIR=/tmp/checkpoints`
- `VERISIGHT_MAX_FOLDS=1` (start with 1, increase after memory testing)
- `HF_ALLOW_PATTERNS=best_fold*.pth`
- `HF_TOKEN=...` (only if checkpoint repo is private)

### 4) Verify mode
- Open `https://<your-space>.hf.space/model-status`
- Expect: `"mode": "real-checkpoint"`

### 5) Point frontend to Space API
- Set `api-base` in `index.html` to your Space URL, for example:
  `https://yourname-bioforgenet-api.hf.space`

### Notes
- On free resources, start with 1 fold and increase gradually.
- If startup is slow, keep fallback mode available as backup.

---

## 🔌 API Reference

### GET `/health`
Backend status + inference mode
```json
{
  "status": "ok",
  "mode": "real-checkpoint",
  "reason": "Found 5 best_fold*.pth checkpoints"
}
```

### GET `/model-status`
Detailed status info
```json
{
  "mode": "real-checkpoint",
  "reason": "Found 5 best_fold*.pth checkpoints",
  "device": "cuda",
  "src_dir": "D:\\image_forgery_detection\\src"
}
```

### POST `/analyze`
Analyze image, return JSON + base64 mask
```bash
curl -X POST http://localhost:8000/analyze -F "file=@image.jpg"
```
Response:
```json
{
  "prediction": "forged",
  "risk_score": 0.78,
  "tamper_area_pct": 12.4,
  "confidence": 0.89,
  "engine": "real-checkpoint (5-fold ensemble TTA)",
  "mask_png_base64": "iVBORw0KGgo..."
}
```

### POST `/analyze-mask` (NEW)
Download raw binary forensic mask as PNG
```bash
curl -X POST http://localhost:8000/analyze-mask -F "file=@image.jpg" \
  --output forgery_mask.png
```
Response headers:
- `X-Prediction`: "forged" or "authentic"
- `X-Risk-Score`: Risk percentage
- `X-Engine`: Inference backend used

**Use for:** Compliance reports, forensic documentation, audit trails

---

## 🎨 Frontend Features

✅ **Premium UI**
- Glassmorphism dark theme
- Gradient buttons & glass panels
- Responsive design (desktop, tablet, mobile)

✅ **Interactive Demo**
- Drag-drop image upload
- Real-time analysis
- Heatmap visualization

✅ **Results Display**
- Prediction class (FORGED / AUTHENTIC)
- Risk score percentage
- Tampered area estimation
- Model confidence bar
- Engine source indicator

✅ **Enterprise Features**
- Forensic mask overlay
- **NEW:** Download PNG for compliance
- Side-by-side visualization
- API status indicator

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3 (glassmorphism), vanilla JS |
| Backend | FastAPI, uvicorn |
| ML Model | PyTorch, U-Net + EfficientNet-B3, 5-fold ensemble |
| Preprocessing | OpenCV, Albumentations |
| Postprocessing | Morphological ops, component analysis, RLE encoding |
| Deployment | Render (frontend + API) + Cloudflare DNS |

---

## 🚀 Deploy to Production (Render + Cloudflare)

### 1) Push code to GitHub
Render deploys from GitHub/GitLab. Commit your current project first.

### 2) Create services from `render.yaml`
1. Open Render Dashboard → **New** → **Blueprint**
2. Select your repository root (`d:/image_forgery_detection`)
3. Render will auto-create:
    - `bioforgenet-api` (Python web service)
    - `bioforgenet-web` (static site)

### 3) Get Render default URLs
After first deploy, note:
- API URL: `https://bioforgenet-api.onrender.com`
- Web URL: `https://bioforgenet-web.onrender.com`

### 4) Configure Cloudflare DNS for `bioforgenet.live`
In Cloudflare DNS:
- `@` → CNAME → `bioforgenet-web.onrender.com` (Proxy ON)
- `www` → CNAME → `bioforgenet-web.onrender.com` (Proxy ON)
- `api` → CNAME → `bioforgenet-api.onrender.com` (Proxy ON)

### 5) Add custom domains in Render
- Static service custom domains:
   - `bioforgenet.live`
   - `www.bioforgenet.live`
- API service custom domain:
   - `api.bioforgenet.live`

### 6) SSL and verification
Wait for SSL provisioning in both Cloudflare and Render (usually a few minutes). Then test:
```bash
curl https://api.bioforgenet.live/health
```

### 7) Optional: real-checkpoint mode in production
Set `VERISIGHT_CHECKPOINT_DIR` in Render API service env vars to the path containing `best_fold*.pth` files.

---

## ⚙️ Inference Pipeline

### Real Model Mode (Real Checkpoint)
1. Read image (BGR)
2. Normalize to RGB
3. Resize to 512×512
4. **Test-Time Augmentation:**
   - Original
   - Horizontal flip
   - Vertical flip
   - 90° rotations
5. **5-Fold Ensemble:** Average predictions across folds
6. **Postprocessing:**
   - Adaptive thresholding
   - Morphological operations
   - Connected component filtering
7. Resize to original dimensions
8. Compute confidence + risk score

### Fallback Mode (No Checkpoints)
1. Read image → grayscale
2. Laplacian edge detection
3. Percentile-based thresholding
4. Morphological opening + dilation
5. Component analysis
6. Edge-based heuristic scoring

**Performance:**
- Real: 2-4 seconds (GPU: 0.5-1s)
- Fallback: 50ms (instant)

---

## 📋 Requirements

- Python 3.10+
- pip or conda
- 8GB RAM (CPU) or 6GB VRAM (GPU)
- ~500MB disk for dependencies

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend offline | `uvicorn api.app:app --reload --port 8000` |
| "No best_fold checkpoints found" | Copy checkpoints to `outputs/checkpoints/` |
| Port 8000 in use | `taskkill /PID [PID] /F` or use `--port 8001` |
| Module not found | `pip install -r requirements.txt` |
| Slow inference | Use GPU or reduce input resolution |
| CORS errors | Check API is running on port 8000 |

For more help, see **DEPLOYMENT.md**

---

## 📖 For Recruiters / Portfolio

This project demonstrates:
- ✅ End-to-end ML product development
- ✅ Full-stack architecture (PyTorch + FastAPI + HTML/CSS/JS)
- ✅ Production deployment (Vercel)
- ✅ Enterprise UX design (glassmorphism, dark theme)
- ✅ Advanced ML techniques (ensemble, TTA, postprocessing)
- ✅ Robust fallback strategies
- ✅ API design best practices (CORS, dual-mode operation)

**Live Demo:** https://your-project.vercel.app (after deployment)

---

## 📄 License

Built for portfolio showcase. Model based on personal research.

---

**Next Steps:**
1. Copy trained checkpoints → see COMPLETION_CHECKLIST.md
2. Test locally → Run quick start commands above
3. Deploy to Vercel → See DEPLOYMENT.md
4. Share with recruiters → Link to live URL

**Status:** ✅ Complete & Ready for Production
