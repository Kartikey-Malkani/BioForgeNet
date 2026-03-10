# Quick Command Reference

## 🚀 1. Start Local Development (Copy & Paste)

### Terminal 1: Frontend
```bash
cd d:\image_forgery_detection\website
python -m http.server 5500
```
→ Open http://localhost:5500

### Terminal 2: Backend API
```bash
cd d:\image_forgery_detection\website
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
```
→ API at http://localhost:8000

---

## ✅ 2. Verify Everything Works

```bash
# In any terminal:
curl http://localhost:8000/health
curl http://localhost:8000/model-status
```

Expected response:
```json
{
  "status": "ok",
  "mode": "proxy-fallback",
  "reason": "No best_fold checkpoints found"
}
```

---

## 🔄 3. Activate Real Model (Copy Checkpoints)

```bash
# 1. Create checkpoint folder
mkdir d:\image_forgery_detection\outputs\checkpoints

# 2. Copy your best_fold*.pth files to this folder
# (From your training output)

# 3. Restart API (stop with Ctrl+C, then run):
cd d:\image_forgery_detection\website
uvicorn api.app:app --reload --port 8000

# 4. Verify it detects real model:
curl http://localhost:8000/model-status
# Should show: "mode": "real-checkpoint"
```

---

## 🚀 4. Deploy to Vercel

```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Deploy
cd d:\image_forgery_detection\website
vercel

# 3. Follow prompts:
#    - Set project name (e.g., verisight-biosecure)
#    - Confirm framework
#    - Deploy!

# Result: https://your-project.vercel.app
```

---

## 📝 5. Common Tasks

### Check API Status
```bash
Invoke-WebRequest -UseBasicParsing http://localhost:8000/health | Select-Object -ExpandProperty Content
```

### Test Real Model (after adding checkpoints)
```bash
curl -X POST http://localhost:8000/analyze -F "file=@test_image.jpg"
```

### Download Forensic Mask
```bash
curl -X POST http://localhost:8000/analyze-mask -F "file=@test_image.jpg" --output mask.png
```

### Kill Process on Port 8000
```bash
netstat -ano | findstr :8000
taskkill /PID [PID_NUMBER] /F
```

### Use Custom Checkpoint Path
```bash
set VERISIGHT_CHECKPOINT_DIR=D:\custom\checkpoint\path
uvicorn api.app:app --reload --port 8000
```

---

## 📚 Documentation Map

| Task | Go To |
|------|-------|
| **Quick start** | README.md |
| **What to do next** | COMPLETION_CHECKLIST.md |
| **Full architecture** | DEPLOYMENT.md |
| **Project overview** | PROJECT_SUMMARY.md |
| **This file** | QUICK_COMMANDS.md |

---

## 🎯 Maximum Impact Checklist

- [ ] Verify local dev (both terminals running, http://localhost:5500 loads)
- [ ] Copy checkpoints (create outputs/checkpoints/, add best_fold*.pth files)
- [ ] Verify real model (/model-status shows "real-checkpoint")
- [ ] Test download button (analyze image, click download, verify PNG)
- [ ] Deploy to Vercel (`vercel` command)
- [ ] Get live URL (https://your-project.vercel.app)
- [ ] Share with recruiters (LinkedIn, portfolio, job apps)

---

## ⏱️ Time Estimates

| Task | Duration |
|------|----------|
| Start local dev | 2 min |
| Copy checkpoints | 5 min |
| Verify real model | 2 min |
| Test download | 3 min |
| Deploy to Vercel | 5 min |
| **Total** | **~15 min** |

---

## 🆘 Troubleshooting

| Issue | Fix |
|-------|-----|
| "Backend offline" | Start API in Terminal 2 |
| Port 8000 used | `taskkill /PID [number] /F` (see above) |
| Module not found | `pip install -r requirements.txt` |
| "proxy-fallback mode" | Copy checkpoints to outputs/checkpoints/ |
| Deployment fails | Check `vercel logs` |

---

## 🎁 File Structure (Key Files)

```
website/
├── index.html          ← Landing page (open this in browser)
├── script.js           ← Frontend logic (includes download button)
├── styles.css          ← Dark theme styling
├── api/app.py          ← Backend API (FastAPI)
├── requirements.txt    ← Dependencies (pip install)
├── vercel.json         ← Deployment config
├── README.md           ← Start here
├── DEPLOYMENT.md       ← Full guide
├── COMPLETION_CHECKLIST.md ← Next steps
├── PROJECT_SUMMARY.md  ← Overview
└── QUICK_COMMANDS.md   ← This file
```

---

**Ready to deploy?**  
→ Copy those checkpoints and follow the "Activate Real Model" section above! 🚀
