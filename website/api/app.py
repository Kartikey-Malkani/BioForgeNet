from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import base64
import torch
import io
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="BioForgeNet API", version="1.0.0")


def _get_max_folds() -> int:
    value = os.getenv("VERISIGHT_MAX_FOLDS", "5").strip()
    try:
        parsed = int(value)
    except ValueError:
        return 5
    return max(1, parsed)


def _cors_origins() -> list[str]:
    env_value = os.getenv("CORS_ORIGINS", "").strip()
    if env_value:
        return [origin.strip() for origin in env_value.split(",") if origin.strip()]
    return [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://bioforgenet.live",
        "https://www.bioforgenet.live",
        "https://bioforgenet-web.onrender.com",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mask_to_base64_png(mask_u8: np.ndarray) -> str:
    heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
    heatmap_rgba = cv2.cvtColor(heatmap, cv2.COLOR_BGR2BGRA)
    alpha = (mask_u8.astype(np.float32) * 0.70).astype(np.uint8)
    heatmap_rgba[:, :, 3] = alpha
    ok, png = cv2.imencode('.png', heatmap_rgba)
    if not ok:
        return ''
    return base64.b64encode(png.tobytes()).decode('utf-8')


def _build_src_path() -> Path:
    app_file = Path(__file__).resolve()
    candidates = [
        app_file.parents[1] / "src",  # /app/src (HF Space Docker layout)
        app_file.parents[2] / "src",  # monorepo layout
        Path("/app/src"),
        Path("/src"),
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "config.py").exists():
            return candidate
    return candidates[0]


def _download_checkpoints_from_hf(target_dir: Path) -> bool:
    repo_id = os.getenv("HF_REPO_ID", "Ritambharam/bioforgenet-checkpoints").strip()
    if not repo_id:
        return False

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print(f"HF download skipped: huggingface_hub not available ({exc})")
        return False

    repo_type = os.getenv("HF_REPO_TYPE", "dataset").strip() or "dataset"
    token = os.getenv("HF_TOKEN", "").strip() or None
    allow_patterns_raw = os.getenv("HF_ALLOW_PATTERNS", "best_fold*.pth")
    allow_patterns = [p.strip() for p in allow_patterns_raw.split(",") if p.strip()]

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )
        return bool(list(target_dir.glob("best_fold*.pth")))
    except Exception as exc:
        print(f"HF checkpoint download failed: {exc}")
        return False


SRC_DIR = _build_src_path()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _find_checkpoint_dir() -> Path | None:
    candidates = []

    env_ckpt = os.getenv("VERISIGHT_CHECKPOINT_DIR")
    if env_ckpt:
        candidates.append(Path(env_ckpt))
    else:
        candidates.append(Path("/tmp/checkpoints"))

    candidates.extend(
        [
            Path(__file__).resolve().parents[2] / "outputs" / "checkpoints",
            Path(__file__).resolve().parents[2] / "checkpoints",
            Path("/kaggle/working/checkpoints"),
        ]
    )

    for ckpt_dir in candidates:
        if ckpt_dir.exists() and list(ckpt_dir.glob("best_fold*.pth")):
            return ckpt_dir

    hf_target = candidates[0]
    if _download_checkpoints_from_hf(hf_target):
        return hf_target

    return None


class ModelService:
    def __init__(self):
        self.ready = False
        self.mode = "proxy-fallback"
        self.reason = "Model not initialized yet"
        self.engine = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self):
        if self.ready:
            return

        try:
            from config import Config
            from inference import InferenceEngine
            from augmentations import get_validation_augmentation
            from postprocess import postprocess_mask

            ckpt_dir = _find_checkpoint_dir()
            if ckpt_dir is None:
                self.mode = "proxy-fallback"
                self.reason = "No best_fold checkpoints found"
                self._postprocess_mask = postprocess_mask
                self._val_aug_fn = get_validation_augmentation
                self.ready = True
                return

            config = Config()
            config.CHECKPOINT_DIR = ckpt_dir
            config.THRESHOLD = float(getattr(config, "THRESHOLD", 0.5))
            config.MIN_AREA = int(getattr(config, "MIN_AREA", 100))
            config.USE_MORPHOLOGY = bool(getattr(config, "USE_MORPHOLOGY", False))
            config.MORPH_KERNEL_SIZE = int(getattr(config, "MORPH_KERNEL_SIZE", 3))

            fold_paths = sorted(ckpt_dir.glob("best_fold*.pth"))
            if not fold_paths:
                self.mode = "proxy-fallback"
                self.reason = f"Checkpoint dir exists but empty: {ckpt_dir}"
                self._postprocess_mask = postprocess_mask
                self._val_aug_fn = get_validation_augmentation
                self.ready = True
                return

            max_folds = _get_max_folds()
            fold_paths = fold_paths[:max_folds]

            engine = InferenceEngine(config=config, device=self.device)
            engine.load_models(fold_paths)

            self.engine = engine
            self.config = config
            self._postprocess_mask = postprocess_mask
            self._val_aug_fn = get_validation_augmentation
            self.mode = "real-checkpoint"
            self.reason = f"Loaded {len(fold_paths)} fold checkpoints from {ckpt_dir} (max={max_folds})"
            self.ready = True

        except Exception as exc:
            self.mode = "proxy-fallback"
            self.reason = f"Initialization error: {exc}"
            self.ready = True

    def infer_with_mask(self, image_bgr: np.ndarray) -> tuple[dict, np.ndarray]:
        if not self.ready:
            self.initialize()

        if self.mode != "real-checkpoint" or self.engine is None:
            return run_proxy_inference_with_mask(image_bgr)

        # Real model path
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        transform = self._val_aug_fn(image_size=self.config.IMAGE_SIZE)
        transformed = transform(image=image_rgb)
        image_tensor = transformed["image"]

        prob_mask = self.engine.predict_single(image_tensor, use_tta=True)
        bin_mask = self._postprocess_mask(
            prob_mask,
            threshold=self.config.THRESHOLD,
            min_area=self.config.MIN_AREA,
            use_morphology=self.config.USE_MORPHOLOGY,
            morph_kernel_size=self.config.MORPH_KERNEL_SIZE,
        )

        resized_mask = cv2.resize(
            (bin_mask * 255).astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        tamper_area_pct = float((resized_mask > 0).mean() * 100.0)
        risk_score = float(np.clip(tamper_area_pct / 20.0, 0.01, 0.99))
        prediction = "forged" if tamper_area_pct > 0 else "authentic"
        confidence = float(np.clip(0.72 + abs(risk_score - 0.5), 0.72, 0.98))
        mask_png_base64 = _mask_to_base64_png(resized_mask)

        result = {
            "prediction": prediction,
            "risk_score": risk_score,
            "tamper_area_pct": tamper_area_pct,
            "confidence": confidence,
            "mask_png_base64": mask_png_base64,
            "engine": "real-checkpoint-v1",
        }
        return result, resized_mask

    def infer(self, image_bgr: np.ndarray) -> dict:
        result, _ = self.infer_with_mask(image_bgr)
        return result


MODEL_SERVICE = ModelService()


def run_proxy_inference_with_mask(image_bgr: np.ndarray) -> tuple[dict, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    lap = cv2.Laplacian(blur, cv2.CV_32F)
    lap = np.abs(lap)
    lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    threshold = np.percentile(lap, 88)
    mask = (lap > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    tamper_area_pct = float(mask.mean() * 100)
    risk_score = float(np.clip(tamper_area_pct / 18.0, 0.01, 0.98))
    prediction = 'forged' if risk_score >= 0.5 else 'authentic'
    confidence = float(np.clip(0.62 + abs(risk_score - 0.5), 0.62, 0.96))

    mask_u8 = (mask * 255).astype(np.uint8)
    mask_png_base64 = _mask_to_base64_png(mask_u8)

    result = {
        'prediction': prediction,
        'risk_score': risk_score,
        'tamper_area_pct': tamper_area_pct,
        'confidence': confidence,
        'mask_png_base64': mask_png_base64,
        'engine': 'proxy-forensic-v1'
    }
    return result, mask_u8


def run_proxy_inference(image_bgr: np.ndarray) -> dict:
    result, _ = run_proxy_inference_with_mask(image_bgr)
    return result


def _decode_upload_to_bgr(content: bytes) -> np.ndarray | None:
    image_np = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(image_np, cv2.IMREAD_COLOR)


@app.get('/')
def root():
    MODEL_SERVICE.initialize()
    return {
        'name': 'BioForgeNet API',
        'status': 'live',
        'mode': MODEL_SERVICE.mode,
        'endpoints': ['/health', '/model-status', '/api/book-demo', '/analyze', '/analyze-mask', '/docs'],
    }


@app.get('/health')
def health():
    MODEL_SERVICE.initialize()
    return {
        'status': 'ok',
        'mode': MODEL_SERVICE.mode,
        'reason': MODEL_SERVICE.reason,
    }


@app.get('/model-status')
def model_status():
    MODEL_SERVICE.initialize()
    return {
        'mode': MODEL_SERVICE.mode,
        'reason': MODEL_SERVICE.reason,
        'device': str(MODEL_SERVICE.device),
        'src_dir': str(SRC_DIR),
    }


# ======================== Demo Request Endpoint ========================

class BookDemoRequest(BaseModel):
    company_name: str
    first_name: str
    last_name: str
    email: str
    phone: str
    industry: str
    company_size: str
    use_case: Optional[str] = None
    message: Optional[str] = None


@app.post('/api/book-demo')
async def book_demo(request: BookDemoRequest):
    """
    Store demo request and send automatic notification email to admin.
    """
    from api.models import DemoRequest, SessionLocal
    from api.email_utils import send_demo_request_email
    
    db = SessionLocal()
    try:
        # Store in database
        demo_req = DemoRequest(
            company_name=request.company_name,
            first_name=request.first_name,
            last_name=request.last_name,
            email=request.email,
            phone=request.phone,
            industry=request.industry,
            company_size=request.company_size,
            use_case=request.use_case,
            message=request.message,
        )
        db.add(demo_req)
        db.commit()
        db.refresh(demo_req)
        
        # Send email notification
        email_sent = send_demo_request_email(
            company_name=request.company_name,
            first_name=request.first_name,
            last_name=request.last_name,
            email=request.email,
            phone=request.phone,
            industry=request.industry,
            company_size=request.company_size,
            use_case=request.use_case,
            message=request.message,
        )
        
        # Update email_sent status
        demo_req.email_sent = email_sent
        db.commit()
        
        return {
            "success": True,
            "message": "Demo request received. We'll get back to you soon!",
            "request_id": demo_req.id,
            "email_sent": email_sent,
        }
    except Exception as e:
        db.rollback()
        print(f"❌ Error processing demo request: {e}")
        return {
            "success": False,
            "message": "Failed to process request. Please try again or email us directly.",
            "error": str(e),
        }
    finally:
        db.close()


# ======================== Image Analysis Endpoints ========================
@app.post('/analyze')
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    image_bgr = _decode_upload_to_bgr(content)

    if image_bgr is None:
        return {
            'prediction': 'authentic',
            'risk_score': 0.0,
            'tamper_area_pct': 0.0,
            'confidence': 0.0,
            'mask_png_base64': '',
            'engine': 'invalid-image'
        }

    return MODEL_SERVICE.infer(image_bgr)


@app.post('/analyze-mask')
async def analyze_mask(file: UploadFile = File(...)):
    """
    Returns raw binary forgery mask as PNG (white=tampered, black=clean).
    Useful for downloadable forensic assets in reports.
    """
    content = await file.read()
    image_bgr = _decode_upload_to_bgr(content)

    if image_bgr is None:
        empty = np.zeros((64, 64), dtype=np.uint8)
        ok, png_bytes = cv2.imencode('.png', empty)
        if not ok:
            return StreamingResponse(io.BytesIO(b''), media_type='image/png')
        return StreamingResponse(io.BytesIO(png_bytes.tobytes()), media_type='image/png')

    result, mask_u8 = MODEL_SERVICE.infer_with_mask(image_bgr)
    ok, png_bytes = cv2.imencode('.png', mask_u8)

    headers = {
        'X-Prediction': str(result.get('prediction', 'unknown')),
        'X-Risk-Score': str(result.get('risk_score', 0.0)),
        'X-Engine': str(result.get('engine', 'unknown')),
    }

    if not ok:
        return StreamingResponse(io.BytesIO(b''), media_type='image/png', headers=headers)

    return StreamingResponse(
        io.BytesIO(png_bytes.tobytes()),
        media_type='image/png',
        headers=headers,
    )
