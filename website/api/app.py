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

app = FastAPI(title="BioForgeNet API", version="1.0.0")


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
    return Path(__file__).resolve().parents[2] / "src"


SRC_DIR = _build_src_path()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _find_checkpoint_dir() -> Path | None:
    candidates = []

    env_ckpt = os.getenv("VERISIGHT_CHECKPOINT_DIR")
    if env_ckpt:
        candidates.append(Path(env_ckpt))

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

            engine = InferenceEngine(config=config, device=self.device)
            engine.load_models(fold_paths)

            self.engine = engine
            self.config = config
            self._postprocess_mask = postprocess_mask
            self._val_aug_fn = get_validation_augmentation
            self.mode = "real-checkpoint"
            self.reason = f"Loaded {len(fold_paths)} fold checkpoints from {ckpt_dir}"
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
