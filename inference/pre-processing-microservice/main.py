#!/usr/bin/env python3
"""
Optical Preprocessing Microservice (pur)
Cible : hera (ROCm 7.2) – 1× RDNA 4 (9070 XT)
DeepEncoder V2 (multi-crop) → OpticalAdapter → prompt_embeds base64
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import io
import time
import uuid
import base64
import logging
from typing import List
from contextlib import contextmanager
from logging import LogRecord

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
from models.optical_adapter import OpticalAdapter

# ==============================================================================
# Configuration – forcée pour hera (1 seul GPU)
# ==============================================================================

ADAPTER_CHECKPOINT = os.getenv(
    "ADAPTER_CHECKPOINT",
    "checkpoints/optical_adapter_phase1/adapter_final.pt"
)

# Sur hera on force explicitement le premier GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # sécurité supplémentaire
DEVICE = torch.device("cuda:0")

# ==============================================================================
# Système de logs (format strict demandé)
# ==============================================================================

class StrictFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str:
        if not hasattr(record, "request_id"):
            record.request_id = "--------"
        if not hasattr(record, "duration_ms"):
            duration_str = ""
        else:
            duration_str = f" | {record.duration_ms:7.1f} ms"

        location = f"{record.filename}:{record.lineno}"
        return (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S.%f')[:-3]} | "
            f"{record.levelname:<5} | "
            f"pid={os.getpid()} | "
            f"req={record.request_id:<8} | "
            f"{location:<20} | "
            f"{record.getMessage()}"
            f"{duration_str}"
        )

def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(StrictFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return logging.getLogger("optical")

logger = setup_logging()

@contextmanager
def timed_stage(request_id: str, stage_name: str, extra_info: str = ""):
    """Ne logue qu'une seule ligne à la fin (DONE + durée)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        msg = f"DONE {stage_name}"
        if extra_info:
            msg += f" | {extra_info}"
        logger.info(
            msg,
            extra={"request_id": request_id, "duration_ms": elapsed_ms}
        )

# ==============================================================================
# Modèles de réponse
# ==============================================================================

class VisualContextResponse(BaseModel):
    request_id: str
    prompt_embeds_b64: str
    num_visual_tokens: int
    dtype: str = "bfloat16"
    timings_ms: dict
    device: str

class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_name: str
    adapter_loaded: bool
    torch_version: str
    hip_version: str | None

# ==============================================================================
# Chargement des modèles
# ==============================================================================

optical_adapter = None

def load_models():
    global optical_adapter

    logger.info(f"Using device {DEVICE} ({torch.cuda.get_device_name(0)})",
                extra={"request_id": "startup"})

    t0 = time.perf_counter()
    try:
        raw = torch.load(
            ADAPTER_CHECKPOINT,
            map_location="cpu",
            weights_only=True
        )

        # Gestion du format de sauvegarde
        if isinstance(raw, dict) and "adapter" in raw:
            state_dict = raw["adapter"]
        else:
            state_dict = raw

        optical_adapter = OpticalAdapter(in_dim=896, out_dim=2048)
        optical_adapter.load_state_dict(state_dict)
        optical_adapter.eval()
        optical_adapter.to(DEVICE)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("OpticalAdapter loaded",
                    extra={"request_id": "startup", "duration_ms": elapsed})

    except Exception as e:
        logger.error(f"Failed to load OpticalAdapter: {e}",
                     extra={"request_id": "startup"})
        raise

# ==============================================================================
# Extraction (placeholders à remplacer par tes vrais scripts)
# ==============================================================================

def extract_visual_tokens(images: List[Image.Image], request_id: str) -> torch.Tensor:
    """
    TODO : Remplace par ton vrai script multicrop DeepEncoder V2.
    Doit retourner un tensor (N, 896)
    """
    with timed_stage(request_id, "DeepEncoder multi-crop"):
        # ------------------------------------------------------------------
        # PLACEHOLDER – à supprimer dès que tu fournis le vrai code
        # ------------------------------------------------------------------
        logger.warning("Using PLACEHOLDER for DeepEncoder – replace me!",
                       extra={"request_id": request_id})
        num_tokens = 512 * len(images)
        return torch.randn(num_tokens, 896, dtype=torch.float32)

def project_with_adapter(visual_tokens: torch.Tensor, request_id: str) -> torch.Tensor:
    with timed_stage(request_id, "OpticalAdapter projection",
                     extra_info=f"tokens={visual_tokens.shape[0]}"):
        with torch.no_grad():
            projected = optical_adapter(visual_tokens.to(DEVICE))
            projected = projected.to(torch.bfloat16)
        return projected.cpu()

def serialize_to_base64(tensor: torch.Tensor, request_id: str) -> str:
    with timed_stage(request_id, "Sérialisation base64"):
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

# ==============================================================================
# FastAPI
# ==============================================================================

app = FastAPI(
    title="Optical Preprocessing Microservice",
    description="hera / ROCm – DeepEncoder + OpticalAdapter → prompt_embeds",
    version="1.1.0-rocm"
)

@app.on_event("startup")
async def startup_event():
    torch.cuda.set_device(0)
    load_models()

@app.get("/health", response_model=HealthResponse)
async def health():
    hip = getattr(torch.version, "hip", None)
    return HealthResponse(
        status="ok",
        device=str(DEVICE),
        gpu_name=torch.cuda.get_device_name(0),
        adapter_loaded=optical_adapter is not None,
        torch_version=torch.__version__,
        hip_version=hip
    )

@app.post("/extract_visual_context", response_model=VisualContextResponse)
async def extract_visual_context(
    images: List[UploadFile] = File(..., description="Une ou plusieurs images (pages)")
):
    request_id = str(uuid.uuid4())[:8]
    total_start = time.perf_counter()
    timings = {}

    logger.info(f"=== NEW REQUEST | {len(images)} image(s) ===",
                extra={"request_id": request_id})

    try:
        # 1. Chargement images
        with timed_stage(request_id, "Chargement images PIL"):
            pil_images = []
            for f in images:
                content = await f.read()
                pil_images.append(Image.open(io.BytesIO(content)).convert("RGB"))

        # 2. DeepEncoder
        visual_tokens = extract_visual_tokens(pil_images, request_id)

        # 3. OpticalAdapter
        projected = project_with_adapter(visual_tokens, request_id)
        num_tokens = projected.shape[0]

        # 4. Sérialisation
        b64_embeds = serialize_to_base64(projected, request_id)

        total_ms = (time.perf_counter() - total_start) * 1000
        timings["total_ms"] = round(total_ms, 1)
        timings["num_visual_tokens"] = num_tokens

        logger.info(
            f"=== END REQUEST | {num_tokens} tokens | total {total_ms:.1f} ms ===",
            extra={"request_id": request_id}
        )

        return VisualContextResponse(
            request_id=request_id,
            prompt_embeds_b64=b64_embeds,
            num_visual_tokens=num_tokens,
            dtype="bfloat16",
            timings_ms=timings,
            device=str(DEVICE)
        )

    except Exception as e:
        logger.exception(f"Processing failed: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=f"[{request_id}] {str(e)}")

# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        workers=1,               # important sur ROCm : un seul worker
        log_level="info"
    )
