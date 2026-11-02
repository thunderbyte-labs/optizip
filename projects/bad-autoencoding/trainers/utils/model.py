"""Model loading utilities."""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ATTENTION_IMPLEMENTATIONS = ("auto", "flash_attention_2", "eager")


def _flash_attn_available() -> bool:
    """Return True when a usable FlashAttention-2 build is installed."""
    try:
        from transformers.utils import is_flash_attn_2_available
    except ImportError:
        return False
    return is_flash_attn_2_available()


def resolve_attn_implementation(requested: str = "auto") -> str:
    """Resolve the attention implementation used by the vendored model.

    ``auto`` selects FlashAttention-2 when available and falls back to eager
    attention otherwise. The fallback is required on platforms without a
    FlashAttention-2 wheel (for example aarch64 + CUDA 13), where the previous
    hardcoded choice made model loading fail with an ImportError.
    """
    if requested not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            f"attn_implementation must be one of {ATTENTION_IMPLEMENTATIONS}, "
            f"got {requested!r}"
        )
    if requested != "auto":
        return requested
    return "flash_attention_2" if _flash_attn_available() else "eager"


def _init_ocr2_query_512(model):
    """Initialise the new 8x8 causal-flow query embedding of DeepEncoder V2.

    OCR-2 was pretrained with fixed 12x12 (768px) and 16x16 (1024px) query
    grids. The "tiny" experiment needs an 8x8 grid (512px). We initialise it by
    bicubic interpolation of the 16x16 ``query_1024`` embedding, which keeps the
    spatial structure of the pretrained causal-flow queries. The embedding is
    then frozen together with the rest of the vision encoder.
    """
    qwen2 = model.model.qwen2_model
    with torch.no_grad():
        w = qwen2.query_1024.weight.detach().float()          # [256, hidden]
        hidden = w.shape[1]
        w = w.view(16, 16, hidden).permute(2, 0, 1).unsqueeze(0)  # [1, hidden, 16, 16]
        w = F.interpolate(w, size=(8, 8), mode="bicubic", align_corners=False)
        w = w.squeeze(0).permute(1, 2, 0).reshape(64, hidden)     # [64, hidden]
        qwen2.query_512.weight.copy_(w.to(qwen2.query_512.weight.dtype))
    print(
        "[OCR-2] Initialised query_512 by bicubic interpolation of "
        f"query_1024 (16x16 -> 8x8, {hidden}d)"
    )


def load_model_and_tokenizer(
    model_name: Optional[str] = None,
    weights_name: Optional[str] = None,
    device: str = 'cuda',
    use_optimized_model: bool = False,
    use_encoder_checkpointing: bool = False,
    load_vision_encoder: bool = True,
    attn_implementation: str = "auto",
    ocr_version: str = "ocr1"
):
    """
    Load DeepSeek-OCR (OCR-1) or DeepSeek-OCR-2 model and tokenizer

    Args:
        model_name: Path to vendored code and config. Defaults to
                    './vendor/deepseek_ocr' (OCR-1) or './vendor/deepseek_ocr_2' (OCR-2).
        weights_name: HuggingFace model identifier (or cache path) for weights.
                    Defaults to 'deepseek-ai/DeepSeek-OCR' or
                    'deepseek-ai/DeepSeek-OCR-2' depending on ``ocr_version``.
        device: Device to load model to ('cuda', 'cuda:0', 'cpu', etc.)
        use_optimized_model: If True, use training-optimized implementation (OCR-1 only)
        use_encoder_checkpointing: If True, enable gradient checkpointing for vision encoder
        load_vision_encoder: If True, load vision encoder components (SAM, ViT, projector).
                            Set to False for text-only regimes to save ~1.5-2GB GPU memory
                            (OCR-1 only; OCR-2 always constructs its encoder).
        attn_implementation: Attention backend: 'auto' (FlashAttention-2 if installed,
                            eager otherwise), 'flash_attention_2', or 'eager'.
        ocr_version: 'ocr1' (DeepSeek-OCR) or 'ocr2' (DeepSeek-OCR-2 / DeepEncoder V2).

    Returns:
        Tuple of (model, tokenizer)
    """
    if ocr_version not in ("ocr1", "ocr2"):
        raise ValueError(f"ocr_version must be 'ocr1' or 'ocr2', got {ocr_version!r}")

    if model_name is None:
        model_name = './vendor/deepseek_ocr_2' if ocr_version == 'ocr2' else './vendor/deepseek_ocr'
    if weights_name is None:
        weights_name = 'deepseek-ai/DeepSeek-OCR-2' if ocr_version == 'ocr2' else 'deepseek-ai/DeepSeek-OCR'

    if ocr_version == 'ocr2' and use_optimized_model:
        raise ValueError("DeepSeek-OCR-2 has no training-optimized implementation; drop --use_optimized_model")

    vendor_path = Path(model_name).expanduser().resolve()
    if not vendor_path.exists():
        raise FileNotFoundError(f"Vendored model path not found: {vendor_path}")

    # Add vendor parent directory to sys.path to enable package imports
    vendor_parent = vendor_path.parent.resolve()
    vendor_parent_str = str(vendor_parent)
    if vendor_parent_str not in sys.path:
        sys.path.insert(0, vendor_parent_str)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(vendor_path),
        trust_remote_code=True
    )

    # Validate flag combination
    if ocr_version == 'ocr1' and use_encoder_checkpointing and not use_optimized_model:
        raise ValueError(
            "Error: --use_encoder_checkpointing requires --use_optimized_model\n"
            "The original implementation does not support gradient checkpointing.\n"
            "Please add --use_optimized_model flag or remove --use_encoder_checkpointing."
        )
    if ocr_version == 'ocr2' and use_encoder_checkpointing:
        raise ValueError("DeepSeek-OCR-2 currently does not support --use_encoder_checkpointing")

    # Import appropriate implementation based on flag
    if ocr_version == 'ocr2':
        print("📦 Using DeepSeek-OCR-2 (DeepEncoder V2) implementation")
        from deepseek_ocr_2.modeling_deepseekocr2 import (
            DeepseekOCR2Config as DeepseekOCRConfig,
            DeepseekOCR2ForCausalLM as DeepseekOCRForCausalLM,
        )
    elif use_optimized_model:
        print("🚀 Using OPTIMIZED DeepSeek-OCR implementation")
        from deepseek_ocr.modeling_deepseekocr_optimized import (
            DeepseekOCRConfigOptimized as DeepseekOCRConfig,
            DeepseekOCRForCausalLMOptimized as DeepseekOCRForCausalLM
        )
        if use_encoder_checkpointing:
            print("   ✓ Gradient checkpointing enabled for vision encoder")
    else:
        print("📦 Using ORIGINAL DeepSeek-OCR implementation")
        from deepseek_ocr.modeling_deepseekocr import DeepseekOCRConfig, DeepseekOCRForCausalLM

    print("Loading config...")
    config = DeepseekOCRConfig.from_pretrained(str(vendor_path))
    resolved_attn_implementation = resolve_attn_implementation(attn_implementation)
    print(f"Attention implementation: {resolved_attn_implementation}")
    config._attn_implementation = resolved_attn_implementation

    print("Loading model...")
    if ocr_version == 'ocr1' and not load_vision_encoder:
        print("   ⚠️  Vision encoder disabled - saving ~1.5-2GB GPU memory")

    # Pass checkpointing flag to optimized version if applicable
    if ocr_version == 'ocr2':
        model = DeepseekOCRForCausalLM.from_pretrained(
            weights_name,
            config=config,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=False,
        )
        _init_ocr2_query_512(model)
    elif use_optimized_model:
        model = DeepseekOCRForCausalLM.from_pretrained(
            weights_name,
            config=config,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=False,
            use_encoder_checkpointing=use_encoder_checkpointing,
            load_vision_encoder=load_vision_encoder
        )
    else:
        model = DeepseekOCRForCausalLM.from_pretrained(
            weights_name,
            config=config,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=False,
            load_vision_encoder=load_vision_encoder
        )

    device_obj = torch.device(device)
    print(f"Ensuring bfloat16 dtype and moving to {device}...")
    model = model.eval().to(torch.bfloat16)
    if os.environ.get("CLONE_MODEL_BEFORE_DEVICE", "0") == "1":
        # Workaround for a ROCm/PyTorch path where every host-to-device copy
        # from a safetensors mmap tensor costs a fixed ~2 s. Copying each
        # storage into anonymous CPU memory first (clone) makes the H2D copy
        # fast. Cache by data_ptr so tied parameters keep sharing storage.
        cache = {}

        def _clone_to_device(tensor):
            key = (tensor.data_ptr(), tuple(tensor.shape),
                   tuple(tensor.stride()), tensor.dtype)
            if key not in cache:
                cache[key] = tensor.clone().to(device_obj)
            return cache[key]

        model._apply(_clone_to_device)
        model = model.to(torch.bfloat16)
    else:
        model = model.to(device_obj).to(torch.bfloat16)

    return model, tokenizer
