"""Configuration constants and data classes for compression trainers."""

from dataclasses import dataclass


# Constants from source code analysis
IMAGE_TOKEN_ID = 128815

# Vision mode configurations
VISION_MODES = {
    'tiny': {'base_size': 512, 'image_size': 512, 'tokens': 73, 'crop_mode': False},
    'small': {'base_size': 640, 'image_size': 640, 'tokens': 111, 'crop_mode': False},
    'base': {'base_size': 1024, 'image_size': 1024, 'tokens': 273, 'crop_mode': False},
    'large': {'base_size': 1280, 'image_size': 1280, 'tokens': 421, 'crop_mode': False},
}

# Vision prompt presets (from DeepSeek-OCR README)
# All presets include leading '\n' and special tokens where appropriate
VISION_PROMPT_PRESETS = {
    'free_ocr': '\nFree OCR.',
    'ocr_grounding': '\n<|grounding|>OCR this image.',
    'markdown': '\n<|grounding|>Convert the document to markdown.',
    'parse_figure': '\nParse the figure.',
    'describe': '\nDescribe this image in detail.',
}

# Legacy constants (kept for backwards compatibility)
GUNDAM_PRESET = {'base_size': 1024, 'image_size': 640, 'crop_mode': True, 'tokens': 273}
VISION_TOKEN_COUNT = 273  # For Gundam preset without cropping

# Vision mode configurations for DeepSeek-OCR-2 (DeepEncoder V2).
#
# DeepEncoder V2 emits a fixed causal-flow query grid whose size depends on the
# input resolution (image_size / 64 squared):
#   512px -> 8x8  = 64 queries
#   768px -> 12x12 = 144 queries
#   1024px -> 16x16 = 256 queries
# The OCR-2 model additionally appends one learnable `view_seperator` token, so
# the number of image placeholder tokens the decoder sees is n_query**2 + 1.
#
# The "tiny" entry uses a query_512 embedding (added in the vendored OCR-2 code
# and initialised by interpolation from query_1024) so the experiment runs on the
# already-rendered 512x512 "tiny" images.
VISION_MODES_OCR2 = {
    'tiny': {'base_size': 512, 'image_size': 512, 'tokens': 65, 'crop_mode': False},
    'small': {'base_size': 768, 'image_size': 768, 'tokens': 145, 'crop_mode': False},
    'base': {'base_size': 1024, 'image_size': 1024, 'tokens': 257, 'crop_mode': False},
}

# Encoder component parameter prefixes (for freeze/unfreeze and param grouping)
# These must match the actual model architecture (DeepseekOCRForCausalLM -> DeepseekOCRModel)
ENCODER_COMPONENTS = ['model.sam_model', 'model.vision_model', 'model.projector']

# OCR-2 (DeepEncoder V2) encoder components: SAM + Qwen2 causal-flow encoder
# + linear projector + the learnable view separator.
ENCODER_COMPONENTS_OCR2 = [
    'model.sam_model',
    'model.qwen2_model',
    'model.projector',
    'model.view_seperator',
]


def get_vision_modes(ocr_version: str = 'ocr1'):
    """Return the vision-mode mapping for the requested OCR model version."""
    return VISION_MODES_OCR2 if ocr_version == 'ocr2' else VISION_MODES


def get_encoder_components(ocr_version: str = 'ocr1'):
    """Return the encoder component prefixes for the requested OCR model version."""
    return ENCODER_COMPONENTS_OCR2 if ocr_version == 'ocr2' else ENCODER_COMPONENTS


@dataclass
class MemoryProfile:
    """Memory usage profile for a batch"""
    batch_size: int
    before_forward_mb: float
    after_forward_mb: float
    after_backward_mb: float
    peak_mb: float
