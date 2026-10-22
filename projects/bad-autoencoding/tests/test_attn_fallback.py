"""Tests for attention backend selection."""

import pytest

from trainers.utils import model as model_utils


def test_explicit_attention_implementations_are_returned_unchanged():
    assert model_utils.resolve_attn_implementation("eager") == "eager"
    assert model_utils.resolve_attn_implementation("flash_attention_2") == "flash_attention_2"


def test_auto_prefers_flash_attention_when_available(monkeypatch):
    monkeypatch.setattr(model_utils, "_flash_attn_available", lambda: True)
    assert model_utils.resolve_attn_implementation("auto") == "flash_attention_2"


def test_auto_falls_back_to_eager_without_flash_attention(monkeypatch):
    monkeypatch.setattr(model_utils, "_flash_attn_available", lambda: False)
    assert model_utils.resolve_attn_implementation("auto") == "eager"


def test_unknown_attention_implementation_is_rejected():
    with pytest.raises(ValueError, match="attn_implementation must be one of"):
        model_utils.resolve_attn_implementation("sdpa")
