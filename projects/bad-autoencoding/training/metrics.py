"""
Text similarity metrics for qualitative evaluation.

Vendored from Fox benchmark (https://github.com/ucaslcl/Fox)
Original file: eval_tools/eval_ocr_test.py
"""

import re

import jieba
import nltk
import torch
import torch.nn.functional as F
from nltk.metrics import f_measure, precision, recall
from nltk.translate import meteor_score


def per_sample_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute the mean negative log-likelihood of each sample.

    This mirrors the shifted cross-entropy used by the model, but keeps the
    values per sample so validation uncertainty and paired tests can be
    computed. Positions with ``ignore_index`` do not contribute.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(shift_labels.shape)
    token_counts = (shift_labels != ignore_index).sum(dim=1).clamp_min(1)
    return token_losses.sum(dim=1) / token_counts


def contain_chinese_string(text):
    """Check if text contains Chinese characters."""
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))


def cal_per_metrics(predict_root_, pred, gt):
    """
    Calculate text similarity metrics between prediction and ground truth.

    Args:
        predict_root_: Unused (kept for API compatibility)
        pred: Predicted/generated text
        gt: Ground truth text

    Returns:
        Dict with keys: bleu, meteor, f_measure, precision, recall, edit_dist
    """
    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))

    return metrics
