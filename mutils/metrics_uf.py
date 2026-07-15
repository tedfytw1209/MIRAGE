import json
from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    average_precision_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


"""Metric computation and wandb-safe value coercion for the UF cohort
scripts (`run_cls_tuning_UF.py`, `run_cls_tuning_UF_multimodaliy.py`),
ported from OphFoundation's `pytorch_image_classification_our_model-v2-UF.py`
(`evaluate_model` / `safe_for_wandb`) so results are directly comparable to
OphFoundation's own numbers. This differs from `mutils.classification`'s
metrics in two ways: averaging is macro (unweighted across classes) rather
than weighted by class support, and every metric is scaled to 0-100 instead
of 0-1. It also adds precision, recall, Cohen's kappa, the confusion
matrix, and the classification report, none of which
`mutils.classification` computes.
"""


def compute_metrics_uf(y_true, y_pred, y_proba, num_classes) -> OrderedDict:
    y_onehot = np.eye(num_classes)[y_true]
    multiclass_present = len(set(y_true)) > 1
    return OrderedDict({
        'accuracy': 100.0 * accuracy_score(y_true, y_pred),
        'f1': 100.0 * f1_score(y_true, y_pred, average='macro', zero_division=0),
        'auroc': 100.0 * roc_auc_score(y_onehot, y_proba, multi_class='ovr', average='macro') if multiclass_present else 0.0,
        'ap': 100.0 * average_precision_score(y_onehot, y_proba, average='macro'),
        'precision': 100.0 * precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': 100.0 * recall_score(y_true, y_pred, average='macro', zero_division=0),
        'kappa': 100.0 * cohen_kappa_score(y_true, y_pred),
        'mcc': 100.0 * matthews_corrcoef(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(num_classes))),
        'classification_report': classification_report(y_true, y_pred, zero_division=0),
    })


def safe_for_wandb(x):
    """Coerce a value into something `wandb.log` can serialize (ported
    verbatim from OphFoundation's util)."""
    if isinstance(x, torch.Tensor):
        try:
            return x.item()
        except Exception:
            return x.detach().cpu().tolist()
    elif isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, (int, float, str, bool)) or x is None:
        return x
    else:
        try:
            json.dumps(x)
            return x
        except Exception:
            return str(x)
