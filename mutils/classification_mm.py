from typing import Optional, Iterable, Union
from collections import OrderedDict
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.amp.autocast_mode import autocast
from torchvision.utils import save_image

import mutils.lr_utils as lru
from mutils.metrics_uf import compute_metrics_uf


"""Multi-modal counterpart of `mutils.classification_uf`. The only real
difference is that a batch's `images` is a dict of per-domain tensors (e.g.
`{'bscan': ..., 'slo': ...}`) rather than a single tensor, so each domain
has to be moved to `device` individually and the debug image dump has to be
done per-domain. Uses the same OphFoundation-derived metric set as
`mutils.classification_uf` (see `mutils.metrics_uf.compute_metrics_uf`).
`EarlyStopping` is unaffected by batch shape or metric set and is reused
as-is from `mutils.classification`.
"""


def train_1_epoch_mm(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    args=None,
):
    assert args is not None, "args must be provided"

    model.train(True)
    optimizer.zero_grad()

    losses, true_labels, predictions, probs_all = [], [], [], []
    for i, batch in enumerate(data_loader):
        images, targets = batch[0], batch[-1]
        images = {k: v.to(device, non_blocking=True) for k, v in images.items()}
        targets = targets.to(device, non_blocking=True)

        # we use a per iteration (instead of per epoch) lr scheduler
        if i % args.accum_iter == 0:
            lru.adjust_learning_rate(optimizer, i / len(data_loader) + epoch, args)

        with autocast('cuda', enabled=True):
            if i == 0 and epoch % 10 == 0:
                epoch_str = str(epoch).zfill(3)
                save_dir = Path(args.output_dir, "debug")
                save_dir.mkdir(exist_ok=True)
                print('Saving images for debugging')
                for domain, domain_images in images.items():
                    print(f'  {domain}', domain_images.shape, domain_images.min().item(), domain_images.max().item())
                    save_image(domain_images, save_dir / f"{epoch_str}_train_{domain}.jpg", normalize=True)
                print('  targets', targets.shape, targets.min().item(), targets.max().item())

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            raise ValueError('Loss is infinite or NaN')

        if (i + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        probs = nn.Softmax(dim=1)(outputs)
        _, preds = torch.max(probs, 1)
        predictions.extend(preds.cpu().detach().numpy())
        true_labels.extend(targets.cpu().detach().numpy())
        probs_all.extend(probs.detach().cpu().numpy())

    y_true = np.array(true_labels)
    y_pred = np.array(predictions)
    y_proba = np.array(probs_all)

    avg_loss = np.mean(losses)
    full_metrics = compute_metrics_uf(y_true, y_pred, y_proba, args.num_classes)
    print("Train confusion_matrix:\n", full_metrics['confusion_matrix'])

    if epoch % 5 == 0:
        print(
            "[Train] Epoch {} - Loss: {:.4f}, Acc: {:.2f}, F1: {:.2f}, AUROC: {:.2f}, MCC: {:.2f}".format(
                epoch, avg_loss, full_metrics['accuracy'], full_metrics['f1'],
                full_metrics['auroc'], full_metrics['mcc'],
            )
        )

    # Lighter set for the per-epoch CSV/wandb log (mirrors OphFoundation's
    #   train_stats, which skips precision/recall/kappa/confusion matrix).
    return OrderedDict({
        'epoch': epoch,
        'loss': avg_loss,
        'accuracy': full_metrics['accuracy'],
        'f1': full_metrics['f1'],
        'auroc': full_metrics['auroc'],
        'ap': full_metrics['ap'],
        'mcc': full_metrics['mcc'],
    })


@torch.no_grad()
def evaluate_mm(
    model: torch.nn.Module,
    data_loader: Iterable,
    epoch: Union[int, str],
    device: torch.device,
    num_class: int,
    mode: str,
    save_path: Optional[Union[str, Path]] = None,
    args=None,
    save_predictions: bool = False,
) -> Optional[dict]:
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    predictions, true_labels, probs_all = [], [], []

    model.eval()

    for bi, batch in enumerate(data_loader):
        images = batch[0]
        targets = batch[-1]
        images = {k: v.to(device, non_blocking=True) for k, v in images.items()}
        targets = targets.to(device, non_blocking=True)

        with autocast('cuda', enabled=True):
            if (
                isinstance(epoch, int)
                and bi == 0
                and epoch % 10 == 0
                and args is not None
            ):
                epoch_str = str(epoch).zfill(3)
                save_dir = Path(args.output_dir, "debug")
                save_dir.mkdir(exist_ok=True)
                print('Saving images for debugging')
                for domain, domain_images in images.items():
                    print(f'  {domain}', domain_images.shape, domain_images.min().item(), domain_images.max().item())
                    save_image(domain_images, save_dir / f"{epoch_str}_{mode}_{bi}_{domain}.jpg", normalize=True)
                print('  targets', targets.shape, targets.min().item(), targets.max().item())

            output = model(images)
            loss = criterion(output, targets)
            losses.append(loss.item())

            probs = nn.Softmax(dim=1)(output)
            _, preds = torch.max(probs, 1)

        predictions.extend(preds.cpu().detach().numpy())
        true_labels.extend(targets.cpu().detach().numpy())
        probs_all.extend(probs.detach().cpu().numpy())

    y_true = np.array(true_labels)
    y_pred = np.array(predictions)
    y_proba = np.array(probs_all)

    if save_predictions:
        print("Saving predictions")
        print('\ttrue_label', y_true.shape)
        print('\tprediction', y_pred.shape)
        assert save_path is not None, "save_path must be provided"
        save_fn = Path(save_path, "predictions.npz")
        np.savez_compressed(
            save_fn,
            true_label_decode_list=y_true,
            prediction_decode_list=y_pred,
            prediction_list=y_proba,
        )
        return

    avg_loss = np.mean(losses)
    metrics = compute_metrics_uf(y_true, y_pred, y_proba, num_class)

    if type(epoch) != str and epoch % 5 == 0:
        print(
            "[{}] Epoch {} - Loss: {:.4f} Acc: {:.2f} F1: {:.2f} AUROC: {:.2f} AP: {:.2f}"
            " Precision: {:.2f} Recall: {:.2f} Kappa: {:.2f} MCC: {:.2f}".format(
                mode, epoch, avg_loss, metrics['accuracy'], metrics['f1'], metrics['auroc'],
                metrics['ap'], metrics['precision'], metrics['recall'], metrics['kappa'], metrics['mcc'],
            )
        )
        print("confusion_matrix:\n", metrics['confusion_matrix'])
        print("classification_report:\n", metrics['classification_report'])

    return OrderedDict({'epoch': epoch, 'loss': avg_loss, **metrics})
