#!/usr/bin/env python3
"""Late-fusion of independently trained bscan-only and slo-only
single-modality classifiers (run_cls_tuning_UF.py) on the UF cohort.

For each (dataset, model, probe-tag) combination, loads the bscan-trained
and slo-trained runs' `predictions.npz` (see `mutils.classification_uf
.evaluate_uf`'s `save_predictions` path), averages their per-sample
post-softmax class probabilities (`prediction_list`) -- i.e. late fusion /
soft-voting ensembling, not the joint-attention early fusion of
`run_cls_tuning_UF_multimodaliy.py` -- and recomputes the OphFoundation
metric set (`mutils.metrics_uf.compute_metrics_uf`) on the fused
predictions.

Mirrors `summarize_bscan_results.py`'s output-dir discovery convention:
    {base_output_dir}/{version}/{seed}/{data_set}/{model_name}_{probe_tag}_w_*/
Each candidate's `args.json` is read to tell the bscan run apart from the
slo run (the checksum in the directory name is opaque).

`predictions.npz` isn't saved by a normal training run of
run_cls_tuning_UF.py; see run_uf_dual_modality_latefusion.sh, which
(re)dumps it from each already-trained checkpoint before calling this
script.
"""
import argparse
import glob
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from mutils.metrics_uf import compute_metrics_uf


EVAL_CSV_COLUMNS = [
    'epoch', 'loss', 'accuracy', 'f1', 'auroc', 'ap', 'precision', 'recall',
    'kappa', 'mcc',
]
EVAL_CSV_HEADER = [
    'Epoch', 'Loss', 'Accuracy', 'F1-score', 'AUROC', 'AP', 'Precision',
    'Recall', 'Kappa', 'MCC',
]
SUMMARY_METRICS = [
    'accuracy', 'f1', 'auroc', 'ap', 'precision', 'recall', 'kappa', 'mcc',
]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--base_output_dir', required=True,
        help='Base dir of the single-modality runs (run_cls_tuning_UF.py'
            ' --base_output_dir).',
    )
    parser.add_argument(
        '--out_base_output_dir', required=True,
        help='Base dir to write late-fusion results to.',
    )
    parser.add_argument('--version', default='v1')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument(
        '--model_names', nargs='+', default=['mirage-base', 'mirage-large'],
        help='fm_config_factory keys used in output dir names.'
            ' (default: %(default)s)',
    )
    parser.add_argument(
        '--probe_tags', nargs='+', default=['linear', 'finetune'],
        help='Output dir tag for each probing mode ("linear" for'
            ' --linear_probing, "finetune" for full fine-tune).'
            ' (default: %(default)s)',
    )
    parser.add_argument(
        '--save_predictions', action='store_true',
        help='Also save the fused predictions.npz next to each result.'
            ' (default: %(default)s)',
    )
    return parser.parse_args()


def find_modality_dir(base_output_dir, version, seed, dataset, model_name, probe_tag, modality):
    pattern = (
        f'{base_output_dir}/{version}/{seed}/{dataset}/'
        f'{model_name}_{probe_tag}_w_*'
    )
    for candidate in sorted(glob.glob(pattern)):
        args_path = Path(candidate) / 'args.json'
        if not args_path.exists():
            continue
        with open(args_path) as f:
            run_args = json.load(f)
        if run_args.get('uf_modality') == modality:
            return Path(candidate)
    return None


def load_probs(run_dir):
    npz_path = run_dir / 'predictions.npz'
    if not npz_path.exists():
        raise FileNotFoundError(
            f'{npz_path} not found. Regenerate it by rerunning'
            ' run_cls_tuning_UF.py with the exact training args plus'
            ' --eval --save_predictions (see'
            ' run_uf_dual_modality_latefusion.sh).'
        )
    data = np.load(npz_path)
    return data['true_label_decode_list'], data['prediction_list']


def late_fuse(bscan_dir, slo_dir):
    y_true_b, probs_b = load_probs(bscan_dir)
    y_true_s, probs_s = load_probs(slo_dir)
    if not np.array_equal(y_true_b, y_true_s):
        raise ValueError(
            f'Test-set label mismatch between {bscan_dir} and {slo_dir}'
            ' -- are these really the same task/split?'
        )
    y_true = y_true_b
    fused_probs = (probs_b + probs_s) / 2.0
    fused_preds = fused_probs.argmax(axis=1)
    num_classes = fused_probs.shape[1]

    metrics = compute_metrics_uf(y_true, fused_preds, fused_probs, num_classes)
    eps = 1e-12
    loss = float(-np.mean(np.log(fused_probs[np.arange(len(y_true)), y_true] + eps)))
    stats = OrderedDict({'epoch': 'LateFusion', 'loss': loss, **metrics})
    return y_true, fused_preds, fused_probs, stats


def main():
    args = get_args()

    summary_rows = []
    for dataset in args.datasets:
        for model_name in args.model_names:
            for probe_tag in args.probe_tags:
                bscan_dir = find_modality_dir(
                    args.base_output_dir, args.version, args.seed, dataset,
                    model_name, probe_tag, 'bscan',
                )
                slo_dir = find_modality_dir(
                    args.base_output_dir, args.version, args.seed, dataset,
                    model_name, probe_tag, 'slo',
                )
                if bscan_dir is None or slo_dir is None:
                    print(
                        f'  [missing] {dataset}/{model_name}/{probe_tag}:'
                        f' bscan={bscan_dir} slo={slo_dir}',
                        file=sys.stderr,
                    )
                    continue

                try:
                    y_true, fused_preds, fused_probs, stats = late_fuse(bscan_dir, slo_dir)
                except (FileNotFoundError, ValueError) as e:
                    print(f'  [error] {dataset}/{model_name}/{probe_tag}: {e}', file=sys.stderr)
                    continue

                out_dir = (
                    Path(args.out_base_output_dir) / args.version / str(args.seed)
                    / dataset / f'{model_name}_{probe_tag}_latefusion'
                )
                out_dir.mkdir(parents=True, exist_ok=True)

                pd.DataFrame(
                    data=[[stats[k] for k in EVAL_CSV_COLUMNS]],
                    columns=EVAL_CSV_HEADER,
                ).to_csv(out_dir / 'test_eval.csv', index=False)

                if args.save_predictions:
                    np.savez_compressed(
                        out_dir / 'predictions.npz',
                        true_label_decode_list=y_true,
                        prediction_decode_list=fused_preds,
                        prediction_list=fused_probs,
                    )

                print(
                    f'{dataset}/{model_name}/{probe_tag}:'
                    f' Acc={stats["accuracy"]:.2f} AUROC={stats["auroc"]:.2f}'
                    f' F1={stats["f1"]:.2f}'
                )
                print('  confusion_matrix:\n', stats['confusion_matrix'])

                summary_rows.append({
                    'dataset': dataset,
                    'model': model_name,
                    'probe': probe_tag,
                    **{m: stats[m] for m in SUMMARY_METRICS},
                })

    if not summary_rows:
        print('No (dataset, model, probe) combination produced a fused result.', file=sys.stderr)
        sys.exit(1)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(args.out_base_output_dir) / args.version / str(args.seed) / 'latefusion_summary.csv'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(summary_df.to_string(index=False))
    print(f'\nWrote summary to {summary_path}')


if __name__ == '__main__':
    main()
