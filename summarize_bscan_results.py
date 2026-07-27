#!/usr/bin/env python3
"""Aggregate run_cls_tuning_bscan.py's per-fold test_eval.csv files into a
mean +/- std summary across folds 0-9, one row per (dataset, model, probe
mode). Mirrors the output_dir convention in run_cls_tuning_bscan.py:
get_output_dir() -- base_output_dir/version/seed/data_set/[foldN/ if fold
!= 0]/{model_name}_{probe_tag}_w_{checksum}/test_eval.csv -- and
test_eval.csv's column names (Epoch/Loss/Accuracy/F1-score/AUROC/AP/
Precision/Recall/Kappa/MCC).
"""
import argparse
import glob
import sys

import pandas as pd


METRIC_COLUMNS = {
    'Accuracy': 'test_acc',
    'F1-score': 'test_f1',
    'AUROC': 'test_auc',
    'AP': 'test_pr',
    'Precision': 'test_precision',
    'Recall': 'test_recall',
    'Kappa': 'test_kappa',
    'MCC': 'test_mcc',
}


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_output_dir', required=True)
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
        '--folds', nargs='+', type=int, default=list(range(10)),
        help='Fold indices to aggregate over. (default: 0-9)',
    )
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def find_test_eval_csvs(base_output_dir, version, seed, dataset, model_name, probe_tag, folds):
    found = []
    for fold in folds:
        if fold == 0:
            pattern = (
                f'{base_output_dir}/{version}/{seed}/{dataset}/'
                f'{model_name}_{probe_tag}_w_*/test_eval.csv'
            )
        else:
            pattern = (
                f'{base_output_dir}/{version}/{seed}/{dataset}/fold{fold}/'
                f'{model_name}_{probe_tag}_w_*/test_eval.csv'
            )
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f'  [missing] {dataset}/{model_name}/{probe_tag} fold {fold}: no match for {pattern}', file=sys.stderr)
            continue
        if len(matches) > 1:
            print(f'  [warn] {dataset}/{model_name}/{probe_tag} fold {fold}: multiple matches, using first of {matches}', file=sys.stderr)
        found.append((fold, matches[0]))
    return found


def main():
    args = get_args()

    summary_rows = []
    for dataset in args.datasets:
        for model_name in args.model_names:
            for probe_tag in args.probe_tags:
                found = find_test_eval_csvs(
                    args.base_output_dir, args.version, args.seed,
                    dataset, model_name, probe_tag, args.folds,
                )
                if not found:
                    continue
                metric_values = {out_col: [] for out_col in METRIC_COLUMNS.values()}
                for _fold, csv_path in found:
                    row = pd.read_csv(csv_path).iloc[0]
                    for col, out_col in METRIC_COLUMNS.items():
                        metric_values[out_col].append(row[col])
                summary_row = {
                    'dataset': dataset,
                    'model': model_name,
                    'probe': probe_tag,
                    'n_folds': len(found),
                }
                for out_col, values in metric_values.items():
                    series = pd.Series(values, dtype=float)
                    summary_row[f'{out_col}_mean'] = series.mean()
                    summary_row[f'{out_col}_std'] = series.std(ddof=0) if len(series) > 1 else float('nan')
                summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out, index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(summary_df.to_string(index=False))
    print(f'\nWrote summary to {args.out}')


if __name__ == '__main__':
    main()
