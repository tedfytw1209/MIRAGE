# Public fundus/bscan classification scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new MIRAGE classification-tuning entry points —
`run_cls_tuning_fundus.py` (public color fundus datasets) and
`run_cls_tuning_bscan.py` (public OCT B-scan datasets) — plus matching
shell wrappers and a docs update.

**Architecture:** Both new Python scripts are structural copies of the
existing `run_cls_tuning.py` (pre-split `train/val/test/Class_x/`
`ImageFolder` datasets, `EarlyStopping`-driven training loop, weighted 0-1
metrics, plain CSV outputs — no wandb). Each sets `args.uf_modality` before
building `model_config`, to pick which MIRAGE input domain
(`mirage_wrapper.MIRAGEFM`) images are routed through: `'slo'` for fundus
(no dedicated `fundus` domain exists), `'bscan'` for the OCT script
(explicit, matching `run_cls_tuning.py`'s implicit default). That single
line is the only functional difference from `run_cls_tuning.py` in each
file. Shell wrappers (`run_fundus.sh`, `run_bscan.sh`) mirror `run_uf.sh`:
single dataset per invocation, `./runner` fan-out over `--weights`,
`--seed 0` only.

**Tech Stack:** Python 3 / PyTorch / torchvision / timm / argparse; bash
shell scripts; no test framework exists in this repo (confirmed: no
`tests/` directory, no pytest usage) and the local dev checkout has no
`torch`/`timm` installed and no pretrained weights or datasets — this is a
Windows dev clone, actual runs happen on a separate GPU cluster (HiPerGator,
per the absolute paths in every existing shell script). Verification in
this plan is therefore syntax-level (`py_compile` / `bash -n`) and manual
diff-review against the originals, not functional execution — see Task 6.

Full spec: `docs/superpowers/specs/2026-07-20-public-fundus-bscan-cls-scripts-design.md`

---

### Task 1: `run_cls_tuning_fundus.py`

**Files:**
- Create: `run_cls_tuning_fundus.py`

- [ ] **Step 1: Create the file**

```python
"""Public fundus-photo dataset classification tuning.

MIRAGE has no dedicated 'fundus' input domain (see
mirage_wrapper.DOMAIN_CONF: only 'bscan', 'slo', 'bscanlayermap' exist), so
color fundus photos are routed through 'slo', the only other en-face 2D
domain. Otherwise identical to run_cls_tuning.py: datasets are pre-split
train/val/test/Class_x/ image folders (see docs/classification_benchmark.md),
loaded with torchvision's ImageFolder.
"""
from typing import Callable
from copy import deepcopy
import json
import os
import sys
import hashlib

import argparse
import datetime
import pandas as pd
import time
from pathlib import Path
import socket

import torch
from torch.utils.data import DataLoader

from timm.loss import LabelSmoothingCrossEntropy
from torchvision import datasets

from mutils import misc
from mutils.classification import train_1_epoch, evaluate, EarlyStopping
from mutils.misc import fix_seeds, SortingHelpFormatter
from fm_cls_config import fm_config_factory



def get_args():
    parser = argparse.ArgumentParser(
        'Public fundus-photo classification experiments (MIRAGE, via the slo domain)',
        add_help=True,
        formatter_class=SortingHelpFormatter
    )

    # Model parameters
    parser.add_argument(
        '--input_size', default=None, type=int,
        help='Images input size. If None, it is automatically set to 512 for'
            ' MIRAGE (default: %(default)s)',
    )
    parser.add_argument(
        '--drop_path', type=float, default=0.1, metavar='PCT',
        help='Drop path rate. (default: %(default)s)',
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.05,
        help='Weight decay. (default: %(default)s)',
    )

    # Optimizer parameters
    parser.add_argument(
        '--lr', type=float, default=1e-5, metavar='LR',
        help='Learning rate. (default: %(default)s)',
    )
    parser.add_argument(
        '--layer_decay', type=float, default=0.75,
        help='Layer-wise LR decay. (default: %(default)s)',
    )
    parser.add_argument(
        '--min_lr', type=float, default=1e-8, metavar='LR',
        help='Lower LR bound for cyclic schedulers that hit 0.'
            ' (default: %(default)s)'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=10, metavar='N',
        help='Epochs to warmup LR'
    )
    parser.add_argument(
        '--smoothing', type=float, default=0.1,
        help='Label smoothing. (default: %(default)s)',
    )
    parser.add_argument(
        '--accum_iter', default=1, type=int,
        help='Accumulate gradient iterations (for increasing the effective'
            ' batch size under memory constraints). (default: %(default)s)'
    )

    # Supervised training params
    parser.add_argument(
        '--linear_probing', action='store_true',
        help='Set to True for not training the encoder weights.',
    )
    parser.add_argument(
        '--resume', default='',
        help='Checkpoint to resume from. (default: %(default)s)',
    )
    parser.add_argument(
        '--pool', default='global', type=str,
        choices=['global', 'cls', 'token_mix'],
        help='Pooling method before the final layer. (default: %(default)s)',
    )
    parser.add_argument(
        '--base_output_dir',
        default='./__output/cls_fundus',
        help='Base output directory for saving results. (default: %(default)s)',
    )

    # Data parameters
    parser.add_argument(
        '--num_workers', default=8, type=int,
        help='Number of workers for data loading. (default: %(default)s)',
    )
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes)'
            ' transfer to GPU. (default: %(default)s)',
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Training parameters
    parser.add_argument(
        '--device', default='cuda',
        help='Device to use for training / testing. (default: %(default)s)',
    )
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Seed for reproducibility. (default: %(default)s)',
    )
    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N',
        help='Start epoch. (default: %(default)s)',
    )
    parser.add_argument(
        '--batch_size', default=None, type=int,
        help='Batch size per GPU (effective batch size is batch_size *'
            ' accum_iter * # gpus). "None" for automatic calculation.'
            ' (default: %(default)s)',
    )
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument(
        '--eval', action='store_true',
        help='Wether to run only the evaluation on the test set.'
            ' (default: %(default)s)',
    )
    parser.add_argument(
        '--early_stopping_epochs', default=20, type=int,
        help='Parameter to control how many epochs to wait for the validation'
            ' loss to improve before stopping. (default: %(default)s)',
    )
    parser.add_argument(
        '--early_stopping_delta', default=0.001, type=float,
        help='Parameter to specify the minimum change in the validation metric'
            ' required to consider it an improvement. (default: %(default)s)',
    )
    parser.add_argument(
        '--early_stopping_delta_two', default=0.001, type=float,
        help='Parameter to specify the minimum change in the validation metric two'
            ' required to consider it an improvement. (default: %(default)s)',
    )
    parser.add_argument(
        '--early_start_from', default=20, type=int,
        help='Parameter to specify the epoch to start taking into account'
            ' the early stopping criteria. (default: %(default)s)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Do not run the experiment, just build the model and print the'
            ' information. (default: %(default)s)',
    )
    parser.add_argument(
        '--version', default='v1',
        help='Version of the experiment. (default: %(default)s)',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite the output directory if it exists. (default: %(default)s)',
    )
    parser.add_argument(
        '--val_metric', default='bacc', type=str,
        help='Validation metric to monitor for early stopping. (default: %(default)s)',
    )
    parser.add_argument(
        '--val_metric_two', default='loss', type=str,
        help='Second validation metric to monitor for early stopping. (default: %(default)s)',
    )
    parser.add_argument(
        '--save_predictions', action='store_true',
        help='Save test predictions. (default: %(default)s)',
    )
    parser.add_argument(
        '--fill', default=None, type=float,
        help='Fill value for affine transformations. (default: %(default)s)',
    )
    parser.add_argument(
        '--affine', action='store_true',
        help='Apply random affine transformations. (default: %(default)s)',
    )
    parser.add_argument('--no_affine', action='store_false', dest='affine')
    parser.set_defaults(affine=True)

    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Pre-trained weights to initialise the model with. (required)',
    )
    # Dataset parameters
    required_parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory for the classification datasets. (required)',
    )
    required_parser.add_argument(
        '--data_set',
        type=str,
        required=True,
        help='Dataset directory name. (required)',
    )

    return parser.parse_args()


def process_args(args):
    hostname = socket.gethostname()
    print(f'Running on {hostname}')

    if args.data_root[-1] != '/':
        args.data_root += '/'
    args.data_path = args.data_root + args.data_set

    # Automatic number of classes calculation
    train_data_path = args.data_path + '/train'
    num_classes = 0
    for class_dir in Path(train_data_path).iterdir():
        if class_dir.is_dir():
            num_classes += 1
    num_samples = 0
    for class_dir in Path(train_data_path).iterdir():
        if class_dir.is_dir():
            num_samples += len(list(class_dir.iterdir()))
    args.num_classes = num_classes
    print(f'Number of classes: {num_classes}')
    print(f'Number of training samples: {num_samples}')

    if args.batch_size is None:
        # Automatic batch size calculation
        # Batch size is closest power of 2 to 1/10 of the dataset, with a
        #   maximum of 64.
        args.batch_size = min(64, 2 ** (int(round(num_samples * 0.25)).bit_length() - 1))
        if args.batch_size < 1:
            args.batch_size = 8
    print(f'Batch size: {args.batch_size}')
    return args


def get_output_dir(args, model_name):
    # Set output directory based on some arguments
    output_dir = args.base_output_dir
    if output_dir[-1] != '/':
        output_dir += '/'
    output_dir += f'{args.version}/'
    output_dir += f'{args.seed}/'
    output_dir += f'{args.data_set}/'
    output_dir += f'{model_name}'
    if args.linear_probing:
        output_dir += '_linear'
    else:
        output_dir += '_finetune'
    if args.weights is not None:
        output_dir += '_w'
    return output_dir


def build_dataset(subset, args, build_transform: Callable, augment=False):
    transform = build_transform(subset, augment)
    root = os.path.join(args.data_path, subset)
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def main(args):
    fix_seeds(args.seed)

    device = torch.device(args.device)

    args = process_args(args)

    # No dedicated 'fundus' domain exists in MIRAGE (mirage_wrapper.DOMAIN_CONF
    # only has 'bscan', 'slo', 'bscanlayermap'), so color fundus photos are
    # routed through 'slo', the only other en-face 2D domain. Its input
    # adapter is single-channel by architecture, so FoundModel.build_transform
    # still converts to grayscale -- this is a cross-domain transfer eval, not
    # a full-fidelity color-fundus setup.
    args.uf_modality = 'slo'

    model_config = None
    model_name = None
    for kw in fm_config_factory.keys():
        if kw in args.weights.lower():
            model_config = fm_config_factory[kw](args)
            model_name = kw
            break
    if model_config is None:
        raise ValueError(f"Unknown model: {args.weights}")

    # Initialize the model
    model = model_config.model
    args = model_config.args

    args.output_dir = get_output_dir(args, model_name)

    model_config.set_requires_grad()

    model.to(device)
    # print(model)

    # Print model info
    n_parameters = sum(p.numel() for p in model.parameters())
    n_tr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (N): %.2e' % (n_parameters))
    print('number of params (N):', n_parameters)
    print('number of trainable params (M): %.2e' % (n_tr_parameters))
    print('number of trainable params (M):', n_tr_parameters)

    # Save args in the name of the model as a checksum and in a json file
    args_vars = vars(args).copy()
    # Remove unnecessary keys
    model_config_keys = [
        'accum_iter', 'drop_path', 'early_start_from', 'early_stopping_delta',
        'early_stopping_delta_two', 'early_stopping_epochs', 'fill', 'weights',
        'input_size', 'layer_decay', 'linear_probing', 'lr', 'min_lr', 'model',
        'affine', 'pool', 'smoothing', 'start_epoch', 'val_metric',
        'val_metric_two', 'warmup_epochs', 'weight_decay',
    ]
    for key in list(args_vars.keys()):
        if key not in model_config_keys:
            args_vars.pop(key, None)
    args_str = json.dumps(args_vars, indent=2, sort_keys=True)
    args_checksum = hashlib.md5(args_str.encode('utf-8')).hexdigest()[:8]
    print(f'Args checksum: {args_checksum}')
    args.output_dir += f'_{args_checksum}/'

    output_dir = Path(args.output_dir)

    # Create output directory
    print(f'> Saving to {args.output_dir}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'args.json', 'w') as f:
        f.write(args_str)

    print(f'Args:\n{args_str}')

    if (
        (output_dir / 'test_eval.csv').exists()
        and not args.overwrite
        and not args.save_predictions
    ):
        print('Experiment already run. Exiting.')
        sys.exit(0)

    if (
        (output_dir / 'predictions.npz').exists()
        and args.save_predictions
        and not args.overwrite
    ):
        print('Predictions already saved. Exiting.')
        sys.exit(0)

    if args.dry_run:
        print('Dry run. Exiting.')
        sys.exit(0)

    optimizer = model_config.get_optimizer(model)

    dataset_train = None
    dataset_val = None
    if not args.eval:
        augment_train = True
        shuffle = True
        dataset_train = build_dataset(
            subset='train',
            args=args,
            build_transform=model_config.build_transform,
            augment=augment_train
        )
        try:
            print(dataset_train.class_to_idx)
        except AttributeError:
            pass
        train_loader = DataLoader(
            dataset_train,
            shuffle=shuffle,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f'Number of training samples: {len(dataset_train)}')

        dataset_val = build_dataset(
            subset='val',
            args=args,
            build_transform=model_config.build_transform,
            augment=False,
        )
        valid_loader = DataLoader(
            dataset_val,
            shuffle=shuffle,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f'Number of validation samples: {len(dataset_val)}')
    else:
        train_loader = None
        valid_loader = None

    if 'cross_train' not in args.data_set.lower():
        dataset_test = build_dataset(
            subset='test',
            args=args,
            build_transform=model_config.build_transform,
            augment=False
        )
        test_loader = DataLoader(
            dataset_test,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f'Number of test samples: {len(dataset_test)}')
    else:
        test_loader = None

    if args.save_predictions:
        assert test_loader is not None
        print('Getting predictions for the best checkpoint')
        args.resume = f'{args.output_dir}/checkpoint-best-model.pth'
        misc.load_model(args=args, model=model, optimizer=None)
        save_path = args.output_dir
        test_stats = evaluate(
            model, test_loader, 'Best', device, args.num_classes, mode='Test',
            save_predictions=True, save_path=save_path
        )
        exit(0)

    if not args.eval:
        if args.smoothing > 0.0:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        greater_is_better = args.val_metric != 'loss'
        greater_is_better_two = args.val_metric_two != 'loss'

        # Initialize early stopping object
        early_stopping = EarlyStopping(
            patience=args.early_stopping_epochs,
            delta=args.early_stopping_delta,
            delta_two=args.early_stopping_delta_two,
            greater_is_better=greater_is_better,
            greater_is_better_two=greater_is_better_two,
            start_from=args.early_start_from,
        )

        start_time = time.time()
        train_stats_all, val_stats_all = [], []
        best_model = argparse.Namespace()
        assert train_loader is not None
        assert valid_loader is not None
        for epoch in range(args.start_epoch, args.epochs):
            try:
                train_stats = train_1_epoch(
                    model,
                    criterion,
                    train_loader,
                    optimizer,
                    device,
                    epoch,
                    args=args,
                )
            except ValueError as e:
                print('Early stopping')
                print(e)
                break

            train_stats_all.append(train_stats.values())

            val_stats = evaluate(
                model, valid_loader, epoch, device, args.num_classes,
                mode='Valid', args=args
            )
            assert val_stats is not None
            val_stats_all.append(val_stats.values())

            # If the validation loss has improved, save checkpoint
            # Check if early stopping criterion is met
            is_best = early_stopping(val_stats[args.val_metric], val_stats[args.val_metric_two], epoch)
            if early_stopping.early_stop:
                print(f'Early stopping @ epoch {epoch}')
                break
            else:
                if is_best and args.output_dir:
                    # Save in memory to avoid writing to disk all the time
                    best_model= argparse.Namespace(
                        # NOTE: Pass model and optimizer state_dicts as
                        #   values (copies), not as references.
                        model=deepcopy(model.state_dict()),
                        optimizer=deepcopy(optimizer.state_dict()),
                        epoch=epoch,
                    )
                    # misc.save_model(args, epoch, model, optimizer)
                    print(
                        f'New best {model_config.__class__.__name__} model'
                        f' on {args.data_set} with seed {args.seed}'
                        f' @ epoch {epoch}'
                        f'\n\t({early_stopping.best_value}, {early_stopping.best_value_two})'
                    )

        misc.save_model(args, epoch=best_model.epoch, model=best_model.model, optimizer=best_model.optimizer)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # Save evaluation results
        pd.DataFrame(
            data=train_stats_all,
            columns=['Epoch', 'Loss', 'BAcc', 'F1-score']  # type: ignore
        ).to_csv(f'{args.output_dir}/train_eval.csv', index=False)


        pd.DataFrame(
            data=val_stats_all,
            columns=['Epoch', 'Loss', 'BAcc', 'AUROC', 'AP', 'F1-score', 'MCC'],  # type: ignore
        ).to_csv(f'{args.output_dir}/valid_eval.csv', index=False)

    if test_loader is not None:
        # Evaluate on the best checkpoint
        args.resume = f'{args.output_dir}/checkpoint-best-model.pth'
        misc.load_model(args=args, model=model, optimizer=optimizer)
        test_stats = evaluate(
            model, test_loader, 'Best', device, args.num_classes, mode='Test'
        )
        assert test_stats is not None
        pd.DataFrame(
            data=[test_stats.values()],
            columns=['Epoch', 'Loss', 'BAcc', 'AUROC', 'AP', 'F1-score', 'MCC'],  # type: ignore
        ).to_csv(f'{args.output_dir}/test_eval.csv', index=False)



if __name__ == '__main__':
    args = get_args()
    main(args)
```

- [ ] **Step 2: Byte-diff against `run_cls_tuning.py` to confirm the only deltas are the module docstring, the parser description string, the `--base_output_dir` default, and the `args.uf_modality = 'slo'` block**

Run: `diff run_cls_tuning.py run_cls_tuning_fundus.py`
Expected: only the 4 hunks described above; everything else identical.

- [ ] **Step 3: Commit**

```bash
git add run_cls_tuning_fundus.py
git commit -m "$(cat <<'EOF'
Add run_cls_tuning_fundus.py for public fundus-photo benchmark datasets

Structural copy of run_cls_tuning.py; routes images through MIRAGE's
'slo' domain since no dedicated 'fundus' domain exists.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `run_cls_tuning_bscan.py`

**Files:**
- Create: `run_cls_tuning_bscan.py`

- [ ] **Step 1: Create the file**

```python
"""Public OCT B-scan dataset classification tuning.

Dedicated entry point for the public OCT B-scan benchmark datasets (e.g.
duke14, glaucoma, oimhs, umn) referenced from OphFoundation's own
finetune-UF-benchmark_*_single.sh scripts. Functionally identical to
run_cls_tuning.py (which already defaults every dataset to MIRAGE's 'bscan'
domain): datasets are pre-split train/val/test/Class_x/ image folders (see
docs/classification_benchmark.md), loaded with torchvision's ImageFolder.
"""
from typing import Callable
from copy import deepcopy
import json
import os
import sys
import hashlib

import argparse
import datetime
import pandas as pd
import time
from pathlib import Path
import socket

import torch
from torch.utils.data import DataLoader

from timm.loss import LabelSmoothingCrossEntropy
from torchvision import datasets

from mutils import misc
from mutils.classification import train_1_epoch, evaluate, EarlyStopping
from mutils.misc import fix_seeds, SortingHelpFormatter
from fm_cls_config import fm_config_factory



def get_args():
    parser = argparse.ArgumentParser(
        'Public OCT B-scan classification experiments (MIRAGE, bscan domain)',
        add_help=True,
        formatter_class=SortingHelpFormatter
    )

    # Model parameters
    parser.add_argument(
        '--input_size', default=None, type=int,
        help='Images input size. If None, it is automatically set to 512 for'
            ' MIRAGE (default: %(default)s)',
    )
    parser.add_argument(
        '--drop_path', type=float, default=0.1, metavar='PCT',
        help='Drop path rate. (default: %(default)s)',
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.05,
        help='Weight decay. (default: %(default)s)',
    )

    # Optimizer parameters
    parser.add_argument(
        '--lr', type=float, default=1e-5, metavar='LR',
        help='Learning rate. (default: %(default)s)',
    )
    parser.add_argument(
        '--layer_decay', type=float, default=0.75,
        help='Layer-wise LR decay. (default: %(default)s)',
    )
    parser.add_argument(
        '--min_lr', type=float, default=1e-8, metavar='LR',
        help='Lower LR bound for cyclic schedulers that hit 0.'
            ' (default: %(default)s)'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=10, metavar='N',
        help='Epochs to warmup LR'
    )
    parser.add_argument(
        '--smoothing', type=float, default=0.1,
        help='Label smoothing. (default: %(default)s)',
    )
    parser.add_argument(
        '--accum_iter', default=1, type=int,
        help='Accumulate gradient iterations (for increasing the effective'
            ' batch size under memory constraints). (default: %(default)s)'
    )

    # Supervised training params
    parser.add_argument(
        '--linear_probing', action='store_true',
        help='Set to True for not training the encoder weights.',
    )
    parser.add_argument(
        '--resume', default='',
        help='Checkpoint to resume from. (default: %(default)s)',
    )
    parser.add_argument(
        '--pool', default='global', type=str,
        choices=['global', 'cls', 'token_mix'],
        help='Pooling method before the final layer. (default: %(default)s)',
    )
    parser.add_argument(
        '--base_output_dir',
        default='./__output/cls_bscan',
        help='Base output directory for saving results. (default: %(default)s)',
    )

    # Data parameters
    parser.add_argument(
        '--num_workers', default=8, type=int,
        help='Number of workers for data loading. (default: %(default)s)',
    )
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes)'
            ' transfer to GPU. (default: %(default)s)',
    )
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Training parameters
    parser.add_argument(
        '--device', default='cuda',
        help='Device to use for training / testing. (default: %(default)s)',
    )
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Seed for reproducibility. (default: %(default)s)',
    )
    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N',
        help='Start epoch. (default: %(default)s)',
    )
    parser.add_argument(
        '--batch_size', default=None, type=int,
        help='Batch size per GPU (effective batch size is batch_size *'
            ' accum_iter * # gpus). "None" for automatic calculation.'
            ' (default: %(default)s)',
    )
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument(
        '--eval', action='store_true',
        help='Wether to run only the evaluation on the test set.'
            ' (default: %(default)s)',
    )
    parser.add_argument(
        '--early_stopping_epochs', default=20, type=int,
        help='Parameter to control how many epochs to wait for the validation'
            ' loss to improve before stopping. (default: %(default)s)',
    )
    parser.add_argument(
        '--early_stopping_delta', default=0.001, type=float,
        help='Parameter to specify the minimum change in the validation metric'
            ' required to consider it an improvement. (default: %(default)s)',
    )
    parser.add_argument(
        '--early_stopping_delta_two', default=0.001, type=float,
        help='Parameter to specify the minimum change in the validation metric two'
            ' required to consider it an improvement. (default: %(default)s)',
    )
    parser.add_argument(
        '--early_start_from', default=20, type=int,
        help='Parameter to specify the epoch to start taking into account'
            ' the early stopping criteria. (default: %(default)s)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Do not run the experiment, just build the model and print the'
            ' information. (default: %(default)s)',
    )
    parser.add_argument(
        '--version', default='v1',
        help='Version of the experiment. (default: %(default)s)',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite the output directory if it exists. (default: %(default)s)',
    )
    parser.add_argument(
        '--val_metric', default='bacc', type=str,
        help='Validation metric to monitor for early stopping. (default: %(default)s)',
    )
    parser.add_argument(
        '--val_metric_two', default='loss', type=str,
        help='Second validation metric to monitor for early stopping. (default: %(default)s)',
    )
    parser.add_argument(
        '--save_predictions', action='store_true',
        help='Save test predictions. (default: %(default)s)',
    )
    parser.add_argument(
        '--fill', default=None, type=float,
        help='Fill value for affine transformations. (default: %(default)s)',
    )
    parser.add_argument(
        '--affine', action='store_true',
        help='Apply random affine transformations. (default: %(default)s)',
    )
    parser.add_argument('--no_affine', action='store_false', dest='affine')
    parser.set_defaults(affine=True)

    required_parser = parser.add_argument_group('required arguments')
    required_parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Pre-trained weights to initialise the model with. (required)',
    )
    # Dataset parameters
    required_parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory for the classification datasets. (required)',
    )
    required_parser.add_argument(
        '--data_set',
        type=str,
        required=True,
        help='Dataset directory name. (required)',
    )

    return parser.parse_args()


def process_args(args):
    hostname = socket.gethostname()
    print(f'Running on {hostname}')

    if args.data_root[-1] != '/':
        args.data_root += '/'
    args.data_path = args.data_root + args.data_set

    # Automatic number of classes calculation
    train_data_path = args.data_path + '/train'
    num_classes = 0
    for class_dir in Path(train_data_path).iterdir():
        if class_dir.is_dir():
            num_classes += 1
    num_samples = 0
    for class_dir in Path(train_data_path).iterdir():
        if class_dir.is_dir():
            num_samples += len(list(class_dir.iterdir()))
    args.num_classes = num_classes
    print(f'Number of classes: {num_classes}')
    print(f'Number of training samples: {num_samples}')

    if args.batch_size is None:
        # Automatic batch size calculation
        # Batch size is closest power of 2 to 1/10 of the dataset, with a
        #   maximum of 64.
        args.batch_size = min(64, 2 ** (int(round(num_samples * 0.25)).bit_length() - 1))
        if args.batch_size < 1:
            args.batch_size = 8
    print(f'Batch size: {args.batch_size}')
    return args


def get_output_dir(args, model_name):
    # Set output directory based on some arguments
    output_dir = args.base_output_dir
    if output_dir[-1] != '/':
        output_dir += '/'
    output_dir += f'{args.version}/'
    output_dir += f'{args.seed}/'
    output_dir += f'{args.data_set}/'
    output_dir += f'{model_name}'
    if args.linear_probing:
        output_dir += '_linear'
    else:
        output_dir += '_finetune'
    if args.weights is not None:
        output_dir += '_w'
    return output_dir


def build_dataset(subset, args, build_transform: Callable, augment=False):
    transform = build_transform(subset, augment)
    root = os.path.join(args.data_path, subset)
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def main(args):
    fix_seeds(args.seed)

    device = torch.device(args.device)

    args = process_args(args)

    # Explicit for clarity/documentation -- MIRAGEFM already defaults to
    # 'bscan' when args.uf_modality is unset, so this does not change
    # behavior relative to run_cls_tuning.py. Spelled out so this script
    # does not silently depend on that implicit default.
    args.uf_modality = 'bscan'

    model_config = None
    model_name = None
    for kw in fm_config_factory.keys():
        if kw in args.weights.lower():
            model_config = fm_config_factory[kw](args)
            model_name = kw
            break
    if model_config is None:
        raise ValueError(f"Unknown model: {args.weights}")

    # Initialize the model
    model = model_config.model
    args = model_config.args

    args.output_dir = get_output_dir(args, model_name)

    model_config.set_requires_grad()

    model.to(device)
    # print(model)

    # Print model info
    n_parameters = sum(p.numel() for p in model.parameters())
    n_tr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (N): %.2e' % (n_parameters))
    print('number of params (N):', n_parameters)
    print('number of trainable params (M): %.2e' % (n_tr_parameters))
    print('number of trainable params (M):', n_tr_parameters)

    # Save args in the name of the model as a checksum and in a json file
    args_vars = vars(args).copy()
    # Remove unnecessary keys
    model_config_keys = [
        'accum_iter', 'drop_path', 'early_start_from', 'early_stopping_delta',
        'early_stopping_delta_two', 'early_stopping_epochs', 'fill', 'weights',
        'input_size', 'layer_decay', 'linear_probing', 'lr', 'min_lr', 'model',
        'affine', 'pool', 'smoothing', 'start_epoch', 'val_metric',
        'val_metric_two', 'warmup_epochs', 'weight_decay',
    ]
    for key in list(args_vars.keys()):
        if key not in model_config_keys:
            args_vars.pop(key, None)
    args_str = json.dumps(args_vars, indent=2, sort_keys=True)
    args_checksum = hashlib.md5(args_str.encode('utf-8')).hexdigest()[:8]
    print(f'Args checksum: {args_checksum}')
    args.output_dir += f'_{args_checksum}/'

    output_dir = Path(args.output_dir)

    # Create output directory
    print(f'> Saving to {args.output_dir}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'args.json', 'w') as f:
        f.write(args_str)

    print(f'Args:\n{args_str}')

    if (
        (output_dir / 'test_eval.csv').exists()
        and not args.overwrite
        and not args.save_predictions
    ):
        print('Experiment already run. Exiting.')
        sys.exit(0)

    if (
        (output_dir / 'predictions.npz').exists()
        and args.save_predictions
        and not args.overwrite
    ):
        print('Predictions already saved. Exiting.')
        sys.exit(0)

    if args.dry_run:
        print('Dry run. Exiting.')
        sys.exit(0)

    optimizer = model_config.get_optimizer(model)

    dataset_train = None
    dataset_val = None
    if not args.eval:
        augment_train = True
        shuffle = True
        dataset_train = build_dataset(
            subset='train',
            args=args,
            build_transform=model_config.build_transform,
            augment=augment_train
        )
        try:
            print(dataset_train.class_to_idx)
        except AttributeError:
            pass
        train_loader = DataLoader(
            dataset_train,
            shuffle=shuffle,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f'Number of training samples: {len(dataset_train)}')

        dataset_val = build_dataset(
            subset='val',
            args=args,
            build_transform=model_config.build_transform,
            augment=False,
        )
        valid_loader = DataLoader(
            dataset_val,
            shuffle=shuffle,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f'Number of validation samples: {len(dataset_val)}')
    else:
        train_loader = None
        valid_loader = None

    if 'cross_train' not in args.data_set.lower():
        dataset_test = build_dataset(
            subset='test',
            args=args,
            build_transform=model_config.build_transform,
            augment=False
        )
        test_loader = DataLoader(
            dataset_test,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f'Number of test samples: {len(dataset_test)}')
    else:
        test_loader = None

    if args.save_predictions:
        assert test_loader is not None
        print('Getting predictions for the best checkpoint')
        args.resume = f'{args.output_dir}/checkpoint-best-model.pth'
        misc.load_model(args=args, model=model, optimizer=None)
        save_path = args.output_dir
        test_stats = evaluate(
            model, test_loader, 'Best', device, args.num_classes, mode='Test',
            save_predictions=True, save_path=save_path
        )
        exit(0)

    if not args.eval:
        if args.smoothing > 0.0:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        greater_is_better = args.val_metric != 'loss'
        greater_is_better_two = args.val_metric_two != 'loss'

        # Initialize early stopping object
        early_stopping = EarlyStopping(
            patience=args.early_stopping_epochs,
            delta=args.early_stopping_delta,
            delta_two=args.early_stopping_delta_two,
            greater_is_better=greater_is_better,
            greater_is_better_two=greater_is_better_two,
            start_from=args.early_start_from,
        )

        start_time = time.time()
        train_stats_all, val_stats_all = [], []
        best_model = argparse.Namespace()
        assert train_loader is not None
        assert valid_loader is not None
        for epoch in range(args.start_epoch, args.epochs):
            try:
                train_stats = train_1_epoch(
                    model,
                    criterion,
                    train_loader,
                    optimizer,
                    device,
                    epoch,
                    args=args,
                )
            except ValueError as e:
                print('Early stopping')
                print(e)
                break

            train_stats_all.append(train_stats.values())

            val_stats = evaluate(
                model, valid_loader, epoch, device, args.num_classes,
                mode='Valid', args=args
            )
            assert val_stats is not None
            val_stats_all.append(val_stats.values())

            # If the validation loss has improved, save checkpoint
            # Check if early stopping criterion is met
            is_best = early_stopping(val_stats[args.val_metric], val_stats[args.val_metric_two], epoch)
            if early_stopping.early_stop:
                print(f'Early stopping @ epoch {epoch}')
                break
            else:
                if is_best and args.output_dir:
                    # Save in memory to avoid writing to disk all the time
                    best_model= argparse.Namespace(
                        # NOTE: Pass model and optimizer state_dicts as
                        #   values (copies), not as references.
                        model=deepcopy(model.state_dict()),
                        optimizer=deepcopy(optimizer.state_dict()),
                        epoch=epoch,
                    )
                    # misc.save_model(args, epoch, model, optimizer)
                    print(
                        f'New best {model_config.__class__.__name__} model'
                        f' on {args.data_set} with seed {args.seed}'
                        f' @ epoch {epoch}'
                        f'\n\t({early_stopping.best_value}, {early_stopping.best_value_two})'
                    )

        misc.save_model(args, epoch=best_model.epoch, model=best_model.model, optimizer=best_model.optimizer)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # Save evaluation results
        pd.DataFrame(
            data=train_stats_all,
            columns=['Epoch', 'Loss', 'BAcc', 'F1-score']  # type: ignore
        ).to_csv(f'{args.output_dir}/train_eval.csv', index=False)


        pd.DataFrame(
            data=val_stats_all,
            columns=['Epoch', 'Loss', 'BAcc', 'AUROC', 'AP', 'F1-score', 'MCC'],  # type: ignore
        ).to_csv(f'{args.output_dir}/valid_eval.csv', index=False)

    if test_loader is not None:
        # Evaluate on the best checkpoint
        args.resume = f'{args.output_dir}/checkpoint-best-model.pth'
        misc.load_model(args=args, model=model, optimizer=optimizer)
        test_stats = evaluate(
            model, test_loader, 'Best', device, args.num_classes, mode='Test'
        )
        assert test_stats is not None
        pd.DataFrame(
            data=[test_stats.values()],
            columns=['Epoch', 'Loss', 'BAcc', 'AUROC', 'AP', 'F1-score', 'MCC'],  # type: ignore
        ).to_csv(f'{args.output_dir}/test_eval.csv', index=False)



if __name__ == '__main__':
    args = get_args()
    main(args)
```

- [ ] **Step 2: Byte-diff against `run_cls_tuning.py`**

Run: `diff run_cls_tuning.py run_cls_tuning_bscan.py`
Expected: only the module docstring, the parser description string, the
`--base_output_dir` default, and the `args.uf_modality = 'bscan'` block
(with its comment) differ.

- [ ] **Step 3: Commit**

```bash
git add run_cls_tuning_bscan.py
git commit -m "$(cat <<'EOF'
Add run_cls_tuning_bscan.py for public OCT B-scan benchmark datasets

Dedicated entry point for datasets like duke14/glaucoma/oimhs/umn.
Functionally same pipeline as run_cls_tuning.py (bscan domain +
ImageFolder split); a named entry point, not new loading logic.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `run_fundus.sh`

**Files:**
- Create: `run_fundus.sh`

- [ ] **Step 1: Create the file**

```bash

source ./venv/bin/activate


# Public fundus-photo classification (color fundus photos routed through
#   MIRAGE's 'slo' domain -- see run_cls_tuning_fundus.py; MIRAGE has no
#   dedicated 'fundus' domain).
#
# Datasets follow MIRAGE's "public dataset setting": pre-split
#   train/val/test/Class_x/ image folders under --data_root (see
#   docs/classification_benchmark.md). Number of classes is auto-inferred
#   from the folder structure; the reference class counts below (from
#   OphFoundation's 2025-1212-finetune-publicbench-fundus-params.sh) are
#   for documentation only.
#
# Known datasets (name: num_class, per OphFoundation reference):
#   Glaucoma_fundus:3 IDRiD_data:5 JSIEC:39 MESSIDOR2:5 PAPILA:3 Retina:4 APTOS2019:5
DATASET=${1:-"Glaucoma_fundus"}
LINEAR_PROBING=${2:-true}  # true: freeze encoder (linear probe); false: full fine-tune

# EDIT ME: root containing pre-split train/val/test/Class_x/ folders for
#   these public fundus datasets (see docs/classification_benchmark.md).
DATA_ROOT="/orange/ruogu.fang/tienyuchang/MIRAGE_data/cls_fundus_public/"

LINEAR_PROBING_FLAG=""
if [ "$LINEAR_PROBING" = "true" ]; then
    LINEAR_PROBING_FLAG="--linear_probing"
fi

./runner python run_cls_tuning_fundus.py \
    --runners 1 \
    -- \
    --version v1 \
    --seed 0 \
    --weights \
        /orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth \
        /orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth \
    $LINEAR_PROBING_FLAG \
    --data_root \
        $DATA_ROOT \
    --data_set \
        $DATASET \
    --base_output_dir \
        /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_fundus
exit
```

- [ ] **Step 2: Commit**

```bash
git add run_fundus.sh
git commit -m "$(cat <<'EOF'
Add run_fundus.sh launcher for run_cls_tuning_fundus.py

Single-dataset wrapper modeled on run_uf.sh: ./runner fan-out over
--weights, seed 0 only (no multi-seed sweep).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `run_bscan.sh`

**Files:**
- Create: `run_bscan.sh`

- [ ] **Step 1: Create the file**

```bash

source ./venv/bin/activate


# Public OCT B-scan classification (MIRAGE's default 'bscan' domain -- see
#   run_cls_tuning_bscan.py; a dedicated entry point for these datasets,
#   functionally the same pipeline as run_cls_tuning.py).
#
# Datasets follow MIRAGE's "public dataset setting": pre-split
#   train/val/test/Class_x/ image folders under --data_root (see
#   docs/classification_benchmark.md). Number of classes is auto-inferred
#   from the folder structure.
#
# Known datasets (per OphFoundation's finetune-UF-benchmark_*_single.sh
#   reference scripts, which benchmark against these same public OCT
#   datasets via their own CSV/fold pipeline): duke14, glaucoma, oimhs, umn
DATASET=${1:-"duke14"}
LINEAR_PROBING=${2:-true}  # true: freeze encoder (linear probe); false: full fine-tune

# EDIT ME: root containing pre-split train/val/test/Class_x/ folders for
#   these public OCT B-scan datasets (see docs/classification_benchmark.md).
DATA_ROOT="/orange/ruogu.fang/tienyuchang/MIRAGE_data/cls_bscan_public/"

LINEAR_PROBING_FLAG=""
if [ "$LINEAR_PROBING" = "true" ]; then
    LINEAR_PROBING_FLAG="--linear_probing"
fi

./runner python run_cls_tuning_bscan.py \
    --runners 1 \
    -- \
    --version v1 \
    --seed 0 \
    --weights \
        /orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth \
        /orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth \
    $LINEAR_PROBING_FLAG \
    --data_root \
        $DATA_ROOT \
    --data_set \
        $DATASET \
    --base_output_dir \
        /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_bscan
exit
```

- [ ] **Step 2: Commit**

```bash
git add run_bscan.sh
git commit -m "$(cat <<'EOF'
Add run_bscan.sh launcher for run_cls_tuning_bscan.py

Single-dataset wrapper modeled on run_uf.sh: ./runner fan-out over
--weights, seed 0 only (no multi-seed sweep).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Document the new scripts

**Files:**
- Modify: `docs/classification_tuning.md` (insert a new section after line 57, i.e. right after the existing "## Adding a new dataset" section and before "## UF cohort (CSV-driven datasets)")

- [ ] **Step 1: Insert the new section**

Insert this text immediately after the existing:

```markdown
## Adding a new dataset

To add a new dataset, you need to respect the dataset structure indicated in [docs/classification_benchmark.md](../docs/classification_benchmark.md).
```

and before the existing:

```markdown
## UF cohort (CSV-driven datasets)
```

New section to insert:

```markdown
## Public fundus and OCT B-scan benchmark scripts

`run_cls_tuning_fundus.py` and `run_cls_tuning_bscan.py` are dedicated
entry points for two families of public benchmark datasets referenced from
[OphFoundation](https://github.com/franciszchen/OphFoundation)'s own
finetuning scripts. Both still use MIRAGE's "public dataset setting"
(pre-split `train/val/test/Class_x/` image folders, see
[docs/classification_benchmark.md](../docs/classification_benchmark.md)) —
not OphFoundation's CSV/fold-split pipeline.

- `run_cls_tuning_fundus.py`: for color fundus photo datasets (e.g.
  `Glaucoma_fundus`, `IDRiD_data`, `JSIEC`, `MESSIDOR2`, `PAPILA`, `Retina`,
  `APTOS2019`). MIRAGE has no dedicated `fundus` domain, so images are
  routed through `slo`, the only other en-face 2D domain — see
  `mirage_wrapper.DOMAIN_CONF`. Its input adapter is single-channel by
  architecture, so this is a cross-domain transfer evaluation, not a
  full-fidelity color setup.
- `run_cls_tuning_bscan.py`: for public OCT B-scan benchmark datasets (e.g.
  `duke14`, `glaucoma`, `oimhs`, `umn`). Uses MIRAGE's default `bscan`
  domain — functionally the same data pipeline as `run_cls_tuning.py`,
  just a dedicated entry point for these datasets.

Both scripts take the same `--weights`/`--data_root`/`--data_set` arguments
as `run_cls_tuning.py`. See `run_fundus.sh`/`run_bscan.sh` for ready-to-edit
copies of these commands.
```

- [ ] **Step 2: Verify placement**

Run: `grep -n "^## " docs/classification_tuning.md`
Expected order: `## Requirements`, `## Data`, `## Usage`, `## Adding a new dataset`, `## Public fundus and OCT B-scan benchmark scripts`, `## UF cohort (CSV-driven datasets)`, `### True multi-modal ...`, `### Subsampling ...`, `### Metrics and wandb logging`, `## Adding a new model`.

- [ ] **Step 3: Commit**

```bash
git add docs/classification_tuning.md
git commit -m "$(cat <<'EOF'
Document run_cls_tuning_fundus.py and run_cls_tuning_bscan.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Verification

No test framework or GPU/data environment exists in this local checkout
(confirmed: no `tests/` dir, no pytest anywhere in the repo; `python -c
"import torch"` fails locally — `torch`/`timm`/`mirage` deps aren't
installed here, and there are no pretrained `.pth` weights or prepared
datasets in this checkout). Real functional verification (`--help`,
`--dry-run`, an actual training run) requires the GPU cluster environment
these scripts are meant to run on. This task covers what **can** be
verified locally: syntax correctness and that the intended deltas are the
only deltas.

**Files:**
- None (verification only)

- [ ] **Step 1: Python syntax check**

Run: `python -m py_compile run_cls_tuning_fundus.py run_cls_tuning_bscan.py`
Expected: exits 0, no output.

- [ ] **Step 2: Shell syntax check**

Run: `bash -n run_fundus.sh && bash -n run_bscan.sh`
Expected: exits 0, no output.

- [ ] **Step 3: Confirm the diffs are exactly the intended ones**

Run: `diff run_cls_tuning.py run_cls_tuning_fundus.py` and
`diff run_cls_tuning.py run_cls_tuning_bscan.py`
Expected: matches Task 1 Step 2 / Task 2 Step 2 exactly — no stray
differences (e.g. no accidental whitespace changes, no dropped lines).

- [ ] **Step 4: Tell the user what's not yet verified**

Report explicitly (not just implied by silence) that `--help` output,
`--dry-run`, and an actual training run have NOT been executed anywhere in
this session, and must be checked by the user on their GPU
cluster/environment where `torch`/`timm`/MIRAGE weights are available,
before relying on these scripts for real experiments.
