from argparse import HelpFormatter
from operator import attrgetter
import random

import torch
import numpy as np
from torch.backends import cudnn
from torch.utils.data import Subset



class SortingHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


def fix_seeds(seed):
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


def subsample_class_balanced(dataset, target_size, seed):
    """Class-balanced subsample of a dataset exposing `.targets`,
    `.data_frame`, and `.root_dir` (e.g. `UFCohortDataset`,
    `UFCohortMultiModalDataset`; see `mutils/dataset_uf.py`).

    Each class keeps a share of `target_size` proportional to its share of
    the full dataset, with every class guaranteed at least one sample; a
    class with fewer samples than its computed share just keeps all of them
    (the shortfall is not redistributed to other classes). Falls back to
    plain random sampling if `target_size` is smaller than the number of
    classes (there's no way to keep every class in that case).
    """
    targets = np.array(dataset.targets)
    unique_classes, class_counts = np.unique(targets, return_counts=True)
    n_classes = len(unique_classes)
    rng = np.random.RandomState(seed)

    if target_size >= len(dataset):
        print(f'new_subset_num ({target_size}) >= dataset size ({len(dataset)}); using full dataset')
        return dataset

    if target_size < n_classes:
        print(
            f'Warning: new_subset_num ({target_size}) < number of classes'
            f' ({n_classes}); falling back to plain random sampling'
        )
        selected_indices = rng.choice(len(dataset), target_size, replace=False).tolist()
    else:
        class_ratios = class_counts / len(targets)
        selected_indices = []
        for class_label, ratio, available in zip(unique_classes, class_ratios, class_counts):
            class_indices = np.where(targets == class_label)[0]
            rng.shuffle(class_indices)
            n_select = min(int((target_size - n_classes) * ratio) + 1, available)
            selected_indices.extend(class_indices[:n_select].tolist())
        selected_indices.sort()

    subset = Subset(dataset, selected_indices)
    subset.targets = [dataset.targets[i] for i in selected_indices]
    subset.data_frame = dataset.data_frame.iloc[selected_indices].reset_index(drop=True)
    subset.root_dir = dataset.root_dir

    new_classes, new_counts = np.unique(np.array(subset.targets), return_counts=True)
    print(
        f'Class-balanced subset: {len(dataset)} -> {len(subset)} samples'
        f' (target {target_size}); class counts: {dict(zip(new_classes.tolist(), new_counts.tolist()))}'
    )
    return subset


def save_model(args, epoch, model, optimizer, loss_scaler=None):
    torch.save(
        {
            'model': model,
            'optimizer': optimizer,
            'epoch': epoch,
            'args': args,
            'loss_scaler': loss_scaler,
        },
        f'{args.output_dir}/checkpoint-best-model.pth',
    )


def load_model(args, model, optimizer, loss_scaler=None):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print('Resume checkpoint %s' % args.resume)
        if (
            'optimizer' in checkpoint
            and 'epoch' in checkpoint
            and not (hasattr(args, 'eval') and args.eval)
            and optimizer is not None
        ):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint and loss_scaler is not None:
                loss_scaler.load_state_dict(checkpoint['loss_scaler'])
            print('With optim & sched!')
