"""
Training script. Should be pretty adaptable to whatever.
"""
import argparse
import os
import shutil

import h5py
import random
import multiprocessing
import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm

from dataloaders.vcr_kd import VCR, VCRLoader, get_batch_tasks
# from dataloaders.vcr import VCR, VCRLoader
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

# This is needed to make the imports work
from allennlp.models import Model
import models

#################################
#################################
######## Data loading stuff
#################################
#################################

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '-params',
    dest='params',
    help='Params location',
    type=str,
)
parser.add_argument(
    '-folder',
    dest='folder',
    help='folder location',
    type=str,
)
parser.add_argument(
    '-batch_size',
    dest='batch_size',
    type=int,
    default=96
)

args = parser.parse_args()

params = Params.from_file(args.params)
train, val = VCR.splits(tasks=('0',), embs_to_load='bert_da',
                        only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True))
NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")

def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
                non_blocking=True)
    return td
num_workers = 12
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': args.batch_size // NUM_GPUS, 'num_gpus': NUM_GPUS, 'num_workers': num_workers}
train_loader = VCRLoader.from_dataset(train, **loader_params, shuffle=False, drop_last=False)
val_loader = VCRLoader.from_dataset(val, **loader_params, shuffle=False, drop_last=False)
# test_loader = VCRLoader.from_dataset(test, **loader_params)

ARGS_RESET_EVERY = 100
# print("Loading {} for {}".format(params['model'].get('type', 'WTF?'), 'rationales' if args.rationale else 'answer'), flush=True)
model = Model.from_params(vocab=train.vocab, params=params['model'])
for submodule in model.detector.backbone.modules():
    if isinstance(submodule, BatchNorm2d):
        submodule.track_running_stats = False
    for p in submodule.parameters():
        p.requires_grad = False

model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()
# Load best
restore_best_checkpoint(model, args.folder)
model.eval()
train_logits = []
train_probs = []
train_labels = []
train_features_penult = []
train_features_last = []
for b, (time_per_batch, batch) in enumerate(time_batch(tqdm(train_loader))):
    with torch.no_grad():
        batch = _to_gpu(batch)
        batch = get_batch_tasks(batch, ('0',))['0']
        output_dict = model(**batch)
        train_logits.append(output_dict['label_logits'].detach().cpu().numpy())
        train_probs.append(output_dict['label_probs'].detach().cpu().numpy())
        train_labels.append(batch['label'].detach().cpu().numpy())

        train_features_penult.append(output_dict['features_penult'].detach().cpu().numpy())
        train_features_last.append(output_dict['features_last'].detach().cpu().numpy())

val_logits = []
val_probs = []
val_labels = []
val_features_penult = []
val_features_last = []

for b, (time_per_batch, batch) in enumerate(time_batch(tqdm(val_loader))):
    with torch.no_grad():
        batch = _to_gpu(batch)
        batch = get_batch_tasks(batch, ('0',))['0']
        output_dict = model(**batch)
        val_logits.append(output_dict['label_logits'].detach().cpu().numpy())
        val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
        val_labels.append(batch['label'].detach().cpu().numpy())

        val_features_penult.append(output_dict['features_penult'].detach().cpu().numpy())
        val_features_last.append(output_dict['features_last'].detach().cpu().numpy())


train_logits = np.concatenate(train_logits, 0)
train_probs = np.concatenate(train_probs, 0)
train_labels = np.concatenate(train_labels, 0)

train_features_penult = np.concatenate(train_features_penult, 0)
train_features_last = np.concatenate(train_features_last, 0)

acc = float(np.mean(train_labels == train_probs.argmax(1)))
# np.save(os.path.join(args.folder, f'train_logits.npy'), train_logits)
# np.save(os.path.join(args.folder, f'train_preds.npy'), train_probs)
# np.save(os.path.join(args.folder, f'train_labels.npy'), train_labels)

np.save(os.path.join(args.folder, f'train_features_penult.npy'), train_features_penult)
np.save(os.path.join(args.folder, f'train_features_last.npy'), train_features_last)
print("Final train accuracy is {:.5f}".format(acc))

val_logits = np.concatenate(val_logits, 0)
val_probs = np.concatenate(val_probs, 0)
val_labels = np.concatenate(val_labels, 0)

val_features_penult = np.concatenate(val_features_penult, 0)
val_features_last = np.concatenate(val_features_last, 0)

acc = float(np.mean(val_labels == val_probs.argmax(1)))
# np.save(os.path.join(args.folder, f'val_logits.npy'), val_logits)
# np.save(os.path.join(args.folder, f'val_preds.npy'), val_probs)
# np.save(os.path.join(args.folder, f'val_labels.npy'), val_labels)

np.save(os.path.join(args.folder, f'val_features_penult.npy'), val_features_penult)
np.save(os.path.join(args.folder, f'val_features_last.npy'), val_features_last)
print("Final val accuracy is {:.5f}".format(acc))

