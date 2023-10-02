"""
Training script. Should be pretty adaptable to whatever.
"""
import argparse
import os
import shutil

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
from tensorboardX import SummaryWriter

from dataloaders.vcr_kd import VCR, VCRLoader, get_batch_tasks2
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint

import logging
import pickle

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
print(f"Load params from {args.params}")

flag = os.path.exists(args.folder)
if not flag:
    raise Exception(f'folder {args.folder} not found')

train, val = VCR.splits(tasks=('0', '1', '2'), embs_to_load='bert_da',
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
            td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[
                k].cuda(
                non_blocking=True)
    return td


num_workers = 12
# num_workers = (4 * NUM_GPUS if NUM_CPUS == 32 else 2*NUM_GPUS)-1
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': args.batch_size // NUM_GPUS, 'num_gpus': NUM_GPUS, 'num_workers': num_workers}
train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)
# test_loader = VCRLoader.from_dataset(test, **loader_params)

ARGS_RESET_EVERY = 100
# print("Loading {} for {}".format(params['model'].get('type', 'WTF?'), 'rationales' if args.rationale else 'answer'), flush=True)
model = Model.from_params(vocab=train.vocab, params=params['model'])

use_epochs = range(0, 30, 10)
param_shapes = print_para(model)
from collections import defaultdict
losses_val = defaultdict(list)
for epoch in use_epochs:
    print(f"start eval at epoch {epoch}")
    losses_val_epoch = defaultdict(list)
    checkpoint = torch.load(os.path.join(args.folder, f'model_state_epoch_{epoch}.th'))
    model.load_state_dict(checkpoint)

    model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()
    model.eval()

    val_probs = {'0': [], '1': [], '2': []}
    val_labels = {'0': [], '1': [], '2': []}
    for b, (time_per_batch, batch) in enumerate(time_batch(tqdm(val_loader))):
        with torch.no_grad():
            batch = _to_gpu(batch)
            answer_label = batch['answer_label']
            rationale_label = batch['rationale_label']
            batch = get_batch_tasks2(batch, ('0', '1', '2'))
            output_dict = model(**batch)
            for task in ('0', '1', '2'):
                losses_val_epoch[task].extend(output_dict[f'loss_{task}'].detach().cpu().numpy().tolist())
                val_probs[task].append(output_dict[f'label_probs_{task}'].detach().cpu().numpy())
            val_labels['0'].append(answer_label.detach().cpu().numpy())
            val_labels['1'].append(answer_label.detach().cpu().numpy())
            val_labels['2'].append(rationale_label.detach().cpu().numpy())
                # val_labels[task].append(batch[f'input_dict_{task}']['label'].detach().cpu().numpy())
            # val_loss_sum += output_dict['loss'].mean().item() * batch['label'].shape[0]
    hits = {}
    val_acc = {}
    for task in ('0', '1', '2'):
        losses_val[task].append(losses_val_epoch)
        val_labels[task] = np.concatenate(val_labels[task], 0)
        val_probs[task] = np.concatenate(val_probs[task], 0)
        np.save(os.path.join(args.folder, f'valpreds_{task}.npy'), val_probs[task])
        hits[task] = val_labels[task] == val_probs[task].argmax(1)
        val_acc[task] = float(np.mean(hits[task]))
    val_qa2r_c_acc = float(np.mean(hits['2'][hits['1']]))
    val_qa2r_w_acc = float(np.mean(hits['2'][~hits['1']]))
    val_q2ar_acc = float(np.mean(hits['1'] & hits['2']))
    print(f"Final val q2a accuracy is {val_acc['1']:.4f} qa2r accuracy is {val_acc['2']:.4f} (c {val_qa2r_c_acc:.4f} "
          f"w {val_qa2r_w_acc:.4f}) q2ar accuracy is {val_q2ar_acc:.4f}")

with open(f'{args.folder}/val_losses.pkl', 'wb') as fp:
    pickle.dump(losses_val, fp)
