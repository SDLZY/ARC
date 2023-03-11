"""
Training script. Should be pretty adaptable to whatever.
"""
import argparse
import os
import sys
PYTHON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PYTHON_PATH)
import shutil
import inspect

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

from dataloaders.vcr_kd import VCR, VCRLoader, get_batch_tasks
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint

import logging
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.enabled=False
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

# This is needed to make the imports work
from allennlp.models import Model
from collections import defaultdict
from kd import my_model_components, model_kd_infonce

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
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)
parser.add_argument(
    '-plot',
    dest='plot',
    help='plot folder location',
    type=str,
)
parser.add_argument(
    '-batch_size',
    dest='batch_size',
    type=int,
    default=96
)
parser.add_argument(
    '-checkpoint',
    type=str,
    default=None
)

args = parser.parse_args()
alpha = 0.11
beta = 0.1
writer = SummaryWriter('runs/' + args.plot)

params = Params.from_file(args.params)
print(f"Load params from {args.params}")

flag = os.path.exists(args.folder)
if not flag:
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)
    start_epoch, val_metric_per_epoch = 0, []
    shutil.copy2(args.params, args.folder)
    shutil.copy2(inspect.getfile(eval('VCR')), args.folder)
logger = open('log.txt', mode='a', encoding='utf8')

TASKS = '1', '2'
train, val = VCR.splits(tasks=TASKS, embs_to_load='bert_da', logits_qr2a=True, features_penult=True,
                        only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True)
                        # only_use_relevant_dets=False
                        )
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

num_workers = (4 * NUM_GPUS if NUM_CPUS == 32 else 2*NUM_GPUS)-1
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': args.batch_size // NUM_GPUS, 'num_gpus': NUM_GPUS, 'num_workers': num_workers}
train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)
# test_loader = VCRLoader.from_dataset(test, **loader_params)

ARGS_RESET_EVERY = 100
# print("Loading {} for {}".format(params['model'].get('type', 'WTF?'), 'rationales' if args.rationale else 'answer'), flush=True)
model = Model.from_params(vocab=train.vocab, params=params['model'])

cwd = os.getcwd()
for module in model.modules():
    try:
        file_path = inspect.getfile(type(module))
        if cwd in file_path:
            shutil.copy(file_path, args.folder)
    except Exception:
        print(f'{module} file cannot save.')

if args.checkpoint is not None:
    model.load_state_dict(torch.load(args.checkpoint))
for submodule in model.backbone.detector.backbone.modules():
    if isinstance(submodule, BatchNorm2d):
        submodule.track_running_stats = False
    for p in submodule.parameters():
        p.requires_grad = False

model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()
optimizer = Optimizer.from_params([x for x in model.named_parameters() if x[1].requires_grad],
                                  params['trainer']['optimizer'])

lr_scheduler_params = params['trainer'].pop("learning_rate_scheduler", None)
scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params) if lr_scheduler_params else None

if flag:
    print("Found folder! restoring", flush=True)
    start_epoch, val_metric_per_epoch = restore_checkpoint(model, optimizer, serialization_dir=args.folder,
                                                           learning_rate_scheduler=scheduler)

param_shapes = print_para(model)
num_batches = 0
for epoch_num in range(start_epoch, params['trainer']['num_epochs'] + start_epoch):
    train_stats = []
    train_results = []
    norms = []
    model.train()
    train_probs = defaultdict(list)
    train_labels = defaultdict(list)
    for b, (time_per_batch, batch) in enumerate(
            time_batch(train_loader if args.no_tqdm else tqdm(train_loader), reset_every=ARGS_RESET_EVERY)):
        batch = _to_gpu(batch)
        answer_label = batch['answer_label']
        rationale_label = batch['rationale_label']
        metadata = batch['metadata']
        batch = get_batch_tasks(batch, TASKS)
        optimizer.zero_grad()
        output_dict = model(**batch)
        loss_1 = output_dict['loss_1'].mean()
        loss_2 = output_dict['loss_2'].mean()
        loss_kd_1 = output_dict['loss_kd'].mean()
        loss_infonce = output_dict['loss_infonce'].mean()
        loss = loss_kd_1 * alpha + loss_1 * (1 - alpha) + loss_2 * (1 - beta) + loss_infonce * beta + output_dict['cnn_regularization_loss'].mean()

        loss.backward()
        for task in TASKS:
            train_probs[task].append(output_dict[f'label_probs_{task}'].detach().cpu().numpy())
            if task == '2':
                train_labels[task].append(rationale_label.detach().cpu().numpy())
            else:
                train_labels[task].append(answer_label.detach().cpu().numpy())

        correct = {}
        for task in TASKS:
            correct[task] = train_labels[task][-1] == train_probs[task][-1].argmax(-1)
        for i in range(answer_label.shape[0]):
            stats = {'index': metadata[i]['ind']}
            for task in TASKS:
                # stats[f'loss_{task}'] = output_dict[f'loss_{task}'][i].item()
                stats[f'correct_{task}'] = correct[task][i]
                stats[f'gt_prob_{task}'] = train_probs[task][-1][i][train_labels[task][-1][i]]
                stats[f'gt_logits_{task}'] = output_dict[f'label_logits_{task}'][i, train_labels[task][-1][i]].item()
            train_stats.append(stats)
        num_batches += 1
        if scheduler:
            scheduler.step_batch(num_batches)

        norms.append(
            clip_grad_norm(model.named_parameters(), max_norm=params['trainer']['grad_norm'], clip=True, verbose=False)
        )
        optimizer.step()
        train_results.append(pd.Series({
            'loss': loss.item(),
            'loss_1': loss_1.item(),
            'loss_2': loss_2.item(),
            'loss_kd_1': loss_kd_1.item(),
            'loss_infonce': loss_infonce.item(),
            'crl': output_dict['cnn_regularization_loss'].mean().item(),
            **(model.module if NUM_GPUS > 1 else model).get_metrics(
                reset=(b % ARGS_RESET_EVERY) == 0),
            'sec_per_batch': time_per_batch,
            'hr_per_epoch': len(train_loader) * time_per_batch / 3600,
        }))
        if b % ARGS_RESET_EVERY == 0 and b > 0:
            norms_df = pd.DataFrame(pd.DataFrame(norms[-ARGS_RESET_EVERY:]).mean(), columns=['norm']).join(
                param_shapes[['shape', 'size']]).sort_values('norm', ascending=False)

            print("e{:2d}b{:5d}/{:5d}. norms: \n{}\nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
                epoch_num, b, len(train_loader),
                norms_df.to_string(formatters={'norm': '{:.2f}'.format}),
                pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
            ), flush=True)
            logger.write("e{:2d}b{:5d}/{:5d}. norms: \n{}\nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
                epoch_num, b, len(train_loader),
                norms_df.to_string(formatters={'norm': '{:.2f}'.format}),
                pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
            ))

        # break

    epoch_results = pd.DataFrame(train_results).mean()
    epoch_stats = pd.DataFrame(train_stats)
    epoch_stats.to_csv(os.path.join(args.folder, f'train_stats_epoch_{epoch_num}.csv'))

    train_hits = {}
    train_acc = {}
    for task in TASKS:
        train_labels[task] = np.concatenate(train_labels[task], 0)
        train_probs[task] = np.concatenate(train_probs[task], 0)
        train_hits[task] = train_labels[task] == train_probs[task].argmax(1)
        train_acc[task] = float(np.mean(train_hits[task]))
    train_qa2r_c_acc = float(np.mean(train_hits['2'][train_hits['1']]))
    train_qa2r_w_acc = float(np.mean(train_hits['2'][~train_hits['1']]))
    epoch_results['qa2r_c'] = train_qa2r_c_acc
    epoch_results['qa2r_w'] = train_qa2r_w_acc
    # train_loss = epoch_stats['loss']
    # train_acc = epoch_stats['accuracy']

    # writer.add_scalar('loss_0/train', epoch_results['loss_0'], epoch_num)
    writer.add_scalar('loss_1/train', epoch_results['loss_1'], epoch_num)
    writer.add_scalar('loss_2/train', epoch_results['loss_2'], epoch_num)
    if 'loss_kd' in epoch_results:
        writer.add_scalar('loss_kd/train', epoch_results['loss_kd'], epoch_num)
    if 'loss_infonce' in epoch_results:
        writer.add_scalar('loss_infonce/train', epoch_results['loss_infonce'], epoch_num)
    writer.add_scalar('crl/train', epoch_results['crl'], epoch_num)
    # writer.add_scalar('accuracy_0/train', epoch_results['accuracy_0'], epoch_num)
    writer.add_scalar('accuracy_1/train', epoch_results['accuracy_1'], epoch_num)
    writer.add_scalar('accuracy_2/train', epoch_results['accuracy_2'], epoch_num)
    writer.add_scalar('accuracy_qa2r_c/train', train_qa2r_c_acc, epoch_num)
    writer.add_scalar('accuracy_qa2r_w/train', train_qa2r_w_acc, epoch_num)

    print("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, epoch_results))
    logger.write("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, epoch_results))

    val_probs = defaultdict(list)
    val_labels = defaultdict(list)
    val_stats = []
    model.eval()
    for b, (time_per_batch, batch) in enumerate(time_batch(tqdm(val_loader))):
        with torch.no_grad():
            batch = _to_gpu(batch)
            # answer_label = batch.pop('answer_label')
            # rationale_label = batch.pop('rationale_label')
            answer_label = batch['answer_label']
            rationale_label = batch['rationale_label']
            metadata = batch['metadata']
            batch = get_batch_tasks(batch, TASKS)
            output_dict = model(**batch)
            for task in TASKS:
                val_probs[task].append(output_dict[f'label_probs_{task}'].detach().cpu().numpy())
            # val_labels['0'].append(answer_label.detach().cpu().numpy())
            val_labels['1'].append(answer_label.detach().cpu().numpy())
            val_labels['2'].append(rationale_label.detach().cpu().numpy())
            correct = {}
            for task in TASKS:
                correct[task] = val_labels[task][-1] == val_probs[task][-1].argmax(-1)
            for i in range(answer_label.shape[0]):
                stats = {'index': metadata[i]['ind']}
                for task in TASKS:
                    # stats[f'loss_{task}'] = output_dict[f'loss_{task}'][i].item()
                    stats[f'correct_{task}'] = correct[task][i]
                    stats[f'gt_prob_{task}'] = val_probs[task][-1][i][val_labels[task][-1][i]]
                    stats[f'gt_logits_{task}'] = output_dict[f'label_logits_{task}'][i, val_labels[task][-1][i]].item()
                val_stats.append(stats)
            # val_loss_sum += output_dict['loss'].mean().item() * batch['label'].shape[0]
    epoch_stats = pd.DataFrame(val_stats)
    epoch_stats.to_csv(os.path.join(args.folder, f'val_stats_epoch_{epoch_num}.csv'))
    hits = {}
    val_acc = {}
    for task in TASKS:
        val_labels[task] = np.concatenate(val_labels[task], 0)
        val_probs[task] = np.concatenate(val_probs[task], 0)
        hits[task] = val_labels[task] == val_probs[task].argmax(1)
        val_acc[task] = float(np.mean(hits[task]))
    val_q2ar_acc = float(np.mean(hits['1'] & hits['2']))
    # val_loss_avg = val_loss_sum / val_labels.shape[0]

    val_metric_per_epoch.append({
        # "accuracy_0": val_acc['0'],
        "accuracy_1": val_acc['1'],
        "accuracy_2": val_acc['2'],
        "accuracy_q2ar": val_q2ar_acc,
    })
    if scheduler:
        scheduler.step(val_metric_per_epoch[-1]['accuracy_q2ar'], epoch_num)

    print(f"Val epoch {epoch_num}  q2a_acc {val_acc['1']:.4f} qa2r_acc {val_acc['2']:.4f} ",
          flush=True)
    logger.write(f"Val epoch {epoch_num}  q2a_acc {val_acc['1']:.4f} qa2r_acc {val_acc['2']:.4f} ")
    # writer.add_scalar('accuracy_0/validation', val_metric_per_epoch[-1]['accuracy_0'], epoch_num)
    writer.add_scalar('accuracy_1/validation', val_metric_per_epoch[-1]['accuracy_1'], epoch_num)
    writer.add_scalar('accuracy_2/validation', val_metric_per_epoch[-1]['accuracy_2'], epoch_num)
    writer.add_scalar('accuracy_q2ar/validation', val_metric_per_epoch[-1]['accuracy_q2ar'], epoch_num)

    history_metrics = list(map(lambda x: x["accuracy_q2ar"], val_metric_per_epoch))
    # history_metrics = list(map(lambda x: x["accuracy_q2a"]+x["accuracy_qa2r"], val_metric_per_epoch))
    if int(np.argmax(history_metrics)) < (len(val_metric_per_epoch) - 1 - params['trainer']['patience']):
        print("Stopping at epoch {:2d}".format(epoch_num))
        break
    save_checkpoint(model, optimizer, args.folder, epoch_num, val_metric_per_epoch,
                    is_best=int(np.argmax(history_metrics)) == (len(val_metric_per_epoch) - 1),
                    learning_rate_scheduler=scheduler)

print("STOPPING. now running the best model on the validation set", flush=True)
# Load best
restore_best_checkpoint(model, args.folder)
model.eval()
val_probs = defaultdict(list)
val_labels = defaultdict(list)
for b, (time_per_batch, batch) in enumerate(time_batch(tqdm(val_loader))):
    with torch.no_grad():
        batch = _to_gpu(batch)
        answer_label = batch['answer_label']
        rationale_label = batch['rationale_label']
        batch = get_batch_tasks(batch, TASKS)
        output_dict = model(**batch)
        for task in TASKS:
            val_probs[task].append(output_dict[f'label_probs_{task}'].detach().cpu().numpy())
        # val_labels['0'].append(answer_label.detach().cpu().numpy())
        val_labels['1'].append(answer_label.detach().cpu().numpy())
        val_labels['2'].append(rationale_label.detach().cpu().numpy())
hits = {}
val_acc = {}
for task in TASKS:
    val_labels[task] = np.concatenate(val_labels[task], 0)
    val_probs[task] = np.concatenate(val_probs[task], 0)
    np.save(os.path.join(args.folder, f'valpreds_{task}.npy'), val_probs[task])
    hits[task] = val_labels[task] == val_probs[task].argmax(1)
    val_acc[task] = float(np.mean(hits[task]))
val_q2ar_acc = float(np.mean(hits['1'] & hits['2']))
print(f"Final val q2a accuracy is {val_acc['1']:.4f} qa2r accuracy is {val_acc['2']:.4f}  q2ar accuracy is {val_q2ar_acc:.4f}")
logger.write(f"Final val q2a accuracy is {val_acc['1']:.4f} qa2r accuracy is {val_acc['2']:.4f}  q2ar accuracy is {val_q2ar_acc:.4f}")
