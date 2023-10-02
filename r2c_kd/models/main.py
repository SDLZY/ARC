import os
import sys
import argparse
import logging
from allennlp.common.util import prepare_global_logging
import random
import time
import numpy as np
import torch
import warnings

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-params', type=str, )
    parser.add_argument('-folder', type=str, )
    parser.add_argument('-resume', action='store_true')
    parser.add_argument('-afd', action='store_true')
    parser.add_argument('-afd3', action='store_true')
    parser.add_argument('-dml', action='store_true')
    parser.add_argument('-batch_size', type=int, default=96, )
    parser.add_argument('-alpha', type=float, )
    parser.add_argument('-T', type=float, )
    parser.add_argument('-afd_lr', type=float, default=None)
    parser.add_argument('-afd_wd', type=float, default=None)
    parser.add_argument('-gan_w', type=float, default=None)
    parser.add_argument('-lr_d', type=float, default=None)
    parser.add_argument('-feat_use_gt', action='store_true')

    parser.add_argument('-feat_distiller', type=str, default='hint', choices=['hint', 'crd'])
    parser.add_argument('-feat_dim', default=512, type=int, help='feature dimension')
    parser.add_argument('-nce_k', default=4096, type=int, help='number of negative samples for NCE')
    parser.add_argument('-nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('-nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    args = parser.parse_args()
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    logdir = os.path.join('logs', f'{args.folder}_{int(time.time())}')
    prepare_global_logging(logdir, False)

    from allennlp.common import Params
    from models.trainer import MyTrainer
    from dataloaders.vcr_q2a_qr2a_kd import VCR, VCRLoader

    params = Params.from_file(args.params)
    train, val = VCR.splits(mode='answer', embs_to_load='bert_da', only_use_relevant_dets=True)
    num_workers = 12

    NUM_GPUS = torch.cuda.device_count()
    loader_params = {'batch_size': args.batch_size // NUM_GPUS, 'num_gpus': NUM_GPUS, 'num_workers': num_workers}
    train_loader = VCRLoader.from_dataset(train, **loader_params)
    val_loader = VCRLoader.from_dataset(val, **loader_params)

    trainer = MyTrainer(
        vocab=train.vocab,
        params=params,
        train_loader=train_loader,
        val_loader=val_loader,
        config=args,
    )
    trainer.train()
