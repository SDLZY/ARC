import multiprocessing
import os
import logging

import shutil
import random
import json
import numpy as np
import pandas as pd
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch import optim
from tensorboardX import SummaryWriter
from allennlp.models import Model
from allennlp.training.optimizers import Optimizer

from utils.pytorch_misc import time_batch, clip_grad_norm, find_latest_checkpoint, move_optimizer_to_cuda
from models import distiller_zoo


class MyTrainer:
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

    def __init__(self, vocab, params, train_loader, val_loader, config):
        '''

        Args:
            params: model, optimizer, scheduler and trainer hyper-parameters.
            train_loader:
            val_loader:
            config: 存取方式，可视化
        '''
        # models在cpu上 通过GPU_NUMS判断是否用或用几块卡
        self._get_env_info()

        self.vocab = vocab
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.use_backbone = 'backbone' in params['models']
        self.use_qa2r = 'qa2r' in params['models']
        # 方便存取
        self.backbone = None
        if self.use_backbone:
            self.backbone, self.optimizer_b, self.scheduler_b = self.init_one_model('backbone', params)

        if self.use_qa2r:
            self.models, self.optimizers, self.schedulers = zip(
                self.init_one_model('qr2a', params),
                self.init_one_model('q2a', params),
                self.init_one_model('qa2r', params),
            )
        else:
            self.models, self.optimizers, self.schedulers = zip(
                self.init_one_model('qr2a', params),
                self.init_one_model('q2a', params),
            )

        self.model_num = len(self.models)

        # 附加组件，DML等
        # self.loss_module = DMLLoss(self.model_num, T=self.config.T, alpha=self.config.alpha)
        self.feat_use_gt = self.config.feat_use_gt
        self.loss_modules = []
        if self.config.feat_distiller == 'crd':
            self.loss_modules.append(CRDLoss(self.config))
        if self.config.afd:
            self.loss_modules.append(AFDLoss(self.models, lr=self.config.afd_lr, weight_decay=self.config.afd_wd))
        if self.config.afd3:
            # self.loss_modules.append(AFDLoss3(self.models, lr=self.config.afd_lr, weight_decay=self.config.afd_wd))
            self.loss_modules.append(AFDLoss3(self.model_num, weight=self.config.gan_w, lr=self.config.lr_d))
        if self.config.dml:
            self.loss_modules.append(DMLLoss(self.model_num, T=self.config.T, alpha=self.config.alpha))

        self.folder = os.path.join('saves', config.folder)
        self.writer_path = os.path.join('runs', config.folder)
        self.writer = None
        self.logger = logging.getLogger(__name__)

        self.start_epoch = 0
        self.num_batches = 0
        self.train_stats = []
        self.val_stats = []
        self.model_stats = []
        self.feat_tracked = [[], []]

    def train(self):
        if not self.pre_process():
            self.logger.info("[*] Error when pre-processing!")
            return

        for epoch in range(self.start_epoch, self.params['trainer']['num_epochs']):
            self.train_one_epoch(epoch, store_stat=True)
            self.validate(epoch, store_stat=True)

            self.update(epoch)
            self.log(epoch, write=True, hist=True)

            if self.check_finished():
                self.save_checkpoint(epoch, is_best=True)
                break
            self.save_checkpoint(epoch)
        self.post_process(epoch)

    def train_one_epoch(self, epoch, store_stat=False) -> pd.DataFrame:
        '''
        训练所有模型一个epoch，迭代优化器和scheduler
        Args:
            epoch: 当前epoch
            store_stat: 是否将训练结果储存在self.*_stats中
        Returns:
            训练结果字典(loss,acc,...)
        '''
        if self.use_backbone:
            self.backbone.train()
        for model in self.models:
            model.train()
        for module in self.loss_modules:
            module.train()
        results = []
        norms = [list() for _ in range(self.model_num)]
        with tqdm(self.train_loader, total=len(self.train_loader.dataset), ncols=120) as pbar:
            for b, (time_per_batch, batches) in enumerate(
                    time_batch(self.train_loader, reset_every=self.ARGS_RESET_EVERY)):
                self.num_batches += 1
                if self.NUM_GPUS > 0:
                    batches = self._to_gpu(batches)
                if self.use_backbone:
                    batches = [self.backbone(batch) for batch in batches]
                batch_size = batches[0]['label'].shape[0]
                answer_labels = batches[0]['label']
                outputs = []
                res = dict()
                for i in range(self.model_num):
                    self.schedulers[i].step_batch(self.num_batches)
                    _output = self.models[i](**batches[i])
                    _output['id'] = torch.tensor([batches[i]['metadata'][j]['ind'] for j in range(batch_size)], device=_output['feats'].device)
                    if self.feat_use_gt:
                        _output['feats'] = _output['feats'][torch.arange(batch_size), answer_labels]
                    outputs.append(_output)

                    if self.NUM_GPUS > 1:
                        acc = self.models[i].module.get_metrics(reset=True)['accuracy']
                    else:
                        acc = self.models[i].get_metrics(reset=True)['accuracy']
                    res[f'M{i}_acc'] = acc

                loss = 0
                for module in self.loss_modules:
                    _loss, _res = module(outputs, is_train=True)
                    loss += _loss
                    res.update(_res)

                for i in range(self.model_num):
                    self.optimizers[i].zero_grad()
                loss.backward()
                for i in range(self.model_num):
                    norms[i].append(
                        clip_grad_norm(self.models[i].named_parameters(), max_norm=self.params['trainer']['grad_norm'],
                                       clip=True, verbose=False))
                    self.optimizers[i].step()

                res['loss'] = loss.item() if isinstance(loss, torch.Tensor) else loss
                res = pd.Series(res)
                if self.config.afd3:
                    self.display_pbar(pbar, epoch, res, keys=['M0_acc', 'M1_acc', 'D_acc0', 'D_acc1'])
                elif self.config.feat_distiller == 'crd':
                    self.display_pbar(pbar, epoch, res, keys=['M0_acc', 'M1_acc', 'loss_crd'])
                else:
                    self.display_pbar(pbar, epoch, res, keys=['M0_acc', 'M1_acc'])
                pbar.update(batch_size)
                results.append(res)
        results = sum(results) / len(results)
        results['epoch'] = epoch
        norms = pd.DataFrame([sum(norm_list) / len(norm_list) for norm_list in norms])
        if store_stat:
            self.train_stats.append(results)
            self.model_stats.append(norms)
        return results

    def validate(self, epoch, store_stat=False) -> pd.DataFrame:
        """验证，存储验证集的模型效果"""
        if self.use_backbone:
            self.backbone.eval()
        for model in self.models:
            model.eval()
        for module in self.loss_modules:
            module.eval()
        results = []
        feat_tracked =[[], []]
        with tqdm(self.val_loader, total=len(self.val_loader.dataset), ncols=120) as pbar:
            for b, (time_per_batch, batches) in enumerate(time_batch(self.val_loader)):
                with torch.no_grad():
                    if self.NUM_GPUS > 0:
                        batches = self._to_gpu(batches)
                    if self.use_backbone:
                        batches = [self.backbone(batch) for batch in batches]
                    batch_size = batches[0]['label'].shape[0]
                    answer_labels = batches[0]['label']

                    outputs = []
                    res = dict()
                    for i in range(self.model_num):
                        _output = self.models[i](**batches[i])
                        torch.tensor([batches[i]['metadata'][j]['ind'] for j in range(batch_size)], device=_output['feats'].device)
                        if self.feat_use_gt:
                            _output['feats'] = _output['feats'][torch.arange(batch_size), answer_labels]
                        outputs.append(_output)
                        feat_tracked[i].append(_output['feats'][random.randint(0, batch_size-1)].flatten().data.cpu().numpy())
                        if self.NUM_GPUS > 1:
                            acc = self.models[i].module.get_metrics(reset=True)['accuracy']
                        else:
                            acc = self.models[i].get_metrics(reset=True)['accuracy']
                        res[f'M{i}_acc'] = acc

                    loss = 0
                    for module in self.loss_modules:
                        _loss, _res = module(outputs)
                        loss += _loss
                        res.update(_res)

                    res['loss'] = loss.item() if isinstance(loss, torch.Tensor) else loss
                    res = pd.Series(res)
                    if self.config.afd3:
                        self.display_pbar(pbar, epoch, res, keys=['M0_acc', 'M1_acc', 'D_acc0', 'D_acc1'])
                    else:
                        self.display_pbar(pbar, epoch, res, keys=['M0_acc', 'M1_acc'])
                    pbar.update(batch_size)
                    results.append(res)
        results = sum(results) / len(results)
        results['epoch'] = epoch
        if store_stat:
            self.val_stats.append(results)
            self.feat_tracked[0].extend(random.sample(feat_tracked[0], 1))
            self.feat_tracked[1].extend(random.sample(feat_tracked[1], 1))
        return results

    def save_checkpoint(self, epoch, is_best=False):
        """存储model和training state（optimizer，scheduler，训练过程中的统计结果）"""
        model_path = os.path.join(self.folder, f"model_state_epoch_{epoch}.th")
        training_state_path = os.path.join(self.folder, f"training_state_epoch_{epoch}.th")

        model_state = dict()
        training_state = {
            'epoch': epoch,
            'train_stats': self.train_stats,
            'val_stats': self.val_stats,
            'optimizers': dict(),
            'schedulers': dict(),
        }

        if self.use_backbone:
            model_state['backbone'] = self._get_model(self.backbone).state_dict()
            training_state['optimizers']['backbone'] = self.optimizer_b.state_dict()
            training_state['schedulers']['backbone'] = self.scheduler_b.lr_scheduler.state_dict()

        model_state['qr2a'] = self._get_model(self.models[0]).state_dict()
        training_state['optimizers']['qr2a'] = self.optimizers[0].state_dict()
        training_state['schedulers']['qr2a'] = self.schedulers[0].lr_scheduler.state_dict()

        model_state['q2a'] = self._get_model(self.models[1]).state_dict()
        training_state['optimizers']['q2a'] = self.optimizers[1].state_dict()
        training_state['schedulers']['q2a'] = self.schedulers[1].lr_scheduler.state_dict()

        if self.use_qa2r:
            model_state['qa2r'] = self._get_model(self.models[2]).state_dict()
            training_state['optimizers']['qa2r'] = self.optimizers[2].state_dict()
            training_state['schedulers']['qa2r'] = self.schedulers[2].lr_scheduler.state_dict()

        torch.save(model_state, model_path)
        torch.save(training_state, training_state_path)

        if is_best:
            shutil.copyfile(model_path, os.path.join(self.folder, "best.th"))
            shutil.copyfile(training_state_path, os.path.join(self.folder, "training_state_best.th"))

    def _get_model(self, model):
        """判断是否是DataParallel"""
        return model.module if isinstance(model, nn.DataParallel) else model

    def load_checkpoint(self, best=True):
        """读取checkpoint和training state"""
        if best:
            checkpoint = torch.load(os.path.join(self.folder, 'best.th'), map_location=torch.device('cpu'))
            training_state = torch.load(os.path.join(self.folder, 'training_state_best.th'), map_location=torch.device('cpu'))
        else:
            checkpoint, training_state = find_latest_checkpoint(self.folder)
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
            training_state = torch.load(training_state, map_location=torch.device('cpu'))

        if self.use_backbone:
            self.backbone.load_state_dict(checkpoint['backbone'])
            self.optimizer_b.load_state_dict(checkpoint['optimizers']['backbone'])
            self.scheduler_b.lr_scheduler.load_state_dict(checkpoint['scheduler']['backbone'])

        self.models[0].load_state_dict(checkpoint['qr2a'])
        self.optimizers[0].load_state_dict(checkpoint['optimizers']['qr2a'])
        self.schedulers[0].lr_scheduler.load_state_dict(checkpoint['scheduler']['qr2a'])

        self.models[1].load_state_dict(checkpoint['q2a'])
        self.optimizers[1].load_state_dict(checkpoint['optimizers']['q2a'])
        self.schedulers[1].lr_scheduler.load_state_dict(checkpoint['scheduler']['q2a'])

        if self.use_qa2r:
            self.models[2].load_state_dict(checkpoint['qa2r'])
            self.optimizers[2].load_state_dict(checkpoint['optimizers']['qa2r'])
            self.schedulers[2].lr_scheduler.load_state_dict(checkpoint['scheduler']['qa2r'])

        self.start_epoch = training_state['epoch']
        self.train_stats = training_state['train_stats']
        self.val_stats = training_state['val_stats']

        self.logger.info(f"[*] Loading model on epoch{self.start_epoch} in {self.folder}:")
        self.logger.info(f"\n[*] ----------train----------")
        self.logger.info('\n' + pd.DataFrame(self.train_stats).to_string(float_format=lambda x: f'{x:.3f}'))
        self.logger.info(f"\n[*] ----------validate----------")
        self.logger.info('\n' + pd.DataFrame(self.val_stats).to_string(float_format=lambda x: f'{x:.3f}'))

    def _get_env_info(self):
        """系统信息"""
        self.ARGS_RESET_EVERY = 100
        self.NUM_GPUS = torch.cuda.device_count()
        self.NUM_CPUS = multiprocessing.cpu_count()

    def _to_gpu(self, batch):
        """递归将Batch移到cuda上"""
        if self.NUM_GPUS > 1:
            return batch
        if isinstance(batch, list) or isinstance(batch, tuple):
            return [self._to_gpu(item) for item in batch]
        elif isinstance(batch, dict):
            for k, v in batch.items():
                if k != 'metadata':
                    batch[k] = self._to_gpu(v)
            return batch
        elif isinstance(batch, torch.Tensor):
            return batch.cuda(non_blocking=True)
        else:
            assert False

    def log(self, epoch, write=False, hist=False):
        """tensorboard数据可视化（在之前基础上加上了模型参数的可视化），标准输出stats"""
        if write:
            for key, value in self.train_stats[epoch].items():
                self.writer.add_scalar(f'{key}/train', value, epoch)

            for key, value in self.val_stats[epoch].items():
                self.writer.add_scalar(f'{key}/validate', value, epoch)

            for i, opt in enumerate(self.optimizers):
                self.writer.add_scalar(f'learning_rate/M{i}', opt.state_dict()['param_groups'][0]['lr'], epoch)

            if hist:
                for i, feats in enumerate(self.feat_tracked):
                    self.writer.add_histogram(f'feat/M{i}', feats[-1], epoch, 'doane')
                for i, model in enumerate(self.models):
                    for name, param in model.named_parameters():
                        if 'bn' not in name.lower():
                            self.writer.add_histogram(f'{name}/M{i}', param, epoch, 'doane')
                for i, module in enumerate(self.loss_modules):
                    for name, param in module.named_parameters():
                        if 'bn' not in name.lower():
                            self.writer.add_histogram(f'{name}/Lmodule{i}', param, epoch, 'doane')


        self.logger.info('\n[*] ---------norms---------')
        self.logger.info('\n' + self.model_stats[epoch].T.to_string(float_format=lambda x: f'{x:.4f}'))
        self.logger.info('\n[*] ---------train---------')
        self.logger.info('\n' + self.train_stats[epoch].to_string(float_format=lambda x: f'{x:.4f}'))
        self.logger.info('\n[*] ---------validate---------')
        self.logger.info('\n' + self.val_stats[epoch].to_string(float_format=lambda x: f'{x:.4f}'))
        self.logger.info(f'[*] Model {self.folder}')
        self.logger.info(f'[*] Config: {self.config}')

    def update(self, epoch):
        """根据验证结果迭代scheduler"""
        for i in range(self.model_num):
            self.schedulers[i].step(self.val_stats[-1][f'M{i}_acc'], epoch)
        if self.config.afd and epoch in [10, 20, 30]:
            for opt in self.loss_modules[0].D_optims:
                opt.param_groups[0]['lr'] *= 0.5
        # if self.config.afd3 and epoch in [5, 10, 15, 20, 25, 30, 35, 40]:
        #     self.loss_modules[0].weight *= 0.5

    def check_finished(self):
        """判断是否停止迭代"""
        val_metric_per_epoch = [self.get_metric(res) for res in self.val_stats]
        if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - self.params['trainer']['patience']):
            return True
        return False

    def pre_process(self):
        """初始化模型参数，将模型与优化器移到cuda上,初始化writer"""
        if self.config.resume:
            self.logger.info("[*] Resume last training state")
            self.load_checkpoint(best=False)
        else:
            if os.path.exists(self.folder):
                while True:
                    cover = input(f"[*] Find {self.folder} exists! Do you want to cover? [yes/no]")
                    if cover == "yes":
                        shutil.rmtree(self.folder)
                        shutil.rmtree(self.writer_path)
                        # shutil.copy2(self.params, self.folder)
                        break
                    elif cover == "no":
                        self.logger.info(f"[*] Please add -resume if you wanna to continue to train {self.folder}")
                        return False
                    else:
                        self.logger.info("[*] Please enter [yes/no]")
                        continue
            self.logger.info(f"[*] Making dir {self.folder}")
            os.makedirs(self.folder, exist_ok=True)
            self.logger.info(f"[*] Dumping config to {self.folder}/")
            with open(os.path.join(self.folder, 'config.json'), 'w') as fp:
                json.dump(self.config.__dict__, fp)
            self.logger.info(f"[*] Dumping params to {self.folder}/")
            with open(os.path.join(self.folder, 'params.json'), 'w') as fp:
                json.dump(self.params.as_dict(), fp)

        self.writer = SummaryWriter(self.writer_path)
        if self.NUM_GPUS > 1:
            if self.use_backbone:
                self.backbone = nn.DataParallel(self.backbone)
            self.models = [nn.DataParallel(model) for model in self.models]
            # self.loss_modules = [nn.DataParallel(module) for module in self.loss_modules]
        if self.NUM_GPUS > 0:
            if self.use_backbone:
                self.backbone = self.backbone.cuda()
                move_optimizer_to_cuda(self.optimizer_b)
            self.models = [model.cuda() for model in self.models]
            self.loss_modules = [module.cuda() for module in self.loss_modules]
            for opt in self.optimizers:
                move_optimizer_to_cuda(opt)
        return True

    def post_process(self, epoch):
        """输出最终结果"""
        self.writer.close()
        self.logger.info(f"[*] Stop at epoch {epoch}.")
        val_metric_per_epoch = [self.get_metric(res) for res in self.val_stats]
        best_epoch = int(np.argmax(val_metric_per_epoch))
        self.logger.info(f"[*] Best metric is {val_metric_per_epoch[best_epoch]: .4f} on epoch {best_epoch}:")
        self.logger.info('\n' + self.val_stats[best_epoch].to_string(float_format=lambda x: f'{x:.4f}'))

    def get_metric(self, result) -> float:
        """从val states中得到metric"""
        return result['M1_acc']

    def display_pbar(self, pbar, epoch, ser, keys=None):
        """tqdm输出"""
        msg = f"[*] epoch{epoch}|- "
        if keys is None:
            keys = ser.keys()
        for key in keys:
            msg += f'{key}:{ser[key]:.3f} '
        pbar.set_description(msg)

    def init_one_model(self, name, params):
        """
        param
            |- models
                |- qr2a, q2a, qa2r ..
            |-trainer
                |- optimizer_qr2a, optimizer_q2a, optimizer_qa2r...
                |- learning_rate_scheduler_qr2a, learning_rate_scheduler_q2a, learning_rate_scheduler_qa2r...
        """
        model = Model.from_params(vocab=self.vocab, params=params['models'][name])
        if hasattr(model, 'detector') and hasattr(model.detector, 'backbone'):
            for submodule in model.detector.backbone.modules():
                if isinstance(submodule, nn.BatchNorm2d):
                    submodule.track_running_stats = False
                for p in submodule.parameters():
                    p.requires_grad = False
        optimizer_params = params['trainer'][f'optimizer_{name}']
        optimizer = Optimizer.from_params([x for x in model.named_parameters() if x[1].requires_grad],
                                          optimizer_params)
        lr_scheduler_params = params['trainer'][f"learning_rate_scheduler_{name}"]
        scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        return model, optimizer, scheduler


class DMLLoss(nn.Module):
    def __init__(self, model_num, T, alpha):
        super().__init__()
        self.model_num = model_num
        self._kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.T = T
        self.alpha = alpha

    def forward(self, outputs, is_train=False):
        assert len(outputs) == self.model_num
        total_loss = 0
        results = {}
        for i in range(self.model_num):
            ce_loss = outputs[i]['loss'].mean()
            kl_loss = 0
            for j in range(self.model_num):
                if i == j:
                    continue
                kl_loss += self.soft_loss(outputs[j]['label_logits'].detach(), outputs[i]['label_logits'])
            loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss / (self.model_num - 1)
            total_loss += loss
            results.update({
                f"M{i}_loss": loss.item(),
                f"M{i}_ce_loss": ce_loss.item(),
                f"M{i}_kl_loss": kl_loss.item(),
            })
        return total_loss, results

    def soft_loss(self, logits_t, logits_s):
        loss = self._kl_loss(F.log_softmax(logits_s / self.T, dim=1), F.softmax(logits_t / self.T, dim=1)) * (
                self.T ** 2)
        return loss


class AFDLoss(nn.Module):
    """Generator用另外一个optimizer"""
    def __init__(self, models, lr, weight_decay):
        super().__init__()
        self.nums = len(models)
        self.D_nets = []
        self.D_optims = []
        self.G_optims = []
        for G_net in models:
            D_net = nn.Sequential(
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
            self.D_nets.append(D_net)
            # 模型的optimizer中有weight_decay
            self.G_optims.append(Adam(params=G_net.parameters(), lr=lr))
            # self.G_optims.append(Adam(params=G_net.parameters(), lr=lr, weight_decay=weight_decay))
            self.D_optims.append(Adam(params=D_net.parameters(), lr=lr, weight_decay=weight_decay))
        self.D_nets = nn.ModuleList(self.D_nets)

    def forward(self, outputs, is_train=False):
        assert len(outputs) == self.nums
        feats = [output['feats'] for output in outputs]
        # loss_D = 0
        results = {}
        for i in range(self.nums):
            feats_curr = feats[i]
            feats_next = feats[(i+1) % self.nums]
            true_probs = self.D_nets[i](feats_next.detach())
            fake_probs = self.D_nets[i](feats_curr.detach())
            D_acc_next = (true_probs > 0.5).float().mean().item()  # 判别下一个正确的比例
            D_acc_curr = (fake_probs > 0.5).float().mean().item()  # 判别当前正确的比例

            loss_D = ((1 - true_probs) ** 2 + fake_probs ** 2).mean()
            if is_train:
                self.D_optims[i].zero_grad()
                loss_D.backward()
                self.D_optims[i].step()

            loss_G = ((1 - self.D_nets[i](feats_curr)) ** 2).mean()
            if is_train:
                self.G_optims[i].zero_grad()
                loss_G.backward(retain_graph=True)
                self.G_optims[i].step()

            results.update({
                f'M{i}_loss_D': loss_D.item(),
                f'M{i}_loss_G': loss_G.item(),
                f'M{i}_D_acc_next': D_acc_next,
                f'M{i}_D_acc_curr': D_acc_curr,
            })
        return 0, results


class AFDLoss2(nn.Module):
    """Generator仍使用模型的optimizer"""
    def __init__(self, model_num, weight, lr=2e-4, weight_decay=1e-4):
        super().__init__()
        self.model_num = model_num
        self.D_nets = []
        self.D_optims = []
        self.weight = weight
        for i in range(self.model_num):
            D_net = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
            self.D_nets.append(D_net)
            self.D_optims.append(Adam(params=D_net.parameters(), lr=lr, weight_decay=weight_decay))
        self.D_nets = nn.ModuleList(self.D_nets)

    def forward(self, outputs, is_train=False):
        assert len(outputs) == self.model_num
        feats = [output['feats'] for output in outputs]
        # loss_D = 0
        loss_G = 0
        results = {}
        for i in range(self.model_num):
            feats_curr = feats[i]
            feats_next = feats[(i+1) % self.model_num]
            true_probs = self.D_nets[i](feats_next.detach())
            fake_probs = self.D_nets[i](feats_curr.detach())
            D_acc_next = (true_probs > 0.5).float().mean().item()  # 判别下一个正确的比例
            D_acc_curr = (fake_probs > 0.5).float().mean().item()  # 判别当前正确的比例

            loss_D = ((1 - true_probs) ** 2 + fake_probs ** 2).mean()
            if is_train:
                self.D_optims[i].zero_grad()
                loss_D.backward()
                self.D_optims[i].step()

            _loss_G = ((1 - self.D_nets[i](feats_curr)) ** 2).mean()
            loss_G += self.weight * _loss_G

            results.update({
                f'M{i}_loss_D': loss_D.item(),
                f'M{i}_loss_G': _loss_G.item(),
                f'M{i}_D_acc_next': D_acc_next,
                f'M{i}_D_acc_curr': D_acc_curr,
            })
        return loss_G, results


class AFDLoss3(nn.Module):
    def __init__(self, model_num, weight, lr, weight_decay=1e-4):
        super().__init__()
        self.model_num = model_num
        self.weight = weight
        self.D_net = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.model_num),
            nn.Sigmoid()
        )
        # self.D_optim = optim.RMSprop(params=self.D_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.D_optim = optim.SGD(params=self.D_net.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, outputs, is_train=False):
        assert len(outputs) == self.model_num
        feats = [output['feats'] for output in outputs]
        results = {}
        probs0 = self.D_net(feats[0].detach())
        probs1 = self.D_net(feats[1].detach())

        D_acc0 = (probs0 > 0.5).float().mean().item()
        D_acc1 = (probs1 > 0.5).float().mean().item()

        # eps = 0.1
        loss_D = ((1 - probs0) ** 2 + probs1 ** 2).mean()
        if is_train:
            self.D_optim.zero_grad()
            loss_D.backward()
            self.D_optim.step()

        loss_G0 = (self.D_net(feats[0]) ** 2).mean()
        loss_G1 = ((1 - self.D_net(feats[1])) ** 2).mean()
        # loss_G = loss_G1 * self.weight
        loss_G = (loss_G0 + loss_G1) * 0.5 * self.weight

        results.update({
            f'loss_D': loss_D.item(),
            f'loss_G0': loss_G0.item(),
            f'loss_G1': loss_G1.item(),
            f'D_acc0': D_acc0,
            f'D_acc1': D_acc1,
        })
        return loss_G, results


class CRDLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.criterion = distiller_zoo.CRDLoss(opt)

    def forward(self, outputs, is_train=False):
        if is_train:
            f_t = outputs[0]['feats']
            f_s = outputs[1]['feats']
            idx = outputs[0]['id']
            loss_crd = self.criterion(f_s, f_t, idx)
            return loss_crd * 0.1, {'loss_crd': loss_crd.item()}
        else:
            return .0, {}


class DistllerZooLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()

    def forward(self, outputs, is_train=False):
        pass