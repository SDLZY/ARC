from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn import functional as F


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='none')

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss


class HintLoss_2(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self, beta=1., detach_t=True, norm=False):
        super(HintLoss_2, self).__init__()
        self.crit = nn.MSELoss(reduction='mean')
        # self.beta = beta
        # self.detach_t = detach_t
        # self.norm = None
        # if norm:
        #     self.norm = Normalize(dim=1)

    def forward(self, feature_t, answer_label, feature_s, rationale_label):
        # assert '0' in outputs and '2' in outputs
        batch_size = answer_label.shape[0]
        feats_t = feature_t[torch.arange(batch_size), answer_label]
        feats_s = feature_s[torch.arange(batch_size), rationale_label]
        # if self.norm is not None:
        #     feats_t = self.norm(feats_t)
        #     feats_s = self.norm(feats_s)
        # if self.detach_t:
        #     feats_t = feats_t.detach()

        loss = self.crit(feats_t, feats_s)
        return loss


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, dim, power=2):
        super(Normalize, self).__init__()
        self.power = power
        self.dim = dim

    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class InfoNCELoss_2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, is_train=False):
        assert '0' in outputs and '2' in outputs
        answer_label = outputs['0']['label']
        rationale_label = outputs['2']['label']
        batch_size = answer_label.shape[0]

        feats_t = outputs['0']['feats']
        feats_s = outputs['2']['feats']

        # 是否detach??
        logits_s = torch.bmm(feats_t, feats_s[torch.arange(batch_size), rationale_label].unsqueeze(-1)).squeeze(-1)
        logits_t = torch.bmm(feats_s, feats_t[torch.arange(batch_size), answer_label].unsqueeze(-1)).squeeze(-1)

        loss_s = self.ce(logits_s/self.opt.nce_t, answer_label)
        loss_t = self.ce(logits_t/self.opt.nce_t, rationale_label)

        p_s = F.softmax(logits_s/self.opt.nce_t, dim=-1)
        pmax_s = p_s.max(-1)[0].mean()
        acc_s = (p_s.argmax(-1) == answer_label).float().mean()

        loss = loss_s + loss_t
        loss = loss[None]
        return loss * self.opt.beta_2, {'nce_2': loss.item(), 'ncep_2': pmax_s.item(), 'nceacc_2': acc_s.item()}

class InfoNCEEmbLoss_2(nn.Module):
    def __init__(self, temp):
        super().__init__()
        # self.opt = opt
        self.ce = nn.CrossEntropyLoss()
        self.embeds = nn.ModuleDict({
            '0': Embed(1536, 1536),
            '2': Embed(1536, 1536)
        })
        self.T = temp
        # self.optimizer = torch.optim.SGD(self.embeds.parameters(), lr=2e-4, weight_decay=1e-4)
        # self.optimizer = torch.optim.Adam(self.embeds.parameters(), lr=2e-4, weight_decay=1e-4)

    def forward(self, features_penult_qr2a, answer_label, features_penult_qa2r, rationale_label, is_train=False):
        # assert '0' in outputs and '2' in outputs
        # answer_label = outputs['0']['label']
        # rationale_label = outputs['2']['label']
        batch_size = answer_label.shape[0]

        # feats_t = outputs['0']['feats']
        # feats_s = outputs['2']['feats']

        feats_t = self.embeds['0'](features_penult_qr2a)
        feats_s = self.embeds['2'](features_penult_qa2r)
        # 是否detach??
        logits_s = torch.bmm(feats_t, feats_s[torch.arange(batch_size), rationale_label].unsqueeze(-1).detach()).squeeze(-1)
        logits_t = torch.bmm(feats_s, feats_t[torch.arange(batch_size), answer_label].unsqueeze(-1).detach()).squeeze(-1)
        loss_s = self.ce(logits_s/self.T, answer_label)
        loss_t = self.ce(logits_t/self.T, rationale_label)

        p_s = F.softmax(logits_s/self.T, dim=-1)
        pmax_s = p_s.max(-1)[0].mean()
        acc_s = (p_s.argmax(-1) == answer_label).float().mean()

        p_t = F.softmax(logits_t/self.T, dim=-1)
        pmax_t = p_t.max(-1)[0].mean()
        acc_t = (p_t.argmax(-1) == rationale_label).float().mean()

        # loss = loss_t
        loss = loss_s + loss_t
        # loss = loss[None]
        # return loss
        return loss, {
            'nce_t': loss_t, 'ncep_t': pmax_t, 'nceacc_t': acc_t,
            'nce_s': loss_s, 'ncep_s': pmax_s, 'nceacc_s': acc_s
        }


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        # self.non_linear = nn.Sequential(
        #     nn.Linear(dim_in, dim_out),
        #     nn.ReLU(),
        #     nn.Linear(dim_out, dim_out),
        # )
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], 4, -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x