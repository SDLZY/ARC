from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn import functional as F

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


class InfoNCELoss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        # self.opt = opt
        self.ce = nn.CrossEntropyLoss()
        self.embeds = nn.ModuleDict({
            '0': Embed(1536, 1536),
            '2': Embed(1536, 1536)
        })
        self.T = temp

    def forward(self, features_penult_qr2a, answer_label, features_penult_qa2r, rationale_label, is_train=False):
        batch_size = answer_label.shape[0]

        feats_t = self.embeds['0'](features_penult_qr2a)
        feats_s = self.embeds['2'](features_penult_qa2r)

        logits_s = torch.bmm(feats_t, feats_s[torch.arange(batch_size), rationale_label].unsqueeze(-1).detach()).squeeze(-1)
        logits_t = torch.bmm(feats_s, feats_t[torch.arange(batch_size), answer_label].unsqueeze(-1).detach()).squeeze(-1)
        loss_s = self.ce(logits_s/self.T, answer_label)
        loss_t = self.ce(logits_t/self.T, rationale_label)
        loss = loss_s + loss_t
        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], 4, -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x