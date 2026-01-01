'''
Adapted from https://github.com/HobbitLong/SupContrast
'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class MySupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(MySupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, feature_i,feature_j, label_i=None, label_j=None):
        device = (torch.device('cuda')
                  if feature_i.is_cuda
                  else torch.device('cpu'))

        l = torch.cat((label_i, label_j), dim=0)
        l = l.contiguous().view(-1,1)
        mask = torch.eq(l, l.T).float().to(device)

        contrast_count = 2
        contrast_feature = torch.cat((feature_i,feature_j), dim=0)
        anchor_feature = contrast_feature

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(len(label_i) + len(label_j)).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask# For each row (xi), all elements except itself are set to 1. Used for computing the denominator

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask# denominator
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_sum=mask.sum(1)
        mask_zero_bool=mask_sum!=0
        own_positive=torch.masked_select(mask_sum,mask_zero_bool)
        own_positive_prob=log_prob[mask_zero_bool]
        mask=mask[mask_zero_bool]
        mean_log_prob_pos = (mask * own_positive_prob).sum(1) / own_positive

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


if __name__ == '__main__':
    a=torch.tensor([[1,2,3],
                    [2,3,4],
                    [3,4,5],
                    [4,5,6]])
    b=torch.tensor([[3,2,3],
                    [4,3,4],
                    [5,4,5],
                    [6,5,6]])
    a = F.normalize(a.float(), dim=1)
    b = F.normalize(b.float(), dim=1)
    label_a=torch.tensor([1,2,3,2])
    label_b=torch.tensor([1,2,3,2])
    supcon=MySupConLoss(temperature=0.5)
    loss=supcon.forward(a,b,label_a,label_b)
    print(loss)
