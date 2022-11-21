from __future__ import print_function

import random

import torch
import torch.nn as nn
import numpy as np


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, args, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.args = args

    def forward(self, features, labels=None, mask=None, stop_grad=False, stop_grad_sd=-1.0):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        if stop_grad:
            anchor_dot_contrast_stpg = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T.detach()),
                self.temperature)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # For hard negatives, code adapted from HCL (https://github.com/joshr17/HCL)
        # =============== hard neg params =================
        tau_plus = self.args.tau_plus
        beta = self.args.beta
        temperature = 0.5
        N = (batch_size - 1) * contrast_count
        # =============== reweight neg =================
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        exp_logits_neg = exp_logits * (1 - mask) * logits_mask
        exp_logits_pos = exp_logits * mask
        pos = exp_logits_pos.sum(dim=1) / mask.sum(1)

        imp = (beta * (exp_logits_neg + 1e-9).log()).exp()
        reweight_logits_neg = (imp * exp_logits_neg) / imp.mean(dim=-1)
        Ng = (-tau_plus * N * pos + reweight_logits_neg.sum(dim=-1)) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
        log_prob = -torch.log(exp_logits / (pos + Ng))
        # ===============================================

        loss_square = mask * log_prob  # only positive positions have elements

        # mix_square = exp_logits
        mix_square = loss_square

        if stop_grad:
            logits_max_stpg, _ = torch.max(anchor_dot_contrast_stpg, dim=1, keepdim=True)
            logits_stpg = anchor_dot_contrast_stpg - logits_max_stpg.detach()
            # =============== reweight neg =================
            exp_logits_stpg = torch.exp(logits_stpg)
            exp_logits_neg_stpg = exp_logits_stpg * (1 - mask) * logits_mask
            exp_logits_pos_stpg = exp_logits_stpg * mask
            pos_stpg = exp_logits_pos_stpg.sum(dim=1) / mask.sum(1)

            imp_stpg = (beta * (exp_logits_neg_stpg + 1e-9).log()).exp()
            reweight_logits_neg_stpg = (imp_stpg * exp_logits_neg_stpg) / imp_stpg.mean(dim=-1)
            Ng_stpg = ((-tau_plus * N * pos_stpg + reweight_logits_neg_stpg.sum(dim=-1)) / (1 - tau_plus))

            # constrain (optional)
            Ng_stpg = torch.clamp(Ng_stpg, min=N * np.e ** (-1 / temperature))
            log_prob_stpg = -torch.log(exp_logits_stpg / (pos_stpg + Ng_stpg))
            # ===============================================
            tmp_square = mask * log_prob_stpg
        else:
            # tmp_square = exp_logits
            tmp_square = loss_square
        if stop_grad:
            ac_square = stop_grad_sd * tmp_square[batch_size:, 0:batch_size].T + (1 - stop_grad_sd) * tmp_square[
                                                                                                      0:batch_size,
                                                                                                      batch_size:]
        else:
            ac_square = tmp_square[0:batch_size, batch_size:]

        mix_square[0:batch_size, batch_size:] = ac_square * self.args.adv_weight
        mix_square[batch_size:, 0:batch_size] = ac_square.T * self.args.adv_weight

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = mix_square.sum(1) / mask.sum(1)

        # loss
        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ori_SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, args, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ori_SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.args = args

    def forward(self, features, labels=None, mask=None, stop_grad=False, stop_grad_sd=-1.0):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        if stop_grad:
            anchor_dot_contrast_stpg = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T.detach()),
                self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        loss_square = mask * log_prob  # only positive position have elements
        mix_square = loss_square
        if stop_grad:
            logits_max_stpg, _ = torch.max(anchor_dot_contrast_stpg, dim=1, keepdim=True)
            logits_stpg = anchor_dot_contrast_stpg - logits_max_stpg.detach()
            # compute log_prob
            exp_logits_stpg = torch.exp(logits_stpg) * logits_mask
            log_prob_stpg = logits_stpg - torch.log(exp_logits_stpg.sum(1, keepdim=True))
            loss_square_stpg = mask * log_prob_stpg
            tmp_square = loss_square_stpg
        else:
            tmp_square = loss_square
        if stop_grad:
            ac_square = stop_grad_sd * tmp_square[batch_size:, 0:batch_size].T + (1 - stop_grad_sd) * tmp_square[
                                                                                                      0:batch_size,
                                                                                                      batch_size:]
        else:
            ac_square = tmp_square[0:batch_size, batch_size:]

        mix_square[0:batch_size, batch_size:] = ac_square * self.args.adv_weight
        mix_square[batch_size:, 0:batch_size] = ac_square.T * self.args.adv_weight

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = mix_square.sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
