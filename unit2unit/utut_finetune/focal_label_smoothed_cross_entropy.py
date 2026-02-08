# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)


@dataclass
class FocalLabelSmoothedCrossEntropyCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    focal_gamma: float = field(
        default=2.0,
        metadata={
            "help": "Focal loss gamma parameter. 0 disables focal loss "
            "(reverts to standard label-smoothed cross-entropy). "
            "Higher values more aggressively down-weight easy examples."
        },
    )


def focal_label_smoothed_nll_loss(
    lprobs, target, epsilon, gamma, ignore_index=None, reduce=True
):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    eps_i = epsilon / (lprobs.size(-1) - 1)
    # Per-token label-smoothed loss (before focal modulation)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss

    # Apply focal modulation: down-weight confident (easy) predictions
    if gamma > 0:
        with torch.no_grad():
            p_t = lprobs.gather(dim=-1, index=target).exp()
            if ignore_index is not None:
                p_t.masked_fill_(pad_mask, 1.0)
            focal_weight = (1.0 - p_t) ** gamma
        loss = focal_weight * loss

    if reduce:
        loss = loss.sum()
        nll_loss = nll_loss.sum()

    return loss, nll_loss


@register_criterion(
    "focal_label_smoothed_cross_entropy",
    dataclass=FocalLabelSmoothedCrossEntropyCriterionConfig,
)
class FocalLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        focal_gamma=2.0,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=ignore_prefix_size,
            report_accuracy=report_accuracy,
        )
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = focal_label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            self.focal_gamma,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss
