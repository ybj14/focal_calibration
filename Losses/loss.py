'''
Implementation of the following loss functions:
1. Cross Entropy
2. Focal Loss
3. Cross Entropy + MMCE_weighted
4. Cross Entropy + MMCE
5. Brier Score
'''

from torch.nn import functional as F
from Losses.focal_loss import FocalLoss
from Losses.focal_loss_adaptive_gamma import FocalLossAdaptive
from Losses.mmce import MMCE, MMCE_weighted
from Losses.brier_score import BrierScore


def _encode_sub(onehot_labels):
    return onehot_labels - 1. / onehot_labels.shape[1]


def _encode_neg(logits):
    return 2 * onehot_labels - 1


def se_loss_s(logits, targets, **kwargs):
    onehot_labels = F.one_hot(targets, num_classes=10).float()
    return F.mse_loss(logits, _encode_sub(onehot_labels), reduction='mean').exp()-1


def mse_loss_s(logits, targets, **kwargs):
    onehot_labels = F.one_hot(targets, num_classes=10).float()
    return F.mse_loss(logits, _encode_sub(onehot_labels), reduction='sum')

def se_loss_n(logits, targets, **kwargs):
    onehot_labels = F.one_hot(targets, num_classes=10).float()
    return F.mse_loss(logits, _encode_neg(onehot_labels), reduction = 'mean').exp()-1


def mse_loss_n(logits, targets, **kwargs):
    onehot_labels = F.one_hot(targets, num_classes=10).float()
    return F.mse_loss(logits, _encode_neg(onehot_labels), reduction = 'sum')


def se_loss(logits, targets, **kwargs):
    return F.mse_loss(logits, F.one_hot(targets, num_classes = 10).float(), reduction='mean').exp()-1


def mse_loss(logits, targets, **kwargs):
    return F.mse_loss(logits, F.one_hot(targets, num_classes = 10).float(), reduction='sum')




def cross_entropy(logits, targets, **kwargs):
    return F.cross_entropy(logits, targets, reduction='sum')


def focal_loss(logits, targets, **kwargs):
    return FocalLoss(gamma=kwargs['gamma'])(logits, targets)


def focal_loss_adaptive(logits, targets, **kwargs):
    return FocalLossAdaptive(gamma=kwargs['gamma'],
                             device=kwargs['device'])(logits, targets)


def mmce(logits, targets, **kwargs):
    ce=F.cross_entropy(logits, targets)
    mmce=MMCE(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)


def mmce_weighted(logits, targets, **kwargs):
    ce=F.cross_entropy(logits, targets)
    mmce=MMCE_weighted(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)


def brier_score(logits, targets, **kwargs):
    return BrierScore()(logits, targets)
