
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn
import torch

class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """

    def __init__(self, sample_per_class, **kwargs):
        super(BalancedSoftmax, self).__init__()
        self.sample_per_class = sample_per_class
        self.kwargs = kwargs

    def forward(self, input, label):
        return balanced_softmax_loss(label, input, self.sample_per_class,
                                     **self.kwargs)


def balanced_softmax_loss(labels, logits, sample_per_class, **kwargs):
  """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
  Args:
    labels: A int tensor of size [batch].
    logits: A float tensor of size [batch, no_of_classes].
    sample_per_class: A int tensor of size [no of classes].
    reduction: string. One of "none", "mean", "sum"
  Returns:
    loss: A float tensor. Balanced Softmax Loss.
  """
  spc = sample_per_class.type_as(logits)
  spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
  logits = logits + spc.log()
  loss = F.cross_entropy(input=logits, target=labels, **kwargs)
  return loss