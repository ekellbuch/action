from ml_collections import config_dict
import torch
from torch import nn
import numpy as np
from action.modules.module import BaseClassifierModule
from action.models import losses
from action.models.base import all_classifiers
from torchmetrics import Accuracy, R2Score, AveragePrecision, Precision, Recall, F1Score, MeanSquaredError

from omegaconf import OmegaConf

class SegmenterModule(BaseClassifierModule):
  def __init__(self, hparams):
    super().__init__(hparams)

    self.test_stage_name = "epoch/test_"

    self.ignore_index = hparams.get('ignore_class', 0)
    weight = hparams.get('class_weights', None)
    if weight is not None:
      weight = torch.tensor(weight, dtype=torch.float32)
    
    self.lambda_weak = torch.tensor(hparams.get('lambda_weak', 0))
    self.lambda_strong = torch.tensor(hparams.get('lambda_strong', 0))
    # next step prediction
    self.lambda_pred = torch.tensor(hparams.get('lambda_pred', 0))
    self.lambda_task = torch.tensor(hparams.get('lambda_task', 0))
    self.lambda_recon = torch.tensor(hparams.get('lambda_recon', 0))
    self.kl_weight = torch.tensor(hparams.get('kl_weight', 1))

    self.model = all_classifiers[self.hparams.classifier](self.hparams.classifier_cfg)

    if self.lambda_strong > 0:
      self.criterion = torch.nn.CrossEntropyLoss(
              weight=weight, ignore_index=self.ignore_index, reduction='mean')

      self.f1_score = F1Score(task='multiclass',
                              num_classes=self.hparams.classifier_cfg.num_classes,
                              average=None,
                              ignore_index=self.ignore_index)

      self.accuracy = Accuracy(task="multiclass",
                               num_classes=self.hparams.classifier_cfg.num_classes,
                               ignore_index=self.ignore_index)

      self.avg_precision = AveragePrecision(task='multiclass',
                                            num_classes=self.hparams.classifier_cfg.num_classes,
                                            average=None,
                                            # ignore_index=self.ignore_index # breaks if batch only has ignore_index
                                            )

    if self.lambda_weak > 0:
      # TODO: weak and strong classes maybe not be the same
      self.criterion_weak = torch.nn.CrossEntropyLoss(
        weight=weight, ignore_index=self.ignore_index, reduction='mean')

      # define hyperparams
      self.accuracy_weak = Accuracy(task="multiclass", num_classes=self.hparams.classifier_cfg.num_classes,
                                    ignore_index=self.ignore_index)

    if self.lambda_pred > 0:
      self.pred_loss = torch.nn.MSELoss(reduction='mean')
      self.mse_pred = MeanSquaredError()

    if self.lambda_recon > 0:
      self.recon_loss = torch.nn.MSELoss(reduction='mean')
      self.mse_recon = MeanSquaredError()

    if self.lambda_task > 0:
      self.task_loss = torch.nn.MSELoss(reduction='mean')
      self.mse_task = MeanSquaredError()
      self.r2score = R2Score()

    # index padding for convolutions
    self.pad = hparams.get('sequence_pad', 0)

    if self.lambda_strong > 0 or self.lambda_weak > 0:
      self.cls_idxs = list(np.setdiff1d(np.arange(self.hparams.classifier_cfg.num_classes), self.ignore_index))

  def _calc_agg_metrics(self, stage):
    # calculate relevant metrics:
    if self.lambda_strong > 0 or self.lambda_weak > 0:
      self.log(f"{stage}accuracy", self.accuracy.compute())
      aps_ = self.avg_precision.compute()
      f1score_ = self.f1_score.compute()
      for m_idx in self.cls_idxs:
        self.log(f"{stage}avg_precision/class_{m_idx}", aps_[m_idx])
        self.log(f"{stage}f1score/class_{m_idx}", f1score_[m_idx])
      aps_ = sum(aps_[self.cls_idxs])/ len(aps_[self.cls_idxs])
      f1score_ = sum(f1score_[self.cls_idxs])/ len(f1score_[self.cls_idxs])
      self.log(f"{stage}avg_precision", aps_)
      self.log(f"{stage}f1score", f1score_)

    if self.lambda_pred > 0:
      self.log(f"{stage}mse_pred", self.mse_pred.compute())
    if self.lambda_recon > 0:
      self.log(f"{stage}mse_recon", self.mse_recon.compute())
    if self.lambda_task > 0:
      self.log(f"{stage}mse_task", self.mse_task.compute())
      self.log(f"{stage}r2_task", self.r2score.compute())

  def _reset_agg_metrics(self):
    if self.lambda_strong > 0:
      self.accuracy.reset()
      self.avg_precision.reset()
      self.f1_score.reset()
    if self.lambda_weak > 0:
      self.accuracy_weak.reset()
    if self.lambda_pred > 0:
      self.mse_pred.reset()
    if self.lambda_recon > 0:
      self.mse_recon.reset()
    if self.lambda_task > 0:
      self.mse_task.reset()
      self.r2score.reset()

  def predict_step(self, batch, batch_idx, dataloader_idx = 0):
    """

    Parameters
    ----------
    data_generator : DataGenerator object
        data generator to serve data batches
    return_scores : True
        return scores before they've been passed through softmax
    remove_pad : True
        remove batch padding from model outputs before returning
    mode : str
        'eval' | 'train'

    Returns
    -------
    dict
        - 'predictions' (list of lists): first list is over datasets; second list is over
          batches in the dataset; each element is a numpy array of the label probability
          distribution
        - 'weak_labels' (list of lists): corresponding weak labels
        - 'labels' (list of lists): corresponding labels

    """
    # push data through model
    markers_wpad = batch['markers']
    outputs_dict = self.model(markers_wpad)
    pad = self.pad
    # remove padding from model output
    if pad > 0:
      for key, val in outputs_dict.items():
        outputs_dict[key] = val[:, pad:-pad, ...] if val is not None else None
    return outputs_dict

  def forward(self, batch):
      """Calculate negative log-likelihood loss for supervised models.

      The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
      requirements low; gradients are accumulated across all chunks before a gradient step is
      taken.

      Parameters
      ----------
      data : dict
          signals are of shape (n_sequences, sequence_length, n_channels)
      accumulate_grad : bool, optional
          accumulate gradient for training step

      Returns
      -------
      dict
          - 'loss' (float): total loss (negative log-like under specified noise dist)
          - other loss terms depending on model hyperparameters

      """
      # push data through model
      markers_wpad = batch['markers']
      outputs_dict = self.model(markers_wpad)
      pad = self.pad
      lambda_strong = self.lambda_strong
      lambda_weak = self.lambda_weak
      lambda_task = self.lambda_task
      lambda_pred = self.lambda_pred
      lambda_recon = self.lambda_recon
      kl_weight = self.kl_weight

      softmax = nn.Softmax(dim=-1)
      # remove padding from supplied data
      if lambda_strong > 0:
          if pad > 0:
              labels_strong = batch['labels_strong'][:, pad:-pad, ...]
          else:
              labels_strong = batch['labels_strong']
          # reshape to fit into class loss; needs to be (n_examples,)
          labels_strong = torch.flatten(labels_strong)
      else:
          labels_strong = None

      if lambda_weak > 0:
          if pad > 0:
              labels_weak = batch['labels_weak'][:, pad:-pad, ...]
          else:
              labels_weak = batch['labels_weak']
          # reshape to fit into class loss; needs to be (n_examples,)
          labels_weak = torch.flatten(labels_weak)
      else:
          labels_weak = None

      if lambda_task > 0:
          if pad > 0:
              tasks = batch['tasks'][:, pad:-pad, ...]
          else:
              tasks = batch['tasks']
      else:
          tasks = None

      # remove padding from model output
      if pad > 0:
          markers = markers_wpad[:, pad:-pad, ...]
          # remove padding from model output
          for key, val in outputs_dict.items():
              outputs_dict[key] = val[:, pad:-pad, ...] if val is not None else None
      else:
          markers = markers_wpad

      # initialize loss to zero
      loss = torch.tensor(0).float().to(markers.device)
      loss_dict = {}

      # ------------------------------------
      # compute loss on weak labels
      # ------------------------------------

      if lambda_weak > 0:
          # reshape predictions to fit into class loss; needs to be (n_examples, n_classes)
          labels_weak_reshape = torch.reshape(
              outputs_dict['labels_weak'], (-1, outputs_dict['labels_weak'].shape[-1]))
          # only compute loss where strong labels do not exist [indicated by a zero]
          if labels_strong is not None:
              idxs_ = labels_strong == 0
              if torch.sum(idxs_) > 0:
                  loss_weak = self.criterion_weak(labels_weak_reshape[idxs_], labels_weak[idxs_])
              else:
                  # if all timepoints are labeled, set weak loss to zero
                  #loss_weak = torch.tensor([0.], device=labels_strong.device)
                  loss_weak = torch.tensor(0., device=labels_strong.device)
          else:
              loss_weak = self.criterion_weak(labels_weak_reshape, labels_weak)

          loss += lambda_weak * loss_weak
          loss_dict['loss_weak'] = loss_weak
          # compute fraction correct on weak labels
          if 'labels' in outputs_dict.keys():
              fc = self.accuracy_weak(torch.argmax(outputs_dict['labels'],-1).flatten(), labels_weak)
              # log
              loss_dict['accuracy_weak'] = fc

      # ------------------------------------
      # compute loss on strong labels
      # ------------------------------------
      if lambda_strong > 0:
          # reshape predictions to fit into class loss; needs to be (n_examples, n_classes)
          labels_strong_reshape = torch.reshape(
              outputs_dict['labels'], (-1, outputs_dict['labels'].shape[-1]))

          if torch.all(labels_strong == self.ignore_index):
            # TODO: hacky fix for loss is nan when all labels are == ignore_index https://github.com/pytorch/pytorch/issues/70348
            loss_strong = torch.tensor(0., device=labels_strong_reshape.device)
          else:
            loss_strong = self.criterion(labels_strong_reshape, labels_strong)
          loss += lambda_strong * loss_strong
          # log
          loss_dict['loss_strong'] = loss_strong
          loss_dict['accuracy'] = self.accuracy(softmax(labels_strong_reshape), labels_strong)

          ap = self.avg_precision(softmax(labels_strong_reshape), labels_strong)
          f1score_ = self.f1_score(softmax(labels_strong_reshape), labels_strong)
          #precision = self.precision(labels_strong_reshape, labels_strong)
          #recall = self.recall(labels_strong_reshape, labels_strong)
          for m_idx in self.cls_idxs:
            loss_dict[f'average_precision/class_{m_idx}'] = ap[m_idx]
            loss_dict[f'f1score/class_{m_idx}'] = f1score_[m_idx]
            #loss_dict[f'precision/class_{m_idx}'] = precision[m_idx]
            #loss_dict[f'recall/class_{m_idx}'] = recall[m_idx]

          loss_dict['avg_precision'] = sum(ap[self.cls_idxs])/len(ap[self.cls_idxs])
          loss_dict['f1score'] = sum(f1score_[self.cls_idxs])/len(f1score_[self.cls_idxs])
          #loss_dict['precision'] = sum(precision) / len(precision)
          #loss_dict['recall'] = sum(recall) / len(recall)

      # ------------------------------------
      # compute loss on one-step predictions
      # ------------------------------------
      if lambda_pred > 0:
          loss_pred = self.pred_loss(markers[:, 1:], outputs_dict['prediction'][:, :-1])
          loss += lambda_pred * loss_pred
          # log
          loss_dict['loss_pred'] = loss_pred
          loss_dict['mse_pred'] = self.mse_pred(markers[:, 1:], outputs_dict['prediction'][:, :-1])

      # ------------------------------------
      # compute reconstruction loss
      # ------------------------------------
      if lambda_recon > 0:
          loss_recon = self.recon_loss(markers, outputs_dict['reconstruction'])
          loss += lambda_recon * loss_recon
          # log
          loss_dict['loss_recon'] = loss_recon
          loss_dict['mse_recon'] = self.mse_recon(markers, outputs_dict['reconstruction'])

      # ------------------------------------
      # compute regression loss on tasks
      # ------------------------------------
      if lambda_task > 0:
          loss_task = self.task_loss(tasks, outputs_dict['task_prediction'])
          loss += lambda_task * loss_task
          loss_dict['r2_task'] = self.r2score(outputs_dict['task_prediction'].view(-1), tasks.view(-1))
          # log
          loss_dict['loss_task'] = loss_task
          loss_dict['mse_task'] = self.mse_task(tasks, outputs_dict['task_prediction'])

      # ------------------------------------
      # compute kl divergence on appx posterior
      # ------------------------------------
      if self.hparams.get('variational', False):
          # multiply by 2 to take into account the fact that we're computing raw mse for decoding
          # and prediction rather than (1 / 2\sigma^2) * MSE
          loss_kl = 2.0 * losses.kl_div_to_std_normal(
              outputs_dict['latent_mean'], outputs_dict['latent_logvar'])
          loss += kl_weight * loss_kl
          # log
          loss_dict['kl_weight'] = kl_weight
          loss_dict['loss_kl'] = loss_kl

      loss_dict['loss'] = loss
      return loss_dict

  def configure_optimizers(self):
    # TODO: add scheduler?
    if self.hparams.optimizer_cfg.get("type", "adam"):
      params = OmegaConf.to_container(self.hparams.optimizer_cfg, resolve=True)
      params.pop('type')
      optimizer = torch.optim.Adam(self.model.get_parameters(), **params,
                       )
    if self.hparams.optimizer_cfg.type == 'adamw':
      optimizer = self._configure_adamw_optimizer()

    return optimizer

  def _configure_adamw_optimizer(self):
      """
      This long function is unfortunately doing something very simple and is being very defensive:
      We are separating out all parameters of the model into two buckets: those that will experience
      weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
      We are then returning the PyTorch optimizer object.
      """
      # separate out all parameters to those that will and won't experience regularizing weight decay
      decay = set()
      no_decay = set()
      whitelist_weight_modules = (torch.nn.Linear,)
      blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)
      for mn, m in self.model.named_modules():
        for pn, p in m.named_parameters():

          fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
          # random note: because named_modules and named_parameters are recursive
          # we will see the same tensors p many many times. but doing it this way
          # allows us to know which parent module any tensor p belongs to...
          if pn.endswith('bias'):
            # all biases will not be decayed
            no_decay.add(fpn)
          elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
            # weights of whitelist modules will be weight decayed
            decay.add(fpn)
          elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
            # weights of blacklist modules will NOT be weight decayed
            no_decay.add(fpn)

      # validate that we considered every parameter
      param_dict = {pn: p for pn, p in self.model.named_parameters()}
      inter_params = decay & no_decay
      union_params = decay | no_decay

      assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
      assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                  % (str(param_dict.keys() - union_params), )
      train_config = self.hparams.optimizer_cfg

      # create the pytorch optimizer object
      optim_groups = [
          {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
          {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
      ]

      optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr, betas=train_config.betas)
      return optimizer

  def training_step(self, batch, batch_idx):
    outputs = self.forward(batch)
    loss = outputs['loss']
    if torch.isnan(loss):
      raise ValueError("Loss is NaN")

    for output in outputs:
      self.log(f"batch/train_{output}", outputs[output])
    return loss

  def on_train_epoch_end(self):
    # TODO: check trainer epoch metric loader
    self._calc_agg_metrics(stage="epoch/train_")
    self._reset_agg_metrics()

  def on_validation_epoch_start(self):
    self._reset_agg_metrics()

  def on_test_epoch_start(self):
    self._reset_agg_metrics()

  def on_test_epoch_end(self):
    self._calc_agg_metrics(stage=self.test_stage_name)

  def on_validation_epoch_end(self):
    self._calc_agg_metrics(stage="epoch/val_")


class SegmenterBsoftModule(SegmenterModule):
  """
  Each model trains a binary classifier on a single class
  """
  def __init__(self, hparams):
    super().__init__(hparams)
    weight = hparams.get('class_weights', None)
    if weight is not None:
      weight = torch.tensor(weight, dtype=torch.float32)
    self.criterion = losses.BalancedSoftmax(sample_per_class=torch.tensor([self.hparams.samples_per_class]).squeeze(),
                                            weight=weight,
                                            ignore_index=self.ignore_index,
                                            reduction='mean')
class SegmenterBsoftWeakModule(SegmenterBsoftModule):
  """
  Each model trains a binary classifier on a single class
  """
  def __init__(self, hparams):
    super().__init__(hparams)
    weight = hparams.get('class_weights', None)
    if weight is not None:
      weight = torch.tensor(weight, dtype=torch.float32)
    self.criterion_weak = losses.BalancedSoftmax(sample_per_class=torch.tensor([self.hparams.samples_per_class]).squeeze(),
                                            weight=weight,
                                            ignore_index=self.ignore_index,
                                            reduction='mean')

class SegmenterBsoftSWeakModule(SegmenterBsoftModule):
  """
  Each model trains a binary classifier on a single class
  """
  def __init__(self, hparams):
    super().__init__(hparams)
    weight = hparams.get('class_weights', None)
    if weight is not None:
      weight = torch.tensor(weight, dtype=torch.float32)
    self.criterion_weak = losses.BalancedSoftmax(sample_per_class=torch.tensor([self.hparams.weak_samples_per_class]).squeeze(),
                                            weight=weight,
                                            ignore_index=self.ignore_index,
                                            reduction='mean')

all_segmenter_modules = {
  "segmenter_module": SegmenterModule,  # multi-class classifier module
  "segmenterBsoft_module": SegmenterBsoftModule,  # apply balanced softmax to strong labels
  "segmenterBsoftWeak_module": SegmenterBsoftWeakModule,  # apply balanced softmax to weak labels
  "segmenterBsoftSWeak_module": SegmenterBsoftSWeakModule,  # apply balanced softmax to weak labels
}

if __name__ == "__main__":
  cfg = config_dict.ConfigDict()
  cfg.module = "segmenter_module"
  cfg.classifier = "segmenter"
  cfg.classifier_cfg = config_dict.ConfigDict()
  cfg.classifier_cfg.backbone = "dtcn"
  cfg.classifier_cfg.input_size = 16
  cfg.classifier_cfg.num_classes = 6
  cfg.classifier_cfg.n_hid_layers = 2
  cfg.classifier_cfg.n_hid_units = 32
  cfg.classifier_cfg.n_lags = 4
  cfg.classifier_cfg.activation = "lrelu"
  cfg.classifier_cfg.sequence_pad = 16
  cfg.optimizer_cfg = config_dict.ConfigDict()
  cfg.optimizer_cfg.lr = 0.001

  # Load  a simple model
  module_args = {"hparams": cfg}
  model = all_segmenter_modules[cfg.module](**module_args)

  assert model is not None


