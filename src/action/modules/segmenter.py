from ml_collections import config_dict
import torch
from torch import nn
import numpy as np
from action.modules.module import ClassifierModule
from action.models import losses
from torchmetrics import Accuracy, R2Score, AveragePrecision, Precision, Recall


class SegmenterModule(ClassifierModule):
  def __init__(self, hparams):
    super().__init__(hparams)
    self.ignore_index = hparams.get('ignore_class', 0)

    weight = hparams.get('class_weights', None)
    if weight is not None:
      weight = torch.tensor(weight, dtype=torch.float32)
    self.criterion = torch.nn.CrossEntropyLoss(
            weight=weight, ignore_index=self.ignore_index, reduction='mean')
    self.criterion_weak = torch.nn.CrossEntropyLoss(
      weight=weight, ignore_index=self.ignore_index, reduction='mean')
    self.pred_loss = torch.nn.MSELoss(reduction='mean')
    self.task_loss = torch.nn.MSELoss(reduction='mean')

    # define hyperparams
    self.accuracy_weak = Accuracy(task="multiclass", num_classes=self.hparams.classifier_cfg.num_classes)
    self.r2score = R2Score()

    self.lambda_weak = torch.tensor(hparams.get('lambda_weak', 0))
    self.lambda_strong = torch.tensor(hparams.get('lambda_strong', 0))
    self.lambda_pred = torch.tensor(hparams.get('lambda_pred', 0))
    self.lambda_task = torch.tensor(hparams.get('lambda_task', 0))
    self.kl_weight = torch.tensor(hparams.get('kl_weight', 1))
    # index padding for convolutions
    self.pad = hparams.get('sequence_pad', 0)
    self.cls_idxs = list(np.setdiff1d(np.arange(self.hparams.classifier_cfg.num_classes),self.ignore_index))

    self.accuracy = Accuracy(task="multiclass", num_classes=self.hparams.classifier_cfg.num_classes)
    self.avg_precision = AveragePrecision(task='multiclass',
                                          num_classes=self.hparams.classifier_cfg.num_classes,
                                          average=None,
                                          ignore_index=self.ignore_index)

    #self.precision = Precision(task="multiclass",
    #                           num_classes=self.hparams.classifier_cfg.num_classes,
    #                           average=None,
    #                           ignore_index=self.ignore_index)

    #self.recall = Recall(task="multiclass",
    #                           num_classes=self.hparams.classifier_cfg.num_classes,
    #                           average=None,
    #                          ignore_index=self.ignore_index)


  def predict_step(self, batch, batch_idx, dataloader_idx=None):
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
      kl_weight = self.kl_weight

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
                  loss_weak = torch.tensor([0.], device=labels_strong.device)
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
          loss_strong = self.criterion(labels_strong_reshape, labels_strong)
          loss += lambda_strong * loss_strong
          # log
          loss_dict['loss_strong'] = loss_strong
          loss_dict['accuracy'] = self.accuracy(labels_strong_reshape, labels_strong)

          #
          ap = self.avg_precision(labels_strong_reshape, labels_strong)
          #precision = self.precision(labels_strong_reshape, labels_strong)
          #recall = self.recall(labels_strong_reshape, labels_strong)
          for m_idx in self.cls_idxs:
            loss_dict[f'average_precision/class_{m_idx}'] = ap[m_idx]
            #loss_dict[f'precision/class_{m_idx}'] = precision[m_idx]
            #loss_dict[f'recall/class_{m_idx}'] = recall[m_idx]

          loss_dict['avg_precision'] = sum(ap[self.cls_idxs])/len(ap[self.cls_idxs])
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

      # ------------------------------------
      # compute regression loss on tasks
      # ------------------------------------
      if lambda_task > 0:
          loss_task = self.task_loss(tasks, outputs_dict['task_prediction'])
          loss += lambda_task * loss_task
          loss_dict['task_r2'] = self.r2score(outputs_dict['task_prediction'], tasks)
          # log
          loss_dict['loss_task'] = loss_task

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
    return torch.optim.Adam(self.model.get_parameters(), **self.hparams.optimizer_cfg,
                            )


all_segmenter_modules = {
  "segmenter_module": SegmenterModule,  # multi-class classifier module
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


