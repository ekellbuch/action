import pytorch_lightning as pl
import torch
from ml_collections import config_dict

from torchmetrics import AveragePrecision
from torchmetrics import Accuracy, Precision, Recall
from torchmetrics import F1Score as F1

import action
from action.models.base import all_classifiers
from action.models.losses import BalancedSoftmax


class BaseClassifierModule(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.hparams.update(hparams)

  def forward(self, batch):
    raise NotImplementedError

  def training_step(self, batch):
    raise NotImplementedError

  def validation_step(self, batch, batch_idx):
      outputs = self.forward(batch)

  def test_step(self, batch,batch_idx):
      outputs = self.forward(batch)



class ClassifierModule(BaseClassifierModule):
  """
  Train a model on a classifier
  data is batch has sequences batch x input_size
  """

  def __init__(self, hparams):
    super().__init__(hparams)
    self.criterion = torch.nn.CrossEntropyLoss()
    self.accuracy = Accuracy(task="multiclass", num_classes=self.hparams.classifier_cfg.num_classes)
    self.avg_precision = AveragePrecision(task='multiclass',
                                          num_classes=self.hparams.classifier_cfg.num_classes,
                                          average=None)
    self.model = all_classifiers[self.hparams.classifier](self.hparams.classifier_cfg)
    self.cls_idxs = range(self.hparams.classifier_cfg.num_classes)
    self.test_stage_name = "epoch/test_"

    #self.precision = Precision(task="multiclass",
    #                           num_classes=self.hparams.classifier_cfg.num_classes,
    #                           average=None)

    #self.recall = Recall(task="multiclass",
    #                           num_classes=self.hparams.classifier_cfg.num_classes,
    #                           average=None)
  def forward(self, batch):
    inputs, target = batch
    output = self.model(inputs)
    loss = self.criterion(output, target)
    accuracy = self.accuracy(output, target)
    ap = self.avg_precision(output, target)
    #precision = self.precision(output, target)
    #recall = self.recall(output, target)
    outputs = {}
    for m_idx in self.cls_idxs:
      outputs[f'avg_precision/class_{m_idx}'] = ap[m_idx]
      #outputs[f'precision/class_{m_idx}'] = precision[m_idx]
      #outputs[f'recall/class_{m_idx}'] = recall[m_idx]

    outputs['loss'] = loss
    outputs['accuracy'] = accuracy
    outputs['avg_precision'] = sum(ap[self.cls_idxs])/len(ap[self.cls_idxs])
    #outputs['precision'] = sum(precision)/len(precision)
    #outputs['recall'] = sum(recall)/len(recall)
    return outputs

  def predict_step(self, batch, batch_idx):
    inputs, target = batch
    output = self.model(inputs)
    return output

  def training_step(self, batch, batch_idx):
    outputs = self.forward(batch)
    loss = outputs['loss']
    if torch.isnan(loss):
      raise ValueError("Loss is NaN")

    for output in outputs:
      self.log(f"batch/train_{output}", outputs[output])
    return loss

  def on_train_epoch_end(self):
    #self._calc_agg_metrics(stage="epoch/train_")
    self._reset_agg_metrics()

  def on_validation_epoch_start(self):
    self._reset_agg_metrics()

  def on_test_epoch_start(self):
    self._reset_agg_metrics()

  def on_test_epoch_end(self):
    self._calc_agg_metrics(stage=self.test_stage_name)

  def on_validation_epoch_end(self):
    self._calc_agg_metrics(stage="epoch/val_")

  def _reset_agg_metrics(self):
    self.accuracy.reset()
    self.avg_precision.reset()

  def _calc_agg_metrics(self, stage):
    self.log(f"{stage}accuracy", self.accuracy.compute())
    aps_ = self.avg_precision.compute()
    #ps_ = self.precision.compute()
    #rec_ = self.recall.compute()
    for m_idx in self.cls_idxs:
      self.log(f"{stage}avg_precision/class_{m_idx}", aps_[m_idx])
      #self.log(f"{stage}precision/class_{m_idx}", ps_[m_idx])
      #self.log(f"{stage}recall/class_{m_idx}", rec_[m_idx])
    aps_ = sum(aps_[self.cls_idxs])/ len(aps_[self.cls_idxs])
    #ps_ = sum(ps_)/ len(ps_)
    #rec_ = sum(rec_)/ len(rec_)
    self.log(f"{stage}avg_precision", aps_)
    #self.log(f"{stage}precision", ps_)
    #self.log(f"{stage}recall", rec_)

    self._reset_agg_metrics()

  def configure_optimizers(self):

    return torch.optim.Adam(self.model.parameters(), **self.hparams.optimizer_cfg,
                            )

class ClassifierSeqModule(ClassifierModule):
  """
  Train a model on a classifier
  data is batch has sequences batch x seq_len x input_size
  classifier model

  """
  def __init__(self, hparams):
    super().__init__(hparams)

  def forward(self, batch):
    softmax = torch.nn.Softmax(dim=1)
    inputs = batch['markers']
    target = batch['labels_strong']
    inputs = inputs.reshape(-1, inputs.shape[-1])
    output = self.model(inputs)
    target = target.reshape(-1)
    loss = self.criterion(output, target)
    predictions = softmax(output)
    accuracy = self.accuracy(predictions, target)
    ap = self.avg_precision(predictions, target)
    #precision = self.precision(predictions, target)
    #recall = self.recall(predictions, target)

    outputs = {}
    for m_idx in self.cls_idxs:
      outputs[f'average_precision/class_{m_idx}'] = ap[m_idx]
      #outputs[f'precision/class_{m_idx}'] = precision[m_idx]
      #outputs[f'recall/class_{m_idx}'] = recall[m_idx]

    outputs['loss'] = loss
    outputs['accuracy'] = accuracy
    outputs['average_precision'] = sum(ap[self.cls_idxs])/len(ap[self.cls_idxs])
    #outputs['precision'] = sum(precision)/len(precision)
    #outputs['recall'] = sum(recall)/len(recall)
    return outputs

class ClassifierSeqModuleBS(ClassifierSeqModule):
  """
  Each model trains a binary classifier on a single class
  """
  def __init__(self, hparams):
    super().__init__(hparams)
    self.criterion = BalancedSoftmax(torch.tensor([self.hparams.samples_per_class]).squeeze())


base_modules = {
  "cls": ClassifierModule,  # multi-class classifier module
  "cls_seq": ClassifierSeqModule,  # multi-class classifier module applied to sequential data
  "cls_seq_bsoftmax": ClassifierSeqModuleBS, # multi-class classifier module applied to sequential data using balanced softmax loss
}


if __name__ == "__main__":
  cfg = config_dict.ConfigDict()
  cfg.module = "cls"
  cfg.classifier = "baseline"
  cfg.classifier_cfg = config_dict.ConfigDict()
  cfg.classifier_cfg.model_tier = "small"
  cfg.classifier_cfg.num_classes = 3
  cfg.classifier_cfg.input_size = 10
  cfg.optimizer_cfg = config_dict.ConfigDict()
  cfg.optimizer_cfg.lr = 0.001

  # Load  a simple model
  module_args = {"hparams": cfg}
  model = base_modules[cfg.module](**module_args)

  # Make a dummy dataset
  dataset = torch.utils.data.TensorDataset(torch.randn(10, 10), torch.randint(0, 3, (10,)))
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

  loss = model.training_step(next(iter(dataloader)), 0)
  assert not torch.isnan(loss)

  # Now onto the more complicated dataset:
  data_cfg = config_dict.ConfigDict()

  data_cfg.data_dir = "/data/ibl"
  data_cfg.batch_size = 32
  data_cfg.num_workers = 8
  data_cfg.input_type = "markers"
  #cfg.expt_ids = ["cortexlab_KS020_2020-02-06-001", "wittenlab_ibl_witten_26_2021-01-27-002"]
  data_cfg.expt_ids = ["cortexlab_KS020_2020-02-06-001"]
  data_cfg.sequence_length = 5
  data_cfg.lambda_strong = 1
  data_cfg.lambda_weak = 1
  # pad before and pad after

  import sys
  import os
  ACTION_DIR = action.__path__[0]

  CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

  sys.path.append("/Users/ekellbuch/Projects/segment/action/src/action/data")
  from datamodule import BaseModule
  ind_data = BaseModule(data_cfg)
  ind_data.prepare_data()
  train_dataset = ind_data.train_dataset
  print(train_dataset)
  train_dataloader = ind_data.train_dataloader()

  cfg.module = "cls_seq"
  module_args = {"hparams": cfg}
  model = all_modules[cfg.module](**module_args)


  batch = next(iter(train_dataloader))
  loss = model.training_step(batch, 0)

  # Let's only use the
