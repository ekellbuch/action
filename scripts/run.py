"""
Train a classifier on top of the features
"""
import hydra
from omegaconf import OmegaConf, open_dict
import os
from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import wandb

import datetime

import sys
ACTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ACTION_DIR + "/src")
from action.data.datamodule import all_datasets
from action.modules import all_modules
from action.callbacks import GradNormCallbackSplit
from action.data.dataloader import compute_sequence_pad

import torch
script_dir = os.path.abspath(os.path.dirname(__file__))

@hydra.main(config_path="script_configs", config_name="ibl_paw", version_base=None)
def main(args):
    train(args)
    return

# Define the model
def train(args):
  if args.seed is not None:
    seed_everything(args.seed)

  try:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        output_dir = hydra_cfg['runtime']['output_dir']
  except:
      pass

  # Set up logging.
  project_name = args.get('project_name', args.data_cfg.test_set)
  experiment_name = args.get('experiment_name', args.module_cfg.classifier)

  if args.trainer_cfg.fast_dev_run:
    logger = None
  else:
    if args.trainer_cfg.logger == "wandb":
        if args.trainer_cfg.accelerator == "ddp":
            kwargs = {"group":"DDP"}
        else:
            kwargs = dict()

        logger = WandbLogger(name=experiment_name,
                             project=project_name, **kwargs)
    elif args.trainer_cfg.logger == "tensorboard":
        logger = TensorBoardLogger(project_name,
                                   name=experiment_name,
                                   )
    else:
        logger = None

  if args.trainer_cfg.logger == "wandb" and not(args.trainer_cfg.fast_dev_run):
      # multi gpu compat
      # https://github.com/Lightning-AI/lightning/issues/5319#issuecomment-869109468
      if os.environ.get("LOCAL_RANK", None) is None:
          os.environ["EXP_LOG_DIR"] = logger.experiment.dir
      args_as_dict = OmegaConf.to_container(args)
      logger.log_hyperparams(args_as_dict)

  # path where data is saved
  today_str = datetime.datetime.now().strftime("%y-%m-%d")
  ctime_str = datetime.datetime.now().strftime("%H-%M-%S")


  experiment_time_str = today_str + "/" + ctime_str
  checkpoint_dir = os.path.join(script_dir,
                                "../",
                                "models",
                                args.module_cfg.classifier,
                                args.module_cfg.module,
                                today_str,
                                ctime_str,
                                )

  # Configure checkpoint and trainer:
  checkpoint = ModelCheckpoint(
      monitor="epoch/val_accuracy",
      mode="max",
      save_last=False,
      dirpath=checkpoint_dir,
  )

  trainerargs = OmegaConf.to_container(args.trainer_cfg)
  trainerargs['logger'] = logger

  if args.trainer_cfg.accelerator == "ddp":
          trainerargs['plugins'] = [
              DDPStrategy(find_unused_parameters=False)
      ]

  all_callbacks = []
  if args.callbacks:
      if args.callbacks.gradnorm:
          all_callbacks.append(GradNormCallbackSplit())
      if args.callbacks.checkpoint_callback:
          all_callbacks.append(checkpoint)
      if args.callbacks.early_stopping:
          all_callbacks.append(EarlyStopping(**args.early_stop_cfg))
      if args.callbacks.get('progress_bar', None):
          all_callbacks.append(ProgressBar(refresh_rate=10))
      if args.callbacks.get('lr_monitor', None):
          all_callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer(**trainerargs, callbacks=all_callbacks)

  # TODO: Effective batch size is split across gpus
  # https://github.com/Lightning-AI/lightning/discussions/3706
  if len(trainer.device_ids) > 1:
    num_gpus = len(trainer.device_ids)
    args.data_cfg.batch_size = int(args.data_cfg.batch_size / num_gpus)
    args.data_cfg.num_workers = int(args.data_cfg.num_workers / num_gpus)

  # Module parameters needed to load data
  if args.module_cfg.get("sequence_pad", None) is None:
    sequence_pad = compute_sequence_pad(dict(args.module_cfg.classifier_cfg))
  else:
    sequence_pad = args.module_cfg.sequence_pad
  extra_kwargs = {
    "sequence_pad": sequence_pad,
    "lambda_weak": args.module_cfg.lambda_weak,
    "lambda_strong": args.module_cfg.lambda_strong,
    "lambda_pred": args.module_cfg.lambda_pred,
    "lambda_recon": args.module_cfg.lambda_recon,
    "lambda_task": args.module_cfg.lambda_task,
  }
  # ------------------------------------------------
  # Load data loaders
  ind_data = all_datasets[args.data_cfg.test_set](args.data_cfg, **extra_kwargs)
  ind_data.setup()

  # ------------------------------------------------
  # Load module
  module_args = args.module_cfg
  OmegaConf.set_struct(module_args, True)

  with open_dict(module_args):
    # inherit from module
    if hasattr(ind_data.train_dataset, "samples_per_class"):
      module_args.samples_per_class = ind_data.train_dataset.samples_per_class.numpy().tolist()
    if hasattr(ind_data.train_dataset, "weak_samples_per_class"):
      module_args.weak_samples_per_class = ind_data.train_dataset.weak_samples_per_class.numpy().tolist()
    module_args.sequence_pad = sequence_pad
    module_args.input_type = args.data_cfg.input_type
    module_args.classifier_cfg.lambda_weak = args.module_cfg.lambda_weak
    module_args.classifier_cfg.lambda_strong = args.module_cfg.lambda_strong
    module_args.classifier_cfg.lambda_pred = args.module_cfg.lambda_pred
    module_args.classifier_cfg.lambda_recon = args.module_cfg.lambda_recon
    module_args.classifier_cfg.lambda_task = args.module_cfg.lambda_task
    module_args.classifier_cfg.num_classes = args.data_cfg.num_classes
    module_args.classifier_cfg.input_size = args.data_cfg.input_size


  module_args = {"hparams": module_args}
  model = all_modules[args.module_cfg.module](**module_args)
  # ------------------------------------------------
  # Load checkpoint
  if args.eval_cfg.ckpt_path is not None:
    try:
      model.model.load_parameters_from_file(args.eval_cfg.ckpt_path)
    except:
      model.load_state_dict(torch.load(args.eval_cfg.ckpt_path)['state_dict'])

  # add output logit dir to args
  if args.trainer_cfg.logger == "wandb" and not(args.trainer_cfg.fast_dev_run):
    try:
      logger.experiment.summary['out/checkpoint_dir'] = checkpoint.dirpath
    except:
      pass

  # Train
  if not(args.eval_cfg.eval_only):
    trainer.fit(model, train_dataloaders=ind_data.train_dataloader(), val_dataloaders=ind_data.val_dataloader())
  # ------------------------------------------------
  if bool(args.eval_cfg.return_model):
      # debug mode
      return ind_data, model, trainer

  # Test
  trainer.test(model, dataloaders=ind_data.test_dataloader())

  # ------------------------------------------------
  # Evaluate in OOD data as well
  # Free data from memory
  # del ind_data

  # Load OOD loader
  ood_data_cfg = args.data_cfg
  if ood_data_cfg.get('ood_expt_ids', None):
    ood_data_cfg.expt_ids = ood_data_cfg.ood_expt_ids
    ood_data_cfg.train_split = 0
    ood_data_cfg.val_split = 0
    ood_data = all_datasets[args.data_cfg.test_set](ood_data_cfg, **extra_kwargs)
    ood_data.setup()
    # Test in ood data:
    model.test_stage_name = "epoch/test_ood_"
    trainer.test(model, dataloaders=ood_data.test_dataloader())


  if args.trainer_cfg.logger == "wandb":
    wandb.finish()


if __name__ == "__main__":
  main()

