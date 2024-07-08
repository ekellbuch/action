import torch
from sklearn.ensemble import RandomForestRegressor
from pytorch_lightning import LightningModule
import numpy as np
from torchmetrics import R2Score, MeanSquaredError


class RegressionNOBPModule(LightningModule):
    """
    Train a random forest regressor

    args:
    hparams: dict
        Dictionary containing the hyperparameters for the model


    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

        self.lambda_weak = hparams.get('lambda_weak', 0)
        self.lambda_strong = hparams.get('lambda_strong', 0)
        # next step prediction
        self.lambda_pred = hparams.get('lambda_pred', 0)
        self.lambda_task = hparams.get('lambda_task', 0)
        self.lambda_recon = hparams.get('lambda_recon', 0)
        self.pad = hparams.get('sequence_pad', 0)

        assert self.lambda_weak + self.lambda_strong + self.lambda_pred + self.lambda_recon == 0
        assert self.lambda_task == 1
        assert self.pad == 0
        self.model = RandomForestRegressor(n_estimators=self.hparams.classifier_cfg.n_estimators, warm_start=True)
        self.model_fitted = False

        if self.lambda_task > 0:
          self.mse_task = MeanSquaredError()
          self.r2score = R2Score()


    def predict_step(self, batch, batch_idx):
        markers = batch['markers'].numpy()
        num_batches = len(markers)
        all_markers = np.concatenate(markers, axis=0)
        y_pred = self.model.predict(all_markers).reshape(num_batches, -1)
        # convert back to torch
        y_pred = torch.from_numpy(y_pred).to(batch['markers'].dtype)
        return y_pred


    def training_step(self, batch, batch_idx):
        markers = batch['markers'].numpy()
        tasks = batch['tasks'].numpy()

        # number of batches
        # fit a model to all batched data concatenated
        all_markers = np.concatenate(markers, axis=0)
        all_tasks = np.concatenate(tasks, axis=0)

        if self.model_fitted:
            total_estimators = self.model.n_estimators + self.hparams.classifier_cfg.n_estimators
            self.model.set_params(n_estimators=total_estimators)
            self.model.fit(all_markers, all_tasks)
        else:
            self.model.fit(all_markers, all_tasks)
            self.model_fitted = True

        # print(f"Training step: {batch_idx}, {self.model.n_estimators}")

        y_pred = self.model.predict(all_markers)

        y_pred = torch.from_numpy(y_pred).to(batch['tasks'].dtype)
        self.r2score(y_pred, batch['tasks'].view(-1))
        self.mse_task(y_pred, batch['tasks'].view(-1))

        self.log(f'batch/train_r2score', self.r2score)
        self.log(f'batch/train_mse', self.mse_task)
        return

    def on_train_epoch_end(self) -> None:
        self.log(f'epoch/train_r2score', self.r2score)
        self.log(f'epoch/train_mse', self.mse_task)

    def validation_step(self, batch, batch_idx):
        num_batches = len(batch['markers'])
        all_markers = np.concatenate(batch['markers'].numpy(), axis=0)
        y_pred = self.model.predict(all_markers)

        # calculate and log metrics
        y_pred = torch.from_numpy(y_pred).to(batch['markers'].dtype)
        self.r2score(y_pred, batch['tasks'].view(-1))
        self.mse_task(y_pred, batch['tasks'].view(-1))

        self.log(f'batch/val_r2score', self.r2score)
        self.log(f'batch/val_mse', self.mse_task)
        return

    def on_validation_epoch_end(self) -> None:
        self.log(f'epoch/val_r2score', self.r2score)
        self.log(f'epoch/val_mse', self.mse_task)


    def test_step(self, batch, batch_idx):
        num_batches = len(batch['markers'])
        all_markers = np.concatenate(batch['markers'].numpy(), axis=0)

        # predict data
        y_pred = self.model.predict(all_markers)
        # calculate and log metrics
        y_pred = torch.from_numpy(y_pred).to(batch['markers'].dtype)
        self.r2score(y_pred, batch['tasks'].view(-1))
        self.mse_task(y_pred, batch['tasks'].view(-1))

        self.log(f'batch/test_r2score', self.r2score)
        self.log(f'batch/test_mse', self.mse_task)
        return

    def on_test_epoch_end(self) -> None:
        self.log(f'epoch/test_r2score', self.r2score)
        self.log(f'epoch/test_mse', self.mse_task)

    def configure_optimizers(self):
        return None  # RandomForest doesn't require an optimizer