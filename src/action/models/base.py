import torch
from torch import nn
import numpy as np
from action.models.utils_model import reparameterize_gaussian


model_tiers = {
    'large': [256, 32],
    'mid': [128, 16],
    'small': [64, 16]
}

class CustomModel(nn.Module):
  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams
    input_size = hparams.get('input_size', 10)
    model_tier = hparams.get('model_tier', 'small')
    num_classes = hparams.get('num_classes', 3)

    hidden_sizes = model_tiers[model_tier]
    self.fc1 = nn.Linear(input_size, hidden_sizes[0])
    self.bn1 = nn.BatchNorm1d(hidden_sizes[0], momentum=0.99, eps=0.001, affine=True, track_running_stats=True)
    self.dropout1 = nn.Dropout(0.3)
    self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
    self.bn2 = nn.BatchNorm1d(hidden_sizes[1], momentum=0.99, eps=0.001, affine=True, track_running_stats=True)
    self.dropout2 = nn.Dropout(0.3)
    self.fc3 = nn.Linear(hidden_sizes[1], num_classes)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.bn1(x)
    x = self.dropout1(x)
    x = torch.relu(self.fc2(x))
    x = self.bn2(x)
    x = self.dropout2(x)
    x = self.fc3(x)
    return x


def get_activation_func_from_str(activation_str):
  if activation_str == 'linear':
    activation_func = None
  elif activation_str == 'relu':
    activation_func = nn.ReLU()
  elif activation_str == 'lrelu':
    activation_func = nn.LeakyReLU(0.05)
  elif activation_str == 'sigmoid':
    activation_func = nn.Sigmoid()
  elif activation_str == 'tanh':
    activation_func = nn.Tanh()
  else:
    raise ValueError('"%s" is an invalid activation function' % activation_str)

  return activation_func


class BaseModel(nn.Module):
    """Template for PyTorch models."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __str__(self):
        """Pretty print model architecture."""
        raise NotImplementedError

    def build_model(self):
        """Build model from hparams."""
        raise NotImplementedError

    @staticmethod
    def _build_linear(global_layer_num, name, in_size, out_size):

        linear_layer = nn.Sequential()

        # add layer (cross entropy loss handles activation)
        layer = nn.Linear(in_features=in_size, out_features=out_size)
        layer_name = str('dense(%s)_layer_%02i' % (name, global_layer_num))
        linear_layer.add_module(layer_name, layer)

        return linear_layer

    @staticmethod
    def _build_mlp(
            global_layer_num, in_size, hid_size, out_size, n_hid_layers=1, activation='lrelu'):

        mlp = nn.Sequential()

        in_size_ = in_size

        # loop over hidden layers (0 layers <-> linear model)
        for i_layer in range(n_hid_layers + 1):

            if i_layer == n_hid_layers:
                out_size_ = out_size
            else:
                out_size_ = hid_size

            # add layer
            layer = nn.Linear(in_features=in_size_, out_features=out_size_)
            name = str('dense_layer_%02i' % global_layer_num)
            mlp.add_module(name, layer)

            # add activation
            if i_layer == n_hid_layers:
                # no activation for final layer
                activation_func = None
            else:
                activation_func = get_activation_func_from_str(activation)
            if activation_func:
                name = '%s_%02i' % (activation, global_layer_num)
                mlp.add_module(name, activation_func)

            # update layer info
            global_layer_num += 1
            in_size_ = out_size_

        return mlp

    def forward(self, *args, **kwargs):
        """Push data through model."""
        raise NotImplementedError

    def save(self, filepath):
        """Save model parameters."""
        save(self.state_dict(), filepath)

    def get_parameters(self):
        """Get all model parameters that have gradient updates turned on."""
        return filter(lambda p: p.requires_grad, self.parameters())

    def load_parameters_from_file(self, filepath):
        """Load parameters from .pt file."""
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))


class EncoderDecoderModel(BaseModel):
  """General wrapper class for behavioral segmentation models."""

  def __init__(self, hparams):
    """

    Parameters
    ----------
    hparams : dict
        - backbone (str): 'temporal-mlp' | 'dtcn' | 'lstm' | 'gru'
        - rng_seed_model (int): random seed to control weight initialization
        - input_size (int): number of input channels
        - num_classes (int): number of classes
        - task_size (int): number of regression tasks
        - batch_pad (int): padding needed to account for convolutions
        - n_hid_layers (int): hidden layers of network architecture
        - n_hid_units (int): hidden units per layer
        - n_lags (int): number of lags in input data to use for temporal convolution
        - activation (str): 'linear' | 'relu' | 'lrelu' | 'sigmoid' | 'tanh'
        - classifier_type (str): 'multiclass' | 'binary' | 'multibinary'
        - class_weights (array-like): weights on classes
        - variational (bool): whether or not model is variational
        - lambda_weak (float): hyperparam on weak label classification
        - lambda_strong (float): hyperparam on srong label classification
        - lambda_pred (float): hyperparam on next step prediction
        - lambda_recon (float): hyperparam on reconstruction
        - lambda_task (float): hyperparam on task regression

    """
    super().__init__()
    self.hparams = hparams

    # model dict will contain some or all of the following components:
    # - encoder: inputs -> latents
    # - classifier: latents -> hand labels
    # - classifier_weak: latents -> heuristic/pseudo labels
    # - task_predictor: latents -> tasks
    # - decoder: latents[t] -> inputs[t]
    # - predictor: latents[t] -> inputs[t+1]
    self.model = nn.ModuleDict()
    self.build_model()

    # label loss based on cross entropy; don't compute gradient when target = 0
    classifier_type = hparams.get('classifier_type', 'multiclass')
    if classifier_type == 'multiclass':
      # multiple mutually exclusive classes, 0 is backgroud class
      ignore_index = hparams.get('ignore_class', 0)
    elif classifier_type == 'binary':
      # single class
      ignore_index = -100  # pytorch default
    elif classifier_type == 'multibinary':
      # multiple non-mutually exclusive classes (each a binary classification)
      raise NotImplementedError
    else:
      raise NotImplementedError("classifier type must be 'multiclass' or 'binary'")

  def __str__(self):
    """Pretty print model architecture."""

    format_str = '\n%s architecture\n' % self.hparams['backbone'].upper()
    format_str += '------------------------\n'

    format_str += 'Encoder:\n'
    for i, module in enumerate(self.model['encoder'].model):
      format_str += str('    {}: {}\n'.format(i, module))
    format_str += '\n'

    if self.hparams.get('variational', False):
      format_str += 'Variational Layers:\n'
      for l in ['latent_mean', 'latent_logvar']:
        for i, module in enumerate(self.model[l]):
          format_str += str('    {}: {}\n'.format(i, module))
      format_str += '\n'

    if 'decoder' in self.model:
      format_str += 'Decoder:\n'
      for i, module in enumerate(self.model['decoder'].model):
        format_str += str('    {}: {}\n'.format(i, module))
      format_str += '\n'

    if 'predictor' in self.model:
      format_str += 'Predictor:\n'
      for i, module in enumerate(self.model['predictor'].model):
        format_str += str('    {}: {}\n'.format(i, module))
      format_str += '\n'

    if 'classifier' in self.model:
      format_str += 'Classifier:\n'
      for i, module in enumerate(self.model['classifier']):
        format_str += str('    {}: {}\n'.format(i, module))
      format_str += '\n'

    if 'classifier_weak' in self.model:
      format_str += 'Classifier Weak:\n'
      for i, module in enumerate(self.model['classifier_weak']):
        format_str += str('    {}: {}\n'.format(i, module))
      format_str += '\n'

    if 'task_predictor' in self.model:
      format_str += 'Task Predictor:\n'
      for i, module in enumerate(self.model['task_predictor']):
        format_str += str('    {}: {}\n'.format(i, module))

    return format_str

  def build_model(self):
    """Construct the model using hparams."""

    # set random seeds for control over model initialization
    rng_seed_model = self.hparams.get('rng_seed_model', 0)
    torch.manual_seed(rng_seed_model)
    np.random.seed(rng_seed_model)

    # select backbone network
    if self.hparams['backbone'].lower() == 'temporal-mlp':
      from action.models.temporalmlp import TemporalMLP as Module
    elif self.hparams['backbone'].lower() == 'tcn':
      raise NotImplementedError('deprecated; use dtcn instead')
    elif self.hparams['backbone'].lower() == 'dtcn':
      from action.models.tcn import DilatedTCN as Module
    elif self.hparams['backbone'].lower() in ['lstm', 'gru']:
      from action.models.rnn import RNN as Module
    elif self.hparams['backbone'].lower() in ['animalst']:
      from action.models.min_gpt import AnimalSTWoPosEmb as Module
    elif self.hparams['backbone'].lower() == 'tgm':
      raise NotImplementedError
      # from action.models.tgm import TGM as Module
    else:
      raise ValueError('"%s" is not a valid backbone network' % self.hparams['backbone'])

    global_layer_num = 0

    # build encoder module
    self.model['encoder'] = Module(self.hparams, type='encoder')
    if self.hparams.get('variational', False):
      self.hparams['kl_weight'] = 1  # weight in front of kl term; anneal this using callback
      self.model['latent_mean'] = self._build_linear(
        global_layer_num=len(self.model['encoder'].model), name='latent_mean',
        in_size=self.hparams['n_hid_units'], out_size=self.hparams['n_hid_units'])
      self.model['latent_logvar'] = self._build_linear(
        global_layer_num=len(self.model['encoder'].model), name='latent_logvar',
        in_size=self.hparams['n_hid_units'], out_size=self.hparams['n_hid_units'])

    # build decoder module
    if self.hparams.get('lambda_recon', 0) > 0:
      self.model['decoder'] = Module(self.hparams, type='decoder')

    # build predictor module
    if self.hparams.get('lambda_pred', 0) > 0:
      self.model['predictor'] = Module(self.hparams, type='decoder')

    # classifier: single linear layer for hand labels
    if self.hparams.get('lambda_strong', 0) > 0:
      self.model['classifier'] = self._build_linear(
        global_layer_num=global_layer_num, name='classification',
        in_size=self.hparams['n_hid_units'], out_size=self.hparams['num_classes'])

    # classifier: single linear layer for heuristic labels
    if self.hparams.get('lambda_weak', 0) > 0:
      self.model['classifier_weak'] = self._build_linear(
        global_layer_num=global_layer_num, name='classification',
        in_size=self.hparams['n_hid_units'], out_size=self.hparams['num_classes'])

    # task regression: single linear layer
    if self.hparams.get('lambda_task', 0) > 0:
      self.model['task_predictor'] = self._build_mlp(
        global_layer_num=global_layer_num, in_size=self.hparams['n_hid_units'],
        hid_size=self.hparams['n_hid_units'], out_size=self.hparams['task_size'],
        n_hid_layers=1)

  def forward(self, x):
    """Process input data.

    Parameters
    ----------
    x : torch.Tensor
        input data of shape (n_sequences, sequence_length, n_markers)

    Returns
    -------
    dict of model outputs/internals as torch tensors
        - 'labels' (torch.Tensor): model classification
           shape of (n_sequences, sequence_length, n_classes)
        - 'labels_weak' (torch.Tensor): model classification of weak/pseudo labels
          shape of (n_sequences, sequence_length, n_classes)
        - 'reconstruction' (torch.Tensor): input decoder prediction
          shape of (n_sequences, sequence_length, n_markers)
        - 'prediction' (torch.Tensor): one-step-ahead prediction
          shape of (n_sequences, sequence_length, n_markers)
        - 'task_prediction' (torch.Tensor): prediction of regression tasks
          (n_sequences, sequence_length, n_tasks)
        - 'embedding' (torch.Tensor): behavioral embedding used for classification/prediction
          in non-variational models
          shape of (n_sequences, sequence_length, embedding_dim)
        - 'mean' (torch.Tensor): mean of appx posterior of latents in variational models
          shape of (n_sequences, sequence_length, embedding_dim)
        - 'logvar' (torch.Tensor): logvar of appx posterior of latents in variational models
          shape of (n_sequences, sequence_length, embedding_dim)
        - 'sample' (torch.Tensor): sample from appx posterior of latents in variational models
          shape of (n_sequences, sequence_length, embedding_dim)

    """
    # push data through encoder to get latent embedding
    # x = B x T x N (e.g. B = 2, T = 500, N = 16)
    x = self.model['encoder'](x)
    if self.hparams.get('variational', False):
      mean = self.model['latent_mean'](x)
      logvar = self.model['latent_logvar'](x)
      z = reparameterize_gaussian(mean, logvar)
    else:
      mean = x
      logvar = None
      z = x

    # push embedding through classifiers to get hand labels
    if self.hparams.get('lambda_strong', 0) > 0:
      y = self.model['classifier'](z)
    else:
      y = None

    # push embedding through linear layer to heuristic/pseudo labels
    if self.hparams.get('lambda_weak', 0) > 0:
      y_weak = self.model['classifier_weak'](z)
    else:
      y_weak = None

    # push embedding through linear layer to get task predictions
    if self.hparams.get('lambda_task', 0) > 0:
      w = self.model['task_predictor'](z)
    else:
      w = None

    # push embedding through decoder network to get data at current time point
    if self.hparams.get('lambda_recon', 0) > 0:
      xt = self.model['decoder'](z)
    else:
      xt = None

    # push embedding through predictor network to get data at subsequent time points
    if self.hparams.get('lambda_pred', 0) > 0:
      xtp1 = self.model['predictor'](z)
    else:
      xtp1 = None

    return {
      'labels': y,  # (n_sequences, sequence_length, n_classes)
      'labels_weak': y_weak,  # (n_sequences, sequence_length, n_classes)
      'reconstruction': xt,  # (n_sequences, sequence_length, n_markers)
      'prediction': xtp1,  # (n_sequences, sequence_length, n_markers)
      'task_prediction': w,  # (n_sequences, sequence_length, n_tasks)
      'embedding': mean,  # (n_sequences, sequence_length, embedding_dim)
      'latent_mean': mean,  # (n_sequences, sequence_length, embedding_dim)
      'latent_logvar': logvar,  # (n_sequences, sequence_length, embedding_dim)
      'sample': z,  # (n_sequences, sequence_length, embedding_dim)
    }


all_classifiers = {
  'baseline': CustomModel,
  'encoder_decoder': EncoderDecoderModel,
}