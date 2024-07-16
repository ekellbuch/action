"""
References
https://github.com/lindermanlab/S5/blob/main/s5/seq_model.py
"""
import torch
import torch.nn as nn
from action.models.base import BaseModel
from action.models.deep_ssm.s5_fjax.ssm import S5Block

class S5Model_BASE(nn.Module):
  """
  Base model composed of S5 blocks.
  """

  def __init__(self, hparams, predictor_block=False):
    super().__init__()
    self.hparams = hparams
    # init model
    self.layers = nn.Sequential(*[
      S5Block(**self.hparams)
      for _ in range(hparams['n_layers'])
    ])

    if predictor_block:
      self.reconstruction_block = None

  def forward(self, x, states=None):
    """
    Compute the LxH output of the stacked encoder given an Lxd_input
    input sequence.
    Args:
         x (float32): input sequence (L, d_input)
    Returns:
        output sequence (float32): (L, d_model)
    """
    for ii, layer in enumerate(self.layers):
      x, states = layer(x, states)

    return x


class SimpleS5(BaseModel):

  def __init__(self, hparams, type='encoder', in_size=None, hid_size=None, out_size=None):
    super().__init__()
    self.hparams = hparams
    self.model = nn.Sequential()
    if type == 'encoder':
      self.build_encoder()
    else:
      # for an S5 model
      assert hid_size is None and hparams.get('n_hid_units') is None
      in_size_ = hparams['input_size'] if in_size is None else in_size
      out_size_ = hparams['input_size'] if out_size is None else out_size
      self.build_decoder(in_size_, out_size_)

  def build_encoder(self):
    """Construct encoder model using hparams."""

    backbone = S5Model_BASE(self.hparams)
    self.model.add_module('backbone', backbone)

  def build_decoder(self, in_size, out_size):
    # TODO: standarize decoder format
    decoder = nn.Linear(in_size, in_size)

    def _init_weights(module):
      if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
          module.bias.data.zero_()
      elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    decoder.apply(_init_weights)

    self.model.add_module('reconstruction_block', decoder)

  def forward(self, x):
    """
    Args:
        tokens (torch.Tensor): (b, num_frames, input_size)
    """
    return self.model(x)


if __name__ == "__main__":
  def tensor_stats(t: torch.Tensor):  # Clone of lovely_tensors for complex support
    return f"tensor[{t.shape}] n={t.shape.numel()}, u={t.mean()}, s={round(t.std().item(), 3)} var={round(t.var().item(), 3)}\n"


  # toy data:
  data = {"batch_size": 32,
          "sequence_length": 20,
          "input_size": 10,
          }

  (b, t, d) = (data["batch_size"], data['sequence_length'],
               data['input_size'])

  tokens = torch.randn((b, t, d))


  # Hparam configuration
  input_size = 10
  hparams = {
    "bidir": False,
    "n_layers": 10,
    "block_count": 1,
    "factor_rank": None,
    "bcInit": "dense",
  }

  hparams['input_size'] = input_size
  hparams['state_dim'] = input_size
  hparams['dim'] = input_size
  print(hparams)
  # check encoder
  model = S5Model_BASE(hparams)
  outputs = model(tokens)
  assert outputs.shape == (b, t, d)

  # check decoder
  decoder = SimpleS5(hparams, type='decoder')
  xt = decoder(outputs)
  assert tokens.shape == xt.shape

  # check full model
  base_model = S5Model_BASE(hparams, predictor_block=True)
  base_outputs = base_model(tokens)
  assert tokens.shape == base_outputs.shape



