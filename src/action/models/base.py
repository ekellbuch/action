import torch
from torch import nn


model_tiers = {
    'large': [256, 32],
    'mid': [128, 16],
    'small': [64, 16]
}

class CustomModel(nn.Module):
  def __init__(self, input_size, model_tier='small', num_classes=2):
    super().__init__()
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



class Encoder(nn.Module):
  def __init__(self, NUM_FEATURES,
               hidden_size_layer_1,
               hidden_size_layer_2,
               dropout_encoder):
    super(Encoder, self).__init__()

    self.input_size = NUM_FEATURES
    self.hidden_size = hidden_size_layer_1
    self.hidden_size_2 = hidden_size_layer_2
    self.n_layers = 2
    self.dropout = dropout_encoder
    self.bidirectional = True

    self.encoder_rnn = nn.GRU(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.n_layers,
                              bias=True,
                              batch_first=True,
                              dropout=self.dropout,
                              bidirectional=self.bidirectional)  # UNRELEASED!

    self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers

  def forward(self, inputs):
    outputs_1, hidden_1 = self.encoder_rnn(inputs)  # UNRELEASED!

    hidden = torch.cat((hidden_1[0, ...], hidden_1[1, ...], hidden_1[2, ...], hidden_1[3, ...]), 1)

    return hidden

class Lambda(nn.Module):
  """Lambda module converts output of encoder to latent vector

  :param hidden_size: hidden size of the encoder
  :param latent_length: latent vector length
  """

  def __init__(self, ZDIMS, hidden_size_layer_1, hidden_size_layer_2, softplus):
    super(Lambda, self).__init__()

    self.hid_dim = hidden_size_layer_1 * 4
    self.latent_length = ZDIMS
    self.softplus = softplus

    self.hidden_to_mean = nn.Linear(self.hid_dim, self.latent_length)
    self.hidden_to_logvar = nn.Linear(self.hid_dim, self.latent_length)

    if self.softplus == True:
      print(
        "Using a softplus activation to ensures that the variance is parameterized as non-negative and activated by a smooth function")
      self.softplus_fn = nn.Softplus()

  def forward(self, hidden):
    """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

    :param cell_output: last hidden state of encoder
    :return: latent vector
    """

    self.mean = self.hidden_to_mean(hidden)
    if self.softplus == True:
      self.logvar = self.softplus_fn(self.hidden_to_logvar(hidden))
    else:
      self.logvar = self.hidden_to_logvar(hidden)

    if self.training:
      std = torch.exp(0.5 * self.logvar)
      eps = torch.randn_like(std)
      return eps.mul(std).add_(self.mean), self.mean, self.logvar
    else:
      return self.mean, self.mean, self.logvar


class RNN_VAE(nn.Module):
  def __init__(self,
               TEMPORAL_WINDOW,
               ZDIMS, NUM_FEATURES,
               FUTURE_DECODER,
               FUTURE_STEPS,
               hidden_size_layer_1,
               hidden_size_layer_2,
               hidden_size_rec,
               hidden_size_pred,
               dropout_encoder,
               dropout_rec,
               dropout_pred,
               softplus):
    super(RNN_VAE, self).__init__()

    self.FUTURE_DECODER = FUTURE_DECODER
    self.seq_len = int(TEMPORAL_WINDOW / 2)
    self.encoder = Encoder(NUM_FEATURES, hidden_size_layer_1, hidden_size_layer_2, dropout_encoder)
    self.lmbda = Lambda(ZDIMS, hidden_size_layer_1, hidden_size_layer_2, softplus)
    self.decoder = Decoder(self.seq_len, ZDIMS, NUM_FEATURES, hidden_size_rec, dropout_rec)
    if FUTURE_DECODER:
      self.decoder_future = Decoder_Future(self.seq_len, ZDIMS, NUM_FEATURES, FUTURE_STEPS, hidden_size_pred,
                                           dropout_pred)

  def forward(self, seq):

    """ Encode input sequence """
    h_n = self.encoder(seq)

    """ Compute the latent state via reparametrization trick """
    z, mu, logvar = self.lmbda(h_n)
    ins = z.unsqueeze(2).repeat(1, 1, self.seq_len)
    ins = ins.permute(0, 2, 1)

    """ Predict the future of the sequence from the latent state"""
    prediction = self.decoder(ins, z)

    if self.FUTURE_DECODER:
      future = self.decoder_future(ins, z)
      return prediction, future, z, mu, logvar
    else:
      return prediction, z, mu, logvar

  class Decoder(nn.Module):
    def __init__(self, TEMPORAL_WINDOW, ZDIMS, NUM_FEATURES, hidden_size_rec, dropout_rec):
      super(Decoder, self).__init__()

      self.num_features = NUM_FEATURES
      self.sequence_length = TEMPORAL_WINDOW
      self.hidden_size = hidden_size_rec
      self.latent_length = ZDIMS
      self.n_layers = 1
      self.dropout = dropout_rec
      self.bidirectional = True

      self.rnn_rec = nn.GRU(self.latent_length, hidden_size=self.hidden_size, num_layers=self.n_layers,
                            bias=True, batch_first=True, dropout=self.dropout, bidirectional=self.bidirectional)

      self.hidden_factor = (2 if self.bidirectional else 1) * self.n_layers  # NEW

      self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size * self.hidden_factor)  # NEW
      self.hidden_to_output = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.num_features)

    def forward(self, inputs, z):
      batch_size = inputs.size(0)  # NEW

      hidden = self.latent_to_hidden(z)  # NEW

      hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)  # NEW

      decoder_output, _ = self.rnn_rec(inputs, hidden)
      prediction = self.hidden_to_output(decoder_output)

      return prediction


all_classifiers = {
  'baseline': CustomModel,
}