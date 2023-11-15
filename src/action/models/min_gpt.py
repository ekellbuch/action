import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from action.models.base import BaseModel

__all__ = ['AnimalSTWoPosEmb']

class CausalSelfAttention(nn.Module):
    """
    From: https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L37

    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, hparams):
        super().__init__()
        assert hparams['n_embd'] % hparams['n_head'] == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(hparams['n_embd'], hparams['n_embd'])
        self.query = nn.Linear(hparams['n_embd'], hparams['n_embd'])
        self.value = nn.Linear(hparams['n_embd'], hparams['n_embd'])
        # regularization
        self.attn_drop = nn.Dropout(hparams['attn_pdrop'])
        self.resid_drop = nn.Dropout(hparams['resid_pdrop'])
        # output projection
        self.proj = nn.Linear(hparams['n_embd'], hparams['n_embd'])
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num_frames = hparams['num_frames']
        num_animals = hparams['num_animals']
        a = torch.tril(torch.ones(num_frames, num_frames))[:, :, None, None]
        b = torch.ones(1, 1, num_animals, num_animals)
        mask = (a * b).transpose(1, 2).reshape(num_frames *
                                               num_animals, -1)[None, None, :, :]
        self.register_buffer("mask", mask)
        self.n_head = hparams['n_head']

    def forward(self, x, mask, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C //
                             self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C //
                               self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C //
                               self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(
            (self.mask * mask[:, None, None, :]) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """
    From: https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L81

    An unassuming Transformer block.
    """

    def __init__(self, hparams):
        super().__init__()
        self.ln1 = nn.LayerNorm(hparams['n_embd'])
        self.ln2 = nn.LayerNorm(hparams['n_embd'])
        self.attn = CausalSelfAttention(hparams)
        self.mlp = nn.Sequential(
            nn.Linear(hparams['n_embd'], 4 * hparams['n_embd']),
            nn.GELU(),
            nn.Linear(4 * hparams['n_embd'], hparams['n_embd']),
            nn.Dropout(hparams['resid_pdrop']),
        )

    def forward(self, inputs):
        (x, mask) = inputs
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return (x, mask)


class AnimalSTWoPosEmb_BASE(nn.Module):
    """
    Each animal in each frame as a token.
    Attention is only applied to the animals in the left frames.
    """

    def __init__(self, hparams, predictor_block=False):
        super().__init__()
        self.hparams = hparams

        # input embedding stem
        self.tok_emb = nn.Linear(hparams['input_size'], hparams['n_embd'])
        self.bn = nn.BatchNorm2d(hparams['input_size'])
        # self.pos_emb = nn.Parameter(torch.zeros(
        #     1, hparams['total_frames'], hparams['n_embd']))
        self.drop = nn.Dropout(hparams['embd_pdrop'])
        # transformer
        self.blocks = nn.Sequential(*[Block(hparams)
                                    for _ in range(hparams['n_hid_layers'])])

        # decoder head
        self.ln_f = nn.LayerNorm(hparams['n_embd'])

        self.proj = nn.Linear(hparams['n_embd'], hparams['n_hid_units'])

        if predictor_block:
            self.build_decoder(hparams['n_hid_units'], hparams['n_hid_units'], hparams['input_size'])
        else:
            self.reconstruction_block = None

        self.apply(self._init_weights)

    def build_decoder(self, in_size, hid_size, out_size):
        self.reconstruction_block = nn.Sequential(
            nn.Linear(hid_size, in_size),
            nn.Tanh(),
            nn.LayerNorm(in_size),
            nn.Linear(in_size, out_size)
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x, mask):
        x = self.drop(x)
        (x, mask) = self.blocks((x, mask))
        x = self.ln_f(x)
        x = self.proj(x)    # (b, t * c, n_hid_units)
        return x

    def forward_encoder(self, tokens):
        """
        Args:
            tokens (torch.Tensor): (b, num_frames, num_animals, input_size)
        """
        b, t, c, d = tokens.shape
        # assert t <= self.total_frames, "Cannot forward, pos_emb is exhausted."
        #tokens = tokens.unsqueeze(dim=-2)
        #c = 1  # number of animals

        masks = ~(torch.isnan(tokens).long().sum(-1).bool())
        #masks = masks.unsqueeze(dim=-1)

        token_embeddings = self.tok_emb(
            self.bn(tokens.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        # position_embeddings = []
        # for i in range(b):
        #     position_embeddings.append(self.pos_emb[:, pos[i]: pos[i] + t, :])
        # position_embeddings = torch.cat(
        #     position_embeddings, dim=0)[:, :, None, :]
        embeddings = token_embeddings.view(b, t * c, -1)
        masks = masks.view(b, -1)

        feat_LR = self.encode(embeddings, masks)

        return feat_LR

    def forward(self, x):
        x = x.unsqueeze(dim=-2)
        output = self.forward_encoder(x)

        if not (self.reconstruction_block is None):
            output = self.reconstruction_block(output)

        return output


class AnimalSTWoPosEmb(BaseModel):

    def __init__(self, hparams, type='encoder', in_size=None, hid_size=None, out_size=None):
        super().__init__()
        self.hparams = hparams
        self.model = nn.Sequential()
        if type == 'encoder':
            self.build_encoder()
        else:
            in_size_ = hparams['n_hid_units'] if in_size is None else in_size
            hid_size_ = hparams['n_hid_units'] if hid_size is None else hid_size
            out_size_ = hparams['input_size'] if out_size is None else out_size
            self.build_decoder(in_size_, hid_size_, out_size_)

    def build_encoder(self):
        """Construct encoder model using hparams."""

        backbone = AnimalSTWoPosEmb_BASE(self.hparams)
        self.model.add_module('backbone', backbone)

    def build_decoder(self, in_size, hid_size, out_size):

        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

        decoder = nn.Sequential(
            nn.Linear(hid_size, in_size),
            nn.Tanh(),
            nn.LayerNorm(in_size),
            nn.Linear(in_size, out_size)
        )

        decoder.apply(_init_weights)

        self.model.add_module('reconstruction_block', decoder)


    def forward(self, x):
        """
        Args:
            tokens (torch.Tensor): (b, num_frames, input_size)
        """
        return self.model(x)


if __name__ == '__main__':

    torch.manual_seed(0)

    hparams = {
        'n_embd': 48,
        'n_hid_layers': 12,
        'n_head': 12,
        'n_hid_units': 5,

        'embd_pdrop': 0.1,
        'resid_pdrop': 0.1,
        'attn_pdrop': 0.1,
        'num_frames': 20,
        'num_animals': 1,
        'input_size': 10,

    }

    model = AnimalSTWoPosEmb(hparams)


    # toy data:
    data = {"batch_size": 32,
            "sequence_length":20,
            "input_size": 10,
            }

    (b, t, d) = (data["batch_size"], data['sequence_length'],
                 data['input_size'])

    tokens = torch.randn((b, t, d))

    # check encoder
    outputs = model(tokens)
    assert outputs.shape == (b, t, hparams['n_hid_units'])

    # check decoder
    decoder = AnimalSTWoPosEmb(hparams, type='decoder')
    xt = decoder(outputs)
    assert tokens.shape == xt.shape

    # check full model
    base_model = AnimalSTWoPosEmb_BASE(hparams, predictor_block=True)

    base_outputs = base_model(tokens)

    assert tokens.shape == base_outputs.shape

    # assert torch.equal(xt, base_outputs)







