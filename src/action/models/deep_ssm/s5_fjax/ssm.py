"""
References:

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple, Optional, Union
import math
from action.models.deep_ssm.s5_fjax.jax_func import associative_scan, lecun_normal
from action.models.deep_ssm.s5_fjax.ssm_init import init_VinvB, init_CV, init_log_steps, make_DPLR_HiPPO, trunc_standard_normal
from torchtyping import TensorType


# Discretization functions
def discretize_bilinear(Lambda: TensorType["num_states"],
                        B_tilde: TensorType["num_states","num_features"],
                        Delta: TensorType["num_states"]
                        ) -> Tuple[TensorType["num_states"], TensorType["num_states","num_features"]]:
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    # TODO: check complex vs real
    # Lambda = torch.view_as_complex(Lambda)
    Identity = torch.ones(Lambda.shape[0])
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde

    #Lambda_bar = torch.view_as_real(Lambda_bar)
    #B_bar = torch.view_as_real(B_bar)

    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    # Identity = torch.ones(Lambda.shape[0], device=Lambda.device) # (replaced by -1)
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - 1))[..., None] * B_tilde
    return Lambda_bar, B_bar


@torch.jit.script
def binary_operator(
    q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]
):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    # return A_j * A_i, A_j * b_i + b_j
    return A_j * A_i, torch.addcmul(b_j, A_j, b_i)


def apply_ssm(
    Lambda_bars: TensorType["num_states"],
    B_bars: TensorType["num_states", "num_features"],
    C_tilde: TensorType["num_features", "num_states"],
    D: TensorType["num_features"],
    input_sequence: TensorType["seq_length", "num_features"],
    prev_state: TensorType["num_states"],
    conj_sym: bool = False,
    bidirectional: bool = False,
)-> Tuple[TensorType["seq_length","num_features"], TensorType["num_states"]]:
    """
    Apply a linear state-space model to an input sequence.
    :param Lambda_bars (torch.Tensor):
    :param B_bars:
    :param C_tilde:
    :param D:
    :param input_sequence:
    :param prev_state:
    :param conj_sym:
    :param bidirectional:
    :return:
    """
    cinput_sequence = input_sequence.type(
        Lambda_bars.dtype
    )  # Cast to correct complex type

    if B_bars.ndim == 3:
        # Dynamic timesteps (significantly more expensive)
        Bu_elements = torch.vmap(lambda B_bar, u: B_bar @ u)(B_bars, cinput_sequence)
    else:
        # Static timesteps
        Bu_elements = torch.vmap(lambda u: B_bars @ u)(cinput_sequence)

    if Lambda_bars.ndim == 1:  # Repeat for associative_scan
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)

    _, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))

    if bidirectional:
        _, xs2 = associative_scan(
            binary_operator, (Lambda_bars, Bu_elements), reverse=True
        )
        xs = torch.cat((xs, xs2), axis=-1)

    # TODO: use prev_state
    Du = torch.vmap(lambda u: D * u)(input_sequence)
    # TODO: the last element of xs (non-bidir) is the hidden state, allow returning it
    if conj_sym:
        return torch.vmap(lambda x: 2*(C_tilde @ x).real)(xs) + Du, xs[-1]
    else:
        return torch.vmap(lambda x: (C_tilde @ x).real)(xs) + Du, xs[-1]


def as_complex(t: torch.Tensor, dtype=torch.complex64):
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt


Initialization = Literal["complex_normal", "lecun_normal", "truncate_standard_normal"]


class S5SSM(torch.nn.Module):
    def __init__(
        self,
        Lambda_re_init: torch.Tensor,
        Lambda_im_init: torch.Tensor,
        V: torch.Tensor,
        Vinv: torch.Tensor,
        H: int,
        P: int,
        C_init: str,
        dt_min: float,
        dt_max: float,
        conj_sym: bool = True,
        clip_eigs: bool = False,
        bidirectional: bool = False,
        step_rescale: float = 1.0,
        discretization: Literal["zoh", "bilinear"] = "bilinear",
        bandlimit: Optional[float] = None,
    ):
        # TODO: conj_sym,
        """The S5 SSM
        Args:
            Lambda_re_init  (complex64): Initial diagonal state matrix       (P,)
            V           (complex64): Eigenvectors used for init          (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init  (P,P)
            h           (int32):     Number of features of input seq
            p           (int32):     state size
            k           (int32):     rank of low-rank factorization (if used)
            C_init      (string):    Specifies How B and C are initialized
                        Options: [factorized: low-rank factorization,
                                dense: dense matrix drawn from Lecun_normal]
                                dense_columns: dense matrix where the columns
                                of B and the rows of C are each drawn from Lecun_normal
                                separately (i.e. different fan-in then the dense option).
                                We found this initialization to be helpful for Pathx.
            discretization: (string) Specifies discretization method
                            options: [zoh: zero-order hold method,
                                    bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_rescale:  (float32): allows for changing the step size, e.g. after training
                                    on a different resolution for the speech commands benchmark
        """
        super().__init__()

        self.conj_sym = conj_sym
        self.C_init = C_init
        self.bidirectional = bidirectional
        self.bandlimit = bandlimit
        self.step_rescale = step_rescale
        self.clip_eigs = clip_eigs

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2 * P
        else:
            local_P = P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = torch.nn.Parameter(Lambda_re_init)
        self.Lambda_im = torch.nn.Parameter(Lambda_im_init)

        Lambda = self.get_lambda()

        # Initialize input to state (B) matrix
        # TODO: remove torch.float
        self.B = torch.nn.Parameter(
            init_VinvB(lecun_normal(), Vinv)((local_P, H), torch.float)
        )
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            #C_shape = (H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            #C_shape = (H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = torch.normal(0, 0.5**0.5, (H, P, 2), dtype=torch.complex64)
        else:
            raise NotImplementedError(
                "C_init method {} not implemented".format(self.C_init))


        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = torch.cat((C_init, C_init), axis=-1,)
                C = torch.nn.Parameter(C)
                self.C_tilde = C[..., 0] + 1j * C[..., 1]
            else:
                C = torch.nn.Parameter(C_init)
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            if self.bidirectional:
                self.C1 = torch.nn.Parameter(init_CV(C_init, (H, local_P), V))
                self.C2 = torch.nn.Parameter(init_CV(C_init, (H, local_P), V))

                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]

                self.C_tilde = torch.cat((C1, C2), axis=-1)

            else:
                C = init_CV(C_init, (H, local_P), V)
                self.C = torch.nn.Parameter(C)

                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]


        # Initialize feedthrough (D) matrix
        self.D = torch.nn.Parameter(torch.rand(H,))

        # Initialize learnable discretization timescale value
        self.log_step = torch.nn.Parameter(init_log_steps(P, dt_min, dt_max))


        if discretization == "zoh":
            self.discretize = discretize_zoh
        elif discretization == "bilinear":
            self.discretize = discretize_bilinear
        else:
            raise ValueError(f"Unknown discretization {discretization}")

        if self.bandlimit is not None:
            step = step_rescale * torch.exp(self.log_step)
            freqs = step / step_rescale * Lambda[:, 1].abs() / (2 * math.pi)
            mask = torch.where(freqs < bandlimit * 0.5, 1, 0)  # (64, )
            self.C = torch.nn.Parameter(
                torch.view_as_real(torch.view_as_complex(self.C) * mask)
            )

    def initial_state(self, batch_size: Optional[int]):
        batch_shape = (batch_size,) if batch_size is not None else ()
        return torch.zeros((*batch_shape, self.C_tilde.shape[-1]))
    def get_lambda(self):
        if self.clip_eigs:
            Lambda = torch.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            Lambda = self.Lambda_re + 1j * self.Lambda_im
        return Lambda
    def get_BC_tilde(self):
        B_tilde = self.B[..., 0] + 1j * self.B[...,1]
        C_tilde = self.C_tilde
        return B_tilde, C_tilde

    # NOTE: can only be used as RNN OR S5(MIMO) (no mixing)
    def forward(self,
                signal: TensorType["batch_size", "seq_length", "num_features"],
                prev_state: TensorType["batch_size", "num_states"],
		            step_rescale: Union[float, torch.Tensor] = 1.0):
 
        B_tilde, C_tilde = self.get_BC_tilde()
        Lambda = self.get_lambda()
       
        if not torch.is_tensor(step_rescale) or step_rescale.ndim == 0:
            step_scale = step_rescale * torch.exp(self.log_step)
        else:
            # TODO: include invididual steps for discretize
            step_scale = step_rescale[:, None] * torch.exp(self.log_step)

        Lambda_bar, B_bar = self.discretize(Lambda, B_tilde, step_scale)

        return apply_ssm(
            Lambda_bar, B_bar, C_tilde, self.D, signal, prev_state, conj_sym=self.conj_sym, bidirectional=self.bidirectional
        )


class S5(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        ssm_size: Optional[int] = None,
        blocks: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bidirectional: bool = False,
        C_init: str = "complex_normal",
        bandlimit: Optional[float] = None,
        conj_sym: bool = False,
        clip_eigs: bool = False,
        step_rescale: float = 1.0,
        discretization: Optional[str] = "bilinear",
    ):
        super().__init__()
        # init ssm
        assert (
            ssm_size % blocks == 0
        ), "blocks should be a factor of ssm_size"

        # init S5SSM
        block_size = ssm_size // blocks
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)


        if conj_sym:
            block_size = block_size // 2
            ssm_size = ssm_size // 2

        Lambda, B, V, B_orig = map(
            lambda v: torch.tensor(v, dtype=torch.complex64),
            (Lambda, B, V, B_orig),
        )

        if blocks > 1:
            Lambda = Lambda[:block_size]
            V = V[:, :block_size]
            Vc = V.conj().T

            # If initializing state matrix A as block-diagonal, put HiPPO approximation
            Lambda = (Lambda * torch.ones((blocks, block_size))).ravel()
            V = torch.block_diag(*([V] * blocks))
            Vinv = torch.block_diag(*([Vc] * blocks))
        else:
            Vinv = V.conj().T

        self.seq = S5SSM(
            H=d_model,
            P=ssm_size,
            Lambda_re_init=Lambda.real,
            Lambda_im_init=Lambda.imag,
            V=V,
            Vinv=Vinv,
            C_init=C_init,
            dt_min=dt_min,
            dt_max=dt_max,
            conj_sym=conj_sym,
            clip_eigs=clip_eigs,
            bidirectional=bidirectional,
            discretization=discretization,
            step_rescale=step_rescale,
            bandlimit= bandlimit)

    def initial_state(self, batch_size: Optional[int] = None):
        return self.seq.initial_state(batch_size)

    def forward(self,
                signal: TensorType["batch_size","seq_length","num_features"],
                prev_state: Optional[TensorType["batch_size","num_states"]]=None,
                ):
        # TODO: include step_rescale?
        if prev_state is None:
            prev_state = self.initial_state(signal.shape[0]).to(signal.device)

        return torch.vmap(lambda s, ps: self.seq(s, prev_state=ps))(
            signal, prev_state
        )


class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class S5Block(torch.nn.Module):
    def __init__(
      self,
      d_model: int,
      ssm_size: int,
      blocks: int = 1,
      dt_min: float = 0.001,
      dt_max: float = 0.1,
      bidirectional: bool = False,
      C_init: str = "complex_normal",
      conj_sym: bool = False,
      clip_eigs: bool = False,
      dropout: float = 0.0,
      activation: str = "gelu",
      prenorm: bool = False,
      batchnorm: bool = False,
      bn_momentum: float = 0.9,
      step_rescale: float = 1.0,
      bandlimit: Optional[float] = None,
      discretization: Optional[str] = "bilinear",
      **kwargs,
    ):
        super().__init__()
        self.batchnorm = batchnorm
        self.activation = activation
        self.prenorm = prenorm

        self.seq = S5(
            d_model=d_model,
            ssm_size=ssm_size,
            blocks=blocks,
            dt_min=dt_min,
            dt_max=dt_max,
            bidirectional=bidirectional,
            C_init=C_init,
            bandlimit=bandlimit,
            conj_sym=conj_sym,
            clip_eigs=clip_eigs,
            step_rescale=step_rescale,
            discretization=discretization,
        )

        if self.activation in ["full_glu"]:
            self.out1 = torch.nn.Linear(d_model)
            self.out2 = torch.nn.Linear(d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = torch.nn.Linear(d_model)

        if self.batchnorm:
            self.norm = torch.nn.BatchNorm1d(d_model, momentum=bn_momentum, track_running_stats=False)
        else:
            self.norm = torch.nn.LayerNorm(d_model)

        self.drop = torch.nn.Dropout(p=dropout)

        self.gelu = F.gelu #if glu else None

    def forward(self,
                x: TensorType["batch_size", "seq_length","num_features"],
                states: Optional[TensorType["batch_size","num_states"]] = None):
        # Standard transfomer-style block with GEGLU/Pre-LayerNorm

        # Prenorm
        skip = x

        if self.prenorm:
            x = self.norm(x)

        if states is None:
            states = self.seq.initial_state(x.shape[0])

        # Apply sequence model
        x, new_state = self.seq(x, states)

        # Apply activation
        if self.activation in ["full_glu"]:
            x = self.drop(self.gelu(x))
            x = self.out1(x) * torch.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(self.gelu(x))
            x = x * torch.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(self.gelu(x))
            x = x * torch.sigmoid(self.out2(x1))
            x = self.drop(x)
        elif self.activation in ["gelu"]:
            x = self.drop(self.gelu(x))
        else:
            raise NotImplementedError(
                "Activation: {} not implemented".format(self.activation))

        x = skip + x
        if not self.prenorm:
            x = self.norm(x)

        return x, new_state


if __name__ == "__main__":

    def tensor_stats(t: torch.Tensor):  # Clone of lovely_tensors for complex support
        return f"tensor[{t.shape}] n={t.shape.numel()}, u={t.mean()}, s={round(t.std().item(), 3)} var={round(t.var().item(), 3)}\n"

        # batch size, input_dim, output_dim

    batch_size = 1
    input_dim = 10
    seq_length = 15

    d_model = input_dim  # dimension of input and output embeddings
    ssm_size = 64
    x = torch.rand([batch_size, seq_length, input_dim])
    model = S5(d_model, ssm_size, )
    print("A", tensor_stats(model.seq.Lambda.data))
    print("B", tensor_stats(model.seq.B.data))
    print("C", tensor_stats(model.seq.C_tilde.data))
    print("C", tensor_stats(model.seq.D.data))

    state = model.initial_state(batch_size)
    res = model(x, prev_state=state)
    print(res[0].shape, res[1].shape)

    # Example 2: (B, L, H) inputs
    x = torch.rand([batch_size, seq_length, input_dim])
    model = S5Block(d_model, ssm_size)
    res = model(x, state)
    print(res[0].shape, res[1].shape)

    exit()
    # Hparam configuration
    hparams = {
        "d_model": 32,
        "ssm_size": 32,
        "block_count": 1,
    }

    model = S5(**hparams)
    # toy data:
    data = {"batch_size": 2,
            "sequence_length":50,
            "input_size": 32,
            }

    (b, t, d) = (data["batch_size"], data['sequence_length'],
                 data['input_size'])

    x = torch.randn((b, t, d))

    # check model
    state = model.initial_state(data['batch_size'])
    res = model(x, prev_state=state)
    print(res[0].shape, res[1].shape)

    outputs = res[0]
    assert outputs.shape == (b, t, data['input_size'])



