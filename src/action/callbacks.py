from pytorch_lightning.callbacks import Callback
from lightning.pytorch.utilities import grad_norm

class GradDebugCallback(Callback):
    def on_before_optimizer_step(self, *args):

        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        model = args[1]
        norms = gradient_norm(model)
        self.log("model/grad_norm", norms)
        print('grad norm', norms, flush=True)

class GradNormCallbackSplit(Callback):
    """
  Logs the gradient norm.
  Edited from https://github.com/Lightning-AI/lightning/issues/1462
  """

    #def on_train_batch_end(self, *args, **kwargs):
    #    model = args[1]

    def on_after_backward(self, *args, **kwargs):
        model = args[1]
        model.log("model/grad_norm", gradient_norm(model))
        has_models = getattr(model, "models", 0)
        if has_models != 0:
            for model_idx, model_ in enumerate(model.models):
                model.log(f"model/grad_norm_{model_idx}",
                          gradient_norm(model_))


class GradNormCallback(Callback):
    """
  Logs the gradient norm.
  Edited from https://github.com/Lightning-AI/lightning/issues/1462
  """

    def on_after_backward(self, trainer, model, outputs):
        model.log("my_model/grad_norm", gradient_norm(model))


def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item()**2
    total_norm = total_norm**(1. / 2)
    return total_norm