"""
Test models
"""
from absl.testing import absltest
from absl.testing import parameterized
from action.models.base import all_classifiers


def create_network_cfg():
  classifier_cfg = {
    "classifier_type": "multiclass",
    "variational": False,
    'lambda_weak': 0.5,  # hyperparam on classifying weak (heuristic) labels
    'lambda_strong': 1,  # hyperparam on classifying strong (hand) labels
    'lambda_pred': 0.5,  # hyperparam on one-step-ahead prediction
    'sequence_pad': 16,  # pad batches with extra data for convolutions

  }
  return classifier_cfg

class ModelTestSimple(parameterized.TestCase):
  @parameterized.named_parameters(
    ("cls_segmenter_dtcn", "segmenter", "dtcn"),
  )
  def test_cls_model(self, classifier, backbone):
    backbone_cfg = {
      "dtcn": {
      "backbone": "dtcn",
      "n_hid_layers": 2,
      'n_hid_units': 32,  # hidden units per hidden layer
      'n_lags': 4,  # half-width of temporal convolution window
      'activation': 'lrelu',  # layer nonlinearity
      "bidirectional": True,
      "dropout": 0.1,
      "input_size": 10,
      "num_classes": 2,
      },
    }

    hparams = create_network_cfg()
    hparams['backbone'] = backbone

    hparams.update(backbone_cfg[backbone])

    # build model
    model = all_classifiers[classifier](hparams)

    assert model is not None


if __name__ == '__main__':
  absltest.main()