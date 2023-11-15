import numpy as np

from sklearn.metrics import precision_score, recall_score
from typeguard import typechecked

from typing import Union, Optional

@typechecked
def get_precision_recall(
        true_classes: np.ndarray,
        pred_classes: np.ndarray,
        background: Union[int, None] = 0,
        n_classes: Optional[int] = None
) -> dict:
    """Compute precision and recall for classifier.

    Parameters
    ----------
    true_classes : array-like
        entries should be in [0, K-1] where K is the number of classes
    pred_classes : array-like
        entries should be in [0, K-1] where K is the number of classes
    background : int or NoneType
        defines the background class that identifies points with no supervised label; these time
        points are omitted from the precision and recall calculations; if NoneType, no background
        class is utilized
    n_classes : int, optional
        total number of non-background classes; if NoneType, will be inferred from true classes

    Returns
    -------
    dict:
        'precision' (array-like): precision for each class (including background class)
        'recall' (array-like): recall for each class (including background class)

    """

    assert true_classes.shape[0] == pred_classes.shape[0]

    # find all data points that are not background
    if background is not None:
        assert background == 0  # need to generalize
        obs_idxs = np.where(true_classes != background)[0]
    else:
        obs_idxs = np.arange(true_classes.shape[0])

    if n_classes is None:
        n_classes = len(np.unique(true_classes[obs_idxs]))

    # set of labels to include in metric computations
    if background is not None:
        labels = np.arange(1, n_classes + 1)
    else:
        labels = np.arange(n_classes)

    precision = precision_score(
        true_classes[obs_idxs], pred_classes[obs_idxs],
        labels=labels, average=None, zero_division=0)
    recall = recall_score(
        true_classes[obs_idxs], pred_classes[obs_idxs],
        labels=labels, average=None, zero_division=0)

    # replace 0s with NaNs for classes with no ground truth
    # for n in range(precision.shape[0]):
    #     if precision[n] == 0 and recall[n] == 0:
    #         precision[n] = np.nan
    #         recall[n] = np.nan

    # compute f1
    p = precision
    r = recall
    f1 = 2 * p * r / (p + r + 1e-10)
    return {'precision': p, 'recall': r, 'f1': f1}

