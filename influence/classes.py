from typing import Callable

import numpy as np

from .explainers import LOO
from .explainers import AKI
# from .explainers import Random
# from .explainers import SubSample


class InfluenceEstimator(object):
    """
    Wrapper for different influence-estimation methods.
        - These methods are model agnostic.
    """
    def __init__(self, method='boostin', params={}, logger=None):
        if method == 'loo':
            self.explainer_ = LOO(**params, logger=logger)
        elif method == 'aki':
            self.explainer_ = AKI(**params, logger=logger)
        # elif method == 'random':
        #     self.explainer_ = Random(**params, logger=logger)
        # elif method == 'subsample':
        #     self.explainer_ = SubSample(**params, logger=logger)
        else:
            raise ValueError(f'Unknown method {method}')

    def fit(self, model, X: np.array, y: np.array, objective: str, loss_fn: Callable):
        """
        - Convert model to internal standardized tree structures.
        - Perform any initialization necessary for the chosen explainer.

        Input
            model: tree ensemble.
            X: 2d array of train data.
            y: 1d array of train targets.
            objective: str, objective function.
            n_class: int, number of classes.

        Return
            Fitted explainer.
        """
        return self.explainer_.fit(model=model, X=X, y=y, objective=objective, loss_fn=loss_fn)
    
    def get_local_influence(self, X: np.array, y: np.array):
        """
        - Compute influence of each training instance on the test loss.

        Input
            X: 2d array of test examples.
            y: 1d array of test targets.
                Could be the actual label or the predicted label depending on the explainer.

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Arrays are returned in the same order as the training data.
        """
        return self.explainer_.get_local_influence(X=X, y=y)
