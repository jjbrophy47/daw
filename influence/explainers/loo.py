import time
import joblib

import numpy as np
from sklearn.base import clone
from sklearn.utils import check_array, check_consistent_length

from .base import Explainer
from . import util


class LOO(Explainer):
    """
    Leave-one-out influence explainer.
    Retrains the model for each training example.

    Local-Influence Semantics
        - Inf.(x_i, x_t) := L(y_t, f_{w/o x_i}(x_t)) - L(y_t, f(x_t))
        - Pos. value means removing x_i increases loss (adding x_i decreases loss, helpful).
        - Neg. value means removing x_i decreases loss (adding x_i increases loss, harmful).

    Note
        - Model agnostic.
        - Supports parallelization.
    """
    def __init__(self, n_jobs=-1, logger=None):
        """
        Input
            n_jobs: int, No. processes to run in parallel.
                -1 means use the no. of available CPU cores.
            logger: object, If not None, output to logger.
        """
        self.n_jobs = n_jobs
        self.logger = logger

    def fit(self, model, X, y, objective, loss_fn):
        """
        - Fit one model with for each training example,
            with that training example removed.

        Note
            - Very memory intensive to save all models,
                may have to switch to a streaming approach.

        Input
            model: tree ensemble.
            X: training data.
            y: training targets.
            objective: str, objective function to use.
            loss_fn: function, loss function.
        """
        super().fit(model, X, y)
        self.loss_fn_ = loss_fn
        self.objective_ = objective
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # select no. processes to run in parallel
        if self.n_jobs == -1:
            n_jobs = joblib.cpu_count()
        else:
            assert self.n_jobs >= 1
            n_jobs = min(self.n_jobs, joblib.cpu_count())

        self.n_jobs_ = n_jobs
        self.original_model_ = model

        return self

    def get_local_influence(self, X, y):
        """
        - Compute influence of each training instance on each test loss.

        Input
            X: 2d array of test data.
            y: 1d array of test targets

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Arrays are returned in the same order as the training data.
        """
        y = check_array(y, ensure_2d=False)
        X = check_array(X, ensure_2d=True)
        check_consistent_length(X, y)
        return self._run_loo(X_test=X, y_test=y)

    # private
    def _run_loo(self, X_test, y_test):
        """
        - Retrain model for each tain example and measure change in train/test loss.

        Return
            - 2d array of average marginals, shape=(no. train, 1 or X_test.shape[0]).
                * Arrays are returned in the same order as the traing data.
        """
        n_train = self.X_train_.shape[0]

        start = time.time()
        if self.logger:
            self.logger.info('\n[INFO] computing LOO values...')
            self.logger.info(f'[INFO] no. cpus: {self.n_jobs_:,}...')

        # fit each model in parallel
        with joblib.Parallel(n_jobs=self.n_jobs_) as parallel:
            original_loss = _get_loss(model=self.original_model_, X=X_test, y=y_test,
                objective=self.objective_, loss_fn=self.loss_fn_)  # (n_test,)
            influence = np.zeros((n_train, X_test.shape[0]), dtype=util.dtype_t)  # (n_train, n_test)

            # trackers
            fits_completed = 0
            fits_remaining = n_train

            # get number of fits to perform for this iteration
            while fits_remaining > 0:
                n = min(100, fits_remaining)

                results = parallel(joblib.delayed(_run_iteration)
                    (self.original_model_, self.X_train_, self.y_train_,
                    train_idx, X_test, y_test, self.objective_, self.loss_fn_,
                    original_loss) for train_idx in range(fits_completed, fits_completed + n))

                # synchronization barrier
                results = np.vstack(results)  # shape=(n, n_test)
                influence[fits_completed: fits_completed+n] = results

                fits_completed += n
                fits_remaining -= n

                if self.logger:
                    cum_time = time.time() - start
                    self.logger.info(f'[INFO] fits: {fits_completed:,} / {n_train:,}'
                                     f', cum. time: {cum_time:.3f}s')

        return influence


def _run_iteration(model, X_train, y_train, train_idx, X_test, y_test,
    objective, loss_fn, original_loss):
    """
    Fit model after leaving out the specified `train_idx` train example.

    Input
        model: tree ensemble.
        X_train: 2d array of training data.
        y_train: 1d array of training targets.
        train_idx: int, index of training example to remove.
        X_test: 2d array of test data.
        y_test: 1d array of test targets.
        objective: function, loss function.
        loss_fn: function, loss function.
        original_loss: 1d array of shape=(n_test,).

    Return
        - 1d array of shape=(X_test.shape[0],) or single float.

    Note
        - Parallelizable method.
    """
    new_X = np.delete(X_train, train_idx, axis=0)
    new_y = np.delete(y_train, train_idx)
    new_model = clone(model).fit(new_X, new_y)

    start = time.time()
    loss = _get_loss(model=new_model, X=X_test, y=y_test, objective=objective, loss_fn=loss_fn)  # shape=(X_test.shape[0],)
    influence = loss - original_loss # shape=(X_test.shape[0],)
    inf_time = time.time() - start

    return influence


def _get_loss(model, X, y, objective, loss_fn):
    """
    Input
        model: predictor.
        X: 2d array of test data.
        y: 1d array of test targets.
        objective: str, 'classification' or 'regression'.
        loss_fn: function, loss function.

    Return
        - 1d array of individual losses of shape=(X.shape[0],).

    Note
        - Parallelizable method.
    """
    if objective == 'regression':
        y_pred = model.predict(X)  # shape=(X.shape[0])
    elif objective == 'binary':
        y_pred = model.predict_proba(X)[:, 1]  # 1d arry of pos. probabilities, shape=(X.shape[0],)
    else:
        assert objective == 'multiclass'
        y_pred = model.predict_proba(X)  # shape=(X.shape[0], no. class)

    result = loss_fn(y, y_pred)  # shape(X.shape[0],) or single float
    return result
