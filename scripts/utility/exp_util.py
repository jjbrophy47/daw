"""
Utility methods to make epxeriments easier.
"""
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error


def eval_model(model, X_test, y_test, objective, loss_fn, logger):
    """
    Evaluate the given model on the test set.
    """
    res = {}

    if objective == 'regression':
        pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, pred, squared=False)
        loss = loss_fn(y_test, pred).mean()  # MSE
        logger.info(f'RMSE: {rmse:.3f}, loss: {loss:.3f}')
        res['rmse'] = rmse
        res['loss'] = loss
        res['pred'] = pred

    elif objective == 'binary':
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, proba[:, 1])
        loss = loss_fn(y_test, proba).mean()  # mean log loss
        logger.info(f'Acc.: {acc:.3f}, AUC: {auc:.3f}, loss: {loss:.3f}')
        res['acc'] = acc
        res['auc'] = auc
        res['loss'] = loss
        res['pred'] = pred
        res['proba'] = proba

    elif objective == 'multiclass':
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        acc = accuracy_score(y_test, pred)
        loss = loss_fn(y_test, proba).mean()  # mean log loss
        logger.info(f'Acc.: {acc:.3f}, loss: {loss:.3f}')
        res['acc'] = acc
        res['loss'] = loss
        res['pred'] = pred
        res['proba'] = proba

    return res
