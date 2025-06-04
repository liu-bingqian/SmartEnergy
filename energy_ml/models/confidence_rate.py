import numpy as np
from tensorflow.keras.models import Sequential


from sklearn.ensemble import (
    RandomForestRegressor,
    BaggingRegressor,
    ExtraTreesRegressor
)
def confidence_rate(model, X_valid):
    TREE_MODELS = (
        RandomForestRegressor,
        BaggingRegressor,
        ExtraTreesRegressor
    )

    if isinstance(model, TREE_MODELS):
        preds = np.array([tree.predict(X_valid) for tree in model.estimators_])

    if isinstance(model, Sequential):
        preds = np.array([model(X_valid, training=True) for i in range(100)])

    variances = np.var(preds, axis=0)
    confidence_rate_matrix = 1 / (1 + np.sqrt(np.max(variances, axis=1)))

    return confidence_rate_matrix