import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from bestatml import BestatClassifier


class FitMeanTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_ = float(np.nanmean(np.asarray(X)[:, 0]))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[:, 0] = X[:, 0] - self.mean_
        return X


def test_preprocessing_is_inside_cv_pipeline(small_classification_data):
    X, y = small_classification_data
    model = BestatClassifier(
        cv=4,
        tune_hyperparameters=False,
        max_models=2,
        preprocessing=FitMeanTransformer(),
        random_state=42,
    )
    model.fit(X, y)
    # The custom transformer is cloned into each candidate Pipeline rather than globally fitted before CV.
    for pipeline in model.fitted_models_.values():
        assert pipeline.named_steps["preprocessor"] is not model.preprocessing
        assert hasattr(pipeline.named_steps["preprocessor"], "fit")


def test_oof_predictions_have_full_coverage(small_classification_data):
    X, y = small_classification_data
    model = BestatClassifier(cv=4, tune_hyperparameters=False, max_models=2, random_state=42)
    model.fit(X, y)
    for pred in model.oof_predictions_.values():
        assert len(pred) == len(X)
        assert np.isfinite(pred).all()
