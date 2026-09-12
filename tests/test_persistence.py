import numpy as np

from bestatml import BestatClassifier


def test_save_load_roundtrip(tmp_path, small_classification_data):
    X, y = small_classification_data
    model = BestatClassifier(cv=3, tune_hyperparameters=False, max_models=2, random_state=42)
    model.fit(X, y)
    path = tmp_path / "model.bestat"
    model.save(path)
    loaded = BestatClassifier.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
    np.testing.assert_allclose(model.predict_proba(X), loaded.predict_proba(X))
