# BestatML

**Automated, leakage-safe ensemble machine learning for Python.**

BestatML provides sklearn-compatible `BestatClassifier` and `BestatRegressor` estimators that combine automatic preprocessing, heterogeneous model selection, cross-validation, out-of-fold prediction generation, tuning, stacking, and blending behind a simple API.

> BestatML is designed to provide strong automated baselines and competitive ensembles. It does not claim that an ensemble will outperform every individual model on every dataset.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Automatic Preprocessing](#automatic-preprocessing)
- [Model Inspection](#model-inspection)
- [Configuration](#configuration)
- [Optional Boosting Libraries](#optional-boosting-libraries)
- [Saving and Loading](#saving-and-loading)
- [Extending the Model Registry](#extending-the-model-registry)
- [sklearn Compatibility](#sklearn-compatibility)
- [Error Handling](#error-handling)
- [Development](#development)
- [Documentation](#documentation)
- [License](#license)

## Features

- Simple sklearn-style `fit`, `predict`, `score`, and `predict_proba` APIs
- Leakage-safe preprocessing inside CV pipelines
- Automatic numeric/categorical pandas handling
- Missing-value imputation and unknown-safe one-hot encoding
- Selective scaling for linear models
- Optional low-variance filtering
- Diverse linear, tree, forest, and gradient-boosting candidates
- Genuine OOF predictions for comparison and meta-learning
- Weighted blending, voting, and stacking with safe fallbacks
- Optional CatBoost, XGBoost, and LightGBM integrations
- Permutation-based feature importance
- Structured training summaries and learned-model inspection
- Reproducible random-state handling
- Versioned persistence with artifact trust warnings
- Extensible model registry

## Installation

Once published to PyPI:

```bash
python -m pip install bestatml
```

Optional CatBoost/XGBoost/LightGBM integrations:

```bash
python -m pip install "bestatml[boosting]"
```

Development install from GitHub/source:

```bash
git clone <your-repository-url>
cd BestatML
python -m pip install -e ".[dev]"
```

Replace `<your-repository-url>` with the actual repository URL when the project is hosted publicly.

## Quick Start

### Classification

```python
from bestatml import BestatClassifier

model = BestatClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(model)
```

### Regression

```python
from bestatml import BestatRegressor

model = BestatRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

The default workflow evaluates multiple model families with CV, generates OOF predictions, selects a compact model set, constructs an ensemble, and refits the selected pipelines on the complete training data.

## Automatic Preprocessing

With a pandas DataFrame, BestatML detects numeric and categorical features and learns transformations inside each CV training fold. Numeric data uses robust median imputation by default; categorical/object/bool data can use most-frequent imputation and one-hot encoding with `handle_unknown="ignore"`.

For linear candidates, scaling can be enabled selectively. Tree-based candidates do not need feature scaling. Optional variance filtering is also fitted inside the pipeline, so it does not inspect validation folds early.

NumPy arrays and supported scipy sparse matrices are also accepted.

## Model Inspection

After fitting:

```python
print(model)
print(model.summary())
print(model.model_scores_)
print(model.selected_models_)
print(model.ensemble_method_)
```

Useful learned attributes include `classes_` for classification, `n_features_in_`, `feature_names_in_`, `best_model_`, `fitted_models_`, `oof_predictions_`, `ensemble_weights_`, `meta_learner_`, and `training_summary_`.

### Feature importance

```python
importance = model.get_feature_importance(
    X_test,
    y_test,
    n_repeats=5,
)
print(importance.head(10))
```

The result is a pandas table with `feature`, `importance`, and `std` columns. This uses permutation importance because the final ensemble may combine different model families.

## Configuration

Important decisions remain configurable while sensible defaults keep the common path simple:

```python
model = BestatClassifier(
    cv=5,
    scoring="roc_auc",
    ensemble_method="stacking",
    tune_hyperparameters=True,
    max_models=6,
    random_state=42,
    search_budget="balanced",
)
```

Search budgets provide a simple resource trade-off:

- `lightweight` — smallest tuning budget
- `balanced` — practical default
- `aggressive` — broader tuning

Advanced users can also control `model_families`, disable tuning, choose an explicit ensemble method, supply a custom sklearn preprocessing transformer, and supply a custom sklearn-compatible meta learner.

## Optional Boosting Libraries

CatBoost, XGBoost, and LightGBM are optional. The core package does not require them for import or baseline operation.

Install the optional group with:

```bash
python -m pip install "bestatml[boosting]"
```

Automatically discovered optional candidates are skipped when their packages are unavailable. When an optional family is explicitly requested and unavailable, BestatML raises a dedicated dependency error with an installation hint.

## Saving and Loading

```python
model.save("model.bestat")
loaded = BestatClassifier.load("model.bestat")

predictions = loaded.predict(X_test)
```

Artifacts contain fitted model state, ensemble state, learned metadata, configuration, and artifact-format metadata.

**Security:** persistence uses `joblib`. Only load artifacts you trust because Python object deserialization can execute code.

## Extending the Model Registry

Advanced users can register sklearn-compatible candidates without modifying the trainer:

```python
from bestatml.models import ModelSpec, register_model
from sklearn.linear_model import LogisticRegression

register_model(ModelSpec(
    name="my_classifier",
    family="linear",
    tasks=("classification",),
    probability=True,
    factory=lambda random_state, n_jobs: LogisticRegression(
        max_iter=2000,
        random_state=random_state,
    ),
))
```

## sklearn Compatibility

Both estimators expose explicit constructor parameters and follow sklearn estimator conventions. They can be cloned and inspected through `get_params()`/`set_params()`, used in sklearn pipelines, and placed inside outer model-selection workflows.

BestatML's own CV remains part of the estimator training lifecycle so learned preprocessing is fitted only where allowed by the relevant fold.

## Error Handling

BestatML validates configuration, training data, target data, and inference schemas before deeper ML operations whenever practical. Important exception types are available from the package root:

```python
from bestatml import BestatMLDataError, BestatMLConfigurationError
```

Messages identify the problem and provide a concrete correction when possible. Wrapped lower-level failures retain their original cause for debugging.

## Development

Create a virtual environment and install the development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the main checks:

```bash
pytest
ruff check .
python -m build
twine check dist/*
```

The tests cover estimator compliance, preprocessing, OOF behavior, ensemble execution, persistence, schema validation, and adversarial leakage behavior.

## Documentation

The README is intentionally a practical introduction rather than the full reference manual. A separate documentation site can provide the exhaustive API reference, configuration reference, architecture notes, and tutorials in the future.

## License

BestatML is released under the MIT License. See [LICENSE](LICENSE).
