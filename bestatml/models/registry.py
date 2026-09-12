from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Literal

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from bestatml.models.boosting import CompatibleClassifier, CompatibleRegressor

Task = Literal["classification", "regression"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    tasks: tuple[Task, ...]
    factory: Callable[[int | None, int], BaseEstimator]
    search_space: dict[str, list] = field(default_factory=dict)
    dense: bool = False
    probability: bool = False
    optional_package: str | None = None


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec | None = None, **kwargs):
    if spec is None:
        spec = ModelSpec(**kwargs)
    _REGISTRY[spec.name] = spec
    return spec


OPTIONAL_FAMILY_PACKAGES = {
    "catboost": "catboost",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}


def missing_optional_dependencies(task: Task, families: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    missing: list[tuple[str, str]] = []
    requested = set(families)
    for family, package in OPTIONAL_FAMILY_PACKAGES.items():
        if family not in requested:
            continue
        try:
            __import__(package)
        except ImportError:
            missing.append((family, package))
    for spec in _REGISTRY.values():
        if task not in spec.tasks or spec.family not in families or not spec.optional_package:
            continue
        try:
            __import__(spec.optional_package)
        except ImportError:
            item = (spec.family, spec.optional_package)
            if item not in missing:
                missing.append(item)
    return tuple(missing)

def available_models(task: Task | None = None) -> tuple[str, ...]:
    names = []
    for name, spec in _REGISTRY.items():
        if task is None or task in spec.tasks:
            names.append(name)
    return tuple(names)


def get_registry(task: Task, families: tuple[str, ...], random_state: int | None, n_jobs: int) -> list[ModelSpec]:
    out = []
    for spec in _REGISTRY.values():
        if task not in spec.tasks or spec.family not in families:
            continue
        if spec.optional_package:
            if os.getenv("BESTATML_DISABLE_OPTIONAL_MODELS") == "1":
                continue
            try:
                __import__(spec.optional_package)
            except ImportError:
                continue
        out.append(spec)
    return out


def _clf(factory, *, rs, jobs):
    return factory(random_state=rs, n_jobs=jobs)


def _reg(factory, *, rs, jobs):
    return factory(random_state=rs, n_jobs=jobs)


register_model(ModelSpec(
    name="logistic_regression",
    family="linear",
    tasks=("classification",),
    probability=True,
    factory=lambda rs, jobs: LogisticRegression(max_iter=1200, C=1.0, solver="lbfgs", random_state=rs),
    search_space={"model__C": [0.05, 0.2, 1.0, 5.0]},
))
register_model(ModelSpec(
    name="decision_tree_classifier",
    family="tree",
    tasks=("classification",),
    probability=True,
    factory=lambda rs, jobs: DecisionTreeClassifier(random_state=rs, min_samples_leaf=2),
    search_space={"model__max_depth": [None, 4, 8, 14], "model__min_samples_leaf": [1, 2, 5, 10]},
))
register_model(ModelSpec(
    name="random_forest_classifier",
    family="random_forest",
    tasks=("classification",),
    probability=True,
    factory=lambda rs, jobs: RandomForestClassifier(n_estimators=280, random_state=rs, n_jobs=jobs, class_weight="balanced"),
    search_space={"model__max_depth": [None, 8, 14, 20], "model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", "log2", 0.6]},
))
register_model(ModelSpec(
    name="extra_trees_classifier",
    family="tree",
    tasks=("classification",),
    probability=True,
    factory=lambda rs, jobs: ExtraTreesClassifier(n_estimators=320, random_state=rs, n_jobs=jobs, class_weight="balanced"),
    search_space={"model__max_depth": [None, 10, 18], "model__min_samples_leaf": [1, 2, 4], "model__max_features": ["sqrt", 0.7, 1.0]},
))
register_model(ModelSpec(
    name="hist_gradient_boosting_classifier",
    family="boosting",
    tasks=("classification",),
    probability=True,
    dense=True,
    factory=lambda rs, jobs: HistGradientBoostingClassifier(max_iter=220, learning_rate=0.06, random_state=rs),
    search_space={"model__max_depth": [None, 4, 8], "model__learning_rate": [0.03, 0.06, 0.12], "model__max_iter": [120, 220]},
))
register_model(ModelSpec(
    name="linear_regression",
    family="linear",
    tasks=("regression",),
    factory=lambda rs, jobs: LinearRegression(n_jobs=jobs),
))
register_model(ModelSpec(
    name="ridge_regression",
    family="linear",
    tasks=("regression",),
    factory=lambda rs, jobs: Ridge(alpha=1.0),
    search_space={"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
))
register_model(ModelSpec(
    name="elastic_net",
    family="linear",
    tasks=("regression",),
    factory=lambda rs, jobs: ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=3000, random_state=rs),
    search_space={"model__alpha": [0.0001, 0.001, 0.01, 0.1], "model__l1_ratio": [0.1, 0.5, 0.9]},
))
register_model(ModelSpec(
    name="decision_tree_regressor",
    family="tree",
    tasks=("regression",),
    factory=lambda rs, jobs: DecisionTreeRegressor(random_state=rs, min_samples_leaf=2),
    search_space={"model__max_depth": [None, 4, 8, 14], "model__min_samples_leaf": [1, 2, 5, 10]},
))
register_model(ModelSpec(
    name="random_forest_regressor",
    family="random_forest",
    tasks=("regression",),
    factory=lambda rs, jobs: RandomForestRegressor(n_estimators=280, random_state=rs, n_jobs=jobs),
    search_space={"model__max_depth": [None, 8, 14, 20], "model__min_samples_leaf": [1, 2, 4], "model__max_features": [1.0, "sqrt", 0.7]},
))
register_model(ModelSpec(
    name="extra_trees_regressor",
    family="tree",
    tasks=("regression",),
    factory=lambda rs, jobs: ExtraTreesRegressor(n_estimators=320, random_state=rs, n_jobs=jobs),
    search_space={"model__max_depth": [None, 10, 18], "model__min_samples_leaf": [1, 2, 4], "model__max_features": [1.0, 0.7, "sqrt"]},
))
register_model(ModelSpec(
    name="hist_gradient_boosting_regressor",
    family="boosting",
    tasks=("regression",),
    dense=True,
    factory=lambda rs, jobs: HistGradientBoostingRegressor(max_iter=220, learning_rate=0.06, random_state=rs),
    search_space={"model__max_depth": [None, 4, 8], "model__learning_rate": [0.03, 0.06, 0.12], "model__max_iter": [120, 220]},
))


def _register_optional():
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        register_model(ModelSpec(
            name="catboost_classifier", family="catboost", tasks=("classification",), probability=True,
            optional_package="catboost", dense=True,
            factory=lambda rs, jobs: CompatibleClassifier(CatBoostClassifier(iterations=350, depth=6, learning_rate=0.05, verbose=False, random_seed=rs, thread_count=jobs)),
            search_space={"model__depth": [4, 6, 8], "model__learning_rate": [0.03, 0.05, 0.1]},
        ))
        register_model(ModelSpec(
            name="catboost_regressor", family="catboost", tasks=("regression",), optional_package="catboost", dense=True,
            factory=lambda rs, jobs: CompatibleRegressor(CatBoostRegressor(iterations=350, depth=6, learning_rate=0.05, verbose=False, random_seed=rs, thread_count=jobs)),
            search_space={"model__depth": [4, 6, 8], "model__learning_rate": [0.03, 0.05, 0.1]},
        ))
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier, XGBRegressor
        register_model(ModelSpec(
            name="xgboost_classifier", family="xgboost", tasks=("classification",), probability=True, optional_package="xgboost",
            factory=lambda rs, jobs: CompatibleClassifier(XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=rs, n_jobs=jobs, eval_metric="logloss")),
            search_space={"model__max_depth": [3, 6, 9], "model__learning_rate": [0.03, 0.06, 0.1]},
        ))
        register_model(ModelSpec(
            name="xgboost_regressor", family="xgboost", tasks=("regression",), optional_package="xgboost",
            factory=lambda rs, jobs: CompatibleRegressor(XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=rs, n_jobs=jobs)),
            search_space={"model__max_depth": [3, 6, 9], "model__learning_rate": [0.03, 0.06, 0.1]},
        ))
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        register_model(ModelSpec(
            name="lightgbm_classifier", family="lightgbm", tasks=("classification",), probability=True, optional_package="lightgbm",
            factory=lambda rs, jobs: CompatibleClassifier(LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=rs, n_jobs=jobs, verbosity=-1)),
            search_space={"model__num_leaves": [15, 31, 63], "model__learning_rate": [0.03, 0.06, 0.1]},
        ))
        register_model(ModelSpec(
            name="lightgbm_regressor", family="lightgbm", tasks=("regression",), optional_package="lightgbm",
            factory=lambda rs, jobs: CompatibleRegressor(LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=rs, n_jobs=jobs, verbosity=-1)),
            search_space={"model__num_leaves": [15, 31, 63], "model__learning_rate": [0.03, 0.06, 0.1]},
        ))
    except ImportError:
        pass


_register_optional()
