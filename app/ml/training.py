"""Modelado y evaluación (fases 3 y 4 de CRISP-ML(Q)).

Reproduce el pipeline del notebook: TF-IDF + Regresión Logística, partición
agrupada por componente (sin fuga de información) y GridSearch con F1 macro.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from app.config import INNER_SPLITS, OUTER_SPLITS, RANDOM_STATE
from app.ml.data import DatasetError, clean_dataset

PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__sublinear_tf": [True, False],
    "clf__C": [0.25, 0.5, 1.0, 2.0, 5.0],
}
ALGORITHM = "TF-IDF + Logistic Regression (class_weight=balanced)"


@dataclass
class TrainResult:
    pipeline: Pipeline
    classes: list[str]
    best_params: dict
    metrics: dict
    per_class: dict
    confusion: dict
    n_components: int
    extra: dict = field(default_factory=dict)


def _make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def train_model(df_raw: pd.DataFrame) -> TrainResult:
    df = clean_dataset(df_raw)
    if df["category"].nunique() < 2:
        raise DatasetError("Se necesitan al menos 2 categorías con ≥4 componentes distintos para entrenar.")

    X, y, groups = df["text"], df["category"], df["component_id"]

    # División agrupada: ningún componente aparece a la vez en train y test.
    # El fold 0 es el hold-out "oficial" (igual que el notebook); los 4 folds
    # (con GridSearch interno en cada uno = CV anidada) estiman la variabilidad real.
    outer = StratifiedGroupKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = {"accuracy": [], "f1_macro": [], "balanced_accuracy": []}
    holdout = None
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups)):
        assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]), "Fuga de datos entre train y test"
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        grid = GridSearchCV(
            _make_pipeline(), PARAM_GRID,
            cv=StratifiedGroupKFold(n_splits=INNER_SPLITS), scoring="f1_macro", n_jobs=-1,
        )
        grid.fit(X_tr, y_tr, groups=groups.iloc[train_idx])
        y_pred = grid.best_estimator_.predict(X_te)
        fold_scores["accuracy"].append(accuracy_score(y_te, y_pred))
        fold_scores["f1_macro"].append(f1_score(y_te, y_pred, average="macro"))
        fold_scores["balanced_accuracy"].append(balanced_accuracy_score(y_te, y_pred))
        if fold == 0:
            holdout = (grid, X_tr, X_te, y_te, y_pred)

    grid, X_tr, X_te, y_te, y_pred = holdout
    best = grid.best_estimator_
    metrics = {
        "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_te, y_pred, average="macro")), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_te, y_pred)), 4),
        "cv_f1_macro": round(float(grid.best_score_), 4),
        "training_samples": int(len(df)),
        "train_split_samples": int(len(X_tr)),
        "test_samples": int(len(X_te)),
        "nested_cv": {
            "folds": OUTER_SPLITS,
            **{f"{k}_mean": round(float(np.mean(v)), 4) for k, v in fold_scores.items()},
            **{f"{k}_std": round(float(np.std(v)), 4) for k, v in fold_scores.items()},
            "f1_macro_per_fold": [round(float(v), 4) for v in fold_scores["f1_macro"]],
        },
        "majority_baseline_accuracy": round(float(y.value_counts(normalize=True).iloc[0]), 4),
    }
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    labels = list(best.classes_)
    cm = confusion_matrix(y_te, y_pred, labels=labels)

    # Modelo final: reentrenado con el 100 % de los datos limpios
    best.fit(X, y)

    return TrainResult(
        pipeline=best,
        classes=[str(c) for c in best.classes_],
        best_params={k: (list(v) if isinstance(v, tuple) else v) for k, v in grid.best_params_.items()},
        metrics=metrics,
        per_class=report,
        confusion={"labels": labels, "matrix": cm.tolist()},
        n_components=int(groups.nunique()),
    )
