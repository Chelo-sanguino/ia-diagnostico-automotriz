"""Lógica de negocio: predicción, registro de versiones y reentrenamiento."""
from __future__ import annotations

import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import BASE_DATASET, MODEL_DIR
from app.ml.data import (
    build_text, load_base_dataset, load_training_file,
)
from app.ml.training import ALGORITHM, TrainResult, train_model
from app.models import ModelTrainingFile, ModelVersion, Prediction, TrainingFile

_lock = threading.RLock()
_cache: dict[int, dict] = {}  # model_version_id -> artefacto cargado en memoria
UPLOAD_ID_OFFSET = 10_000     # component_id de cada archivo subido = id_archivo * OFFSET + n


# ---------------------------------------------------------------- versiones --
def get_active_model(db: Session) -> ModelVersion | None:
    return db.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))


def register_version(db: Session, result: TrainResult, source: str,
                     files: list[TrainingFile], activate: bool) -> ModelVersion:
    mv = ModelVersion(
        algorithm=ALGORITHM, source=source, artifact_path="",
        accuracy=result.metrics["accuracy"], f1_macro=result.metrics["f1_macro"],
        balanced_accuracy=result.metrics["balanced_accuracy"],
        training_samples=result.metrics["training_samples"], test_samples=result.metrics["test_samples"],
        best_params=result.best_params,
        details={
            "classes": result.classes, "per_class": result.per_class, "confusion_matrix": result.confusion,
            "nested_cv": result.metrics["nested_cv"], "cv_f1_macro": result.metrics["cv_f1_macro"],
            "majority_baseline_accuracy": result.metrics["majority_baseline_accuracy"],
            "n_components": result.n_components,
        },
        is_active=False,
    )
    db.add(mv)
    db.flush()  # obtiene mv.id
    path = MODEL_DIR / f"model_v{mv.id}.joblib"
    joblib.dump({"pipeline": result.pipeline, "classes": result.classes, "version": mv.id}, path)
    mv.artifact_path = str(path)
    for f in files:
        db.add(ModelTrainingFile(model_version_id=mv.id, training_file_id=f.id))
    if activate:
        db.execute(update(ModelVersion).values(is_active=False))
        mv.is_active = True
    db.commit()
    return mv


def ensure_initial_model(db: Session) -> ModelVersion:
    """Si no hay modelo activo (primer arranque), entrena con el dataset base."""
    with _lock:
        active = get_active_model(db)
        if active and Path(active.artifact_path).exists():
            return active
        result = train_model(load_base_dataset(BASE_DATASET))
        return register_version(db, result, source="initial", files=[], activate=True)


def activate_version(db: Session, version_id: int) -> ModelVersion:
    with _lock:
        mv = db.get(ModelVersion, version_id)
        if mv is None:
            raise LookupError(f"No existe el modelo {version_id}")
        if not Path(mv.artifact_path).exists():
            raise FileNotFoundError("El artefacto del modelo ya no existe en disco")
        db.execute(update(ModelVersion).values(is_active=False))
        mv.is_active = True
        db.commit()
        return mv


# --------------------------------------------------------------- predicción --
def _load_artifact(mv: ModelVersion) -> dict:
    with _lock:
        if mv.id not in _cache:
            _cache.clear()  # solo mantenemos el modelo en uso
            _cache[mv.id] = joblib.load(mv.artifact_path)
        return _cache[mv.id]


def predict_and_log(db: Session, payload: dict) -> Prediction:
    mv = get_active_model(db)
    if mv is None:
        raise RuntimeError("No hay un modelo activo")
    artifact = _load_artifact(mv)
    pipe = artifact["pipeline"]
    text = build_text(payload["symptoms"], payload["step_1"], payload["result_1"],
                      payload["step_2"], payload["result_2"])
    proba = pipe.predict_proba([text])[0]
    order = np.argsort(proba)[::-1][:3]
    classes = pipe.classes_
    top = [{"category": str(classes[i]), "probability": round(float(proba[i]), 4)} for i in order]
    row = Prediction(
        model_version_id=mv.id, **payload,
        predicted_category=top[0]["category"], confidence=top[0]["probability"], top_predictions=top,
    )
    db.add(row)
    db.commit()
    return row


# ------------------------------------------------------------ reentrenamiento --
def retrain(db: Session, include_uploaded: bool, activate: bool) -> tuple[ModelVersion, ModelVersion | None, list[TrainingFile]]:
    with _lock:
        previous = get_active_model(db)
        frames = [load_base_dataset(BASE_DATASET)]
        files: list[TrainingFile] = []
        if include_uploaded:
            files = list(db.scalars(select(TrainingFile).order_by(TrainingFile.id)))
            for f in files:
                frames.append(load_training_file(Path(f.stored_path), id_offset=f.id * UPLOAD_ID_OFFSET))
        df = pd.concat(frames, ignore_index=True)
        result = train_model(df)
        mv = register_version(db, result, source="retrain", files=files, activate=activate)
        return mv, previous, files


def validate_and_store_upload(db: Session, original_name: str, stored_path: Path) -> TrainingFile:
    """Valida el archivo (lanza DatasetError) y lo registra en la BD."""
    df = load_training_file(stored_path)
    rec = TrainingFile(
        original_name=original_name, stored_path=str(stored_path),
        file_format=stored_path.suffix.lower().lstrip("."),
        rows_count=int(len(df)), categories_count=int(df["category"].nunique()),
    )
    db.add(rec)
    db.commit()
    return rec
