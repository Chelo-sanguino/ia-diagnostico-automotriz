"""Endpoints REST del servicio de IA (prefijo /api/v1)."""
from __future__ import annotations

import random
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import schemas, services
from app.config import BASE_DATASET, MAX_UPLOAD_BYTES, UPLOAD_DIR
from app.database import get_db
from app.ml.data import DatasetError, load_base_dataset
from app.models import ModelVersion, Prediction, TrainingFile

router = APIRouter(prefix="/api/v1")


@router.post(
    "/predict", response_model=schemas.PredictResponse, tags=["Predicción"],
    summary="Enviar datos al modelo y obtener la clasificación",
    description="Recibe los síntomas y los resultados de dos pasos de diagnóstico y devuelve el "
                "sistema del vehículo más probable (ABS, Cooling, Fuel, …) con su probabilidad y el top-3. "
                "La consulta queda registrada en el historial.",
)
def predict(body: schemas.PredictRequest, db: Session = Depends(get_db)):
    try:
        row = services.predict_and_log(db, body.model_dump())
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    return schemas.PredictResponse(
        id=row.id, predicted_category=row.predicted_category, confidence=row.confidence,
        top_predictions=row.top_predictions, model_version=row.model_version_id, created_at=row.created_at,
    )


@router.get(
    "/predictions", response_model=schemas.PredictionHistory, tags=["Predicción"],
    summary="Historial de predicciones previas",
    description="Lista paginada (más recientes primero) con los datos enviados y el resultado.",
)
def list_predictions(
    limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0),
    category: str | None = Query(None, description="Filtrar por categoría predicha"),
    db: Session = Depends(get_db),
):
    q = select(Prediction)
    count_q = select(func.count()).select_from(Prediction)
    if category:
        q, count_q = q.where(Prediction.predicted_category == category), count_q.where(Prediction.predicted_category == category)
    items = db.scalars(q.order_by(Prediction.id.desc()).limit(limit).offset(offset)).all()
    return schemas.PredictionHistory(total=db.scalar(count_q) or 0, limit=limit, offset=offset, items=items)


@router.get(
    "/metrics", response_model=schemas.MetricsOut, tags=["Modelo"],
    summary="Métricas de rendimiento del modelo activo",
    description="Accuracy, F1 macro y Balanced Accuracy sobre componentes no vistos, métricas por clase, "
                "matriz de confusión y estimación de variabilidad (CV agrupada anidada).",
)
def metrics(db: Session = Depends(get_db)):
    mv = services.get_active_model(db)
    if mv is None:
        raise HTTPException(503, "No hay un modelo activo")
    d = mv.details
    return schemas.MetricsOut(
        model_version=mv.id, algorithm=mv.algorithm, created_at=mv.created_at, accuracy=mv.accuracy,
        f1_macro=mv.f1_macro, balanced_accuracy=mv.balanced_accuracy, training_samples=mv.training_samples,
        test_samples=mv.test_samples, best_params=mv.best_params, classes=d["classes"],
        nested_cv=d.get("nested_cv"), majority_baseline_accuracy=d.get("majority_baseline_accuracy"),
        per_class=d["per_class"], confusion_matrix=d["confusion_matrix"],
    )


@router.post(
    "/retrain", response_model=schemas.RetrainResponse, tags=["Modelo"],
    summary="Solicitar el reentrenamiento del modelo",
    description="Entrena una nueva versión con el dataset base + los archivos subidos en "
                "`/training-files`. Genera nuevas métricas, guarda la versión y (opcionalmente) la activa. "
                "Tarda unos segundos.",
    responses={422: {"model": schemas.ErrorOut}},
)
def retrain(body: schemas.RetrainRequest = schemas.RetrainRequest(), db: Session = Depends(get_db)):
    try:
        new, previous, files = services.retrain(db, body.include_uploaded, body.activate)
    except DatasetError as exc:
        raise HTTPException(422, str(exc))
    return schemas.RetrainResponse(
        message=f"Modelo v{new.id} entrenado con {new.training_samples} muestras"
                + (" y activado." if body.activate else " (no activado)."),
        activated=new.is_active, new_model=new, previous_active_model=previous, files_used=files,
    )


@router.post(
    "/training-files", response_model=schemas.TrainingFileOut, status_code=201, tags=["Entrenamiento"],
    summary="Enviar un archivo de entrenamiento",
    description="Sube un `.csv` (columnas: category, symptoms, step_1, result_1, step_2, result_2; "
                "opcionales: subcategory, component_id) o un `.json` con el formato original del dataset. "
                "El archivo se valida y queda disponible para el próximo `/retrain`.",
    responses={422: {"model": schemas.ErrorOut}, 413: {"model": schemas.ErrorOut}},
)
def upload_training_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    name = Path(file.filename or "").name
    suffix = Path(name).suffix.lower()
    if suffix not in (".csv", ".json"):
        raise HTTPException(422, "Formato no soportado: use .csv o .json")
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "El archivo supera el límite de 10 MB")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    stored = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe}"
    stored.write_bytes(content)
    try:
        return services.validate_and_store_upload(db, name, stored)
    except DatasetError as exc:
        stored.unlink(missing_ok=True)
        raise HTTPException(422, str(exc))


@router.get("/training-files", response_model=list[schemas.TrainingFileOut], tags=["Entrenamiento"],
            summary="Listar archivos de entrenamiento subidos")
def list_training_files(db: Session = Depends(get_db)):
    return db.scalars(select(TrainingFile).order_by(TrainingFile.id.desc())).all()


@router.get("/models", response_model=list[schemas.ModelVersionOut], tags=["Modelo"],
            summary="Versiones del modelo (control de cambios)")
def list_models(db: Session = Depends(get_db)):
    return db.scalars(select(ModelVersion).order_by(ModelVersion.id.desc())).all()


@router.post("/models/{model_id}/activate", response_model=schemas.ModelVersionOut, tags=["Modelo"],
             summary="Activar una versión anterior del modelo (rollback)",
             responses={404: {"model": schemas.ErrorOut}})
def activate_model(model_id: int, db: Session = Depends(get_db)):
    try:
        return services.activate_version(db, model_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc))


@router.get("/examples", response_model=list[schemas.ExampleOut], tags=["Predicción"],
            summary="Ejemplos reales del dataset para probar la predicción")
def examples(n: int = Query(5, ge=1, le=20)):
    df = load_base_dataset(BASE_DATASET)
    cols = ["category", "subcategory", "symptoms", "step_1", "result_1", "step_2", "result_2"]
    sample = df[cols].drop_duplicates(subset=["category", "subcategory"]).to_dict("records")
    random.shuffle(sample)
    return sample[:n]
