"""Esquemas Pydantic: contrato de la API (se reflejan en Swagger/OpenAPI)."""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    symptoms: str = Field(..., min_length=3, max_length=500, description="Síntoma(s) observados.",
                          examples=["ABS warning light on"])
    step_1: str = Field("", max_length=300, description="Primer paso de diagnóstico realizado.",
                        examples=["Check ABS fuse"])
    result_1: str = Field("", max_length=300, description="Resultado del primer paso.", examples=["Blown"])
    step_2: str = Field("", max_length=300, description="Segundo paso de diagnóstico.",
                        examples=["Inspect wiring to ABS module"])
    result_2: str = Field("", max_length=300, description="Resultado del segundo paso.",
                          examples=["Good wiring"])


class TopPrediction(BaseModel):
    category: str
    probability: float


class PredictResponse(BaseModel):
    id: int
    predicted_category: str = Field(..., description="Sistema del vehículo más probable.")
    confidence: float = Field(..., ge=0, le=1, description="Probabilidad de la clase predicha.")
    top_predictions: list[TopPrediction]
    model_version: int
    created_at: datetime


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    symptoms: str
    step_1: str
    result_1: str
    step_2: str
    result_2: str
    predicted_category: str
    confidence: float
    top_predictions: list[TopPrediction]
    model_version_id: int


class PredictionHistory(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[PredictionOut]


class ModelVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    algorithm: str
    source: str
    accuracy: float
    f1_macro: float
    balanced_accuracy: float
    training_samples: int
    test_samples: int
    best_params: dict[str, Any]
    is_active: bool
    created_at: datetime


class MetricsOut(BaseModel):
    model_version: int
    algorithm: str
    created_at: datetime
    accuracy: float
    f1_macro: float
    balanced_accuracy: float
    training_samples: int
    test_samples: int
    best_params: dict[str, Any]
    classes: list[str]
    nested_cv: Optional[dict[str, Any]] = Field(None, description="Media y desviación en 4 particiones agrupadas.")
    majority_baseline_accuracy: Optional[float] = None
    per_class: dict[str, Any] = Field(..., description="precision / recall / f1 / support por clase.")
    confusion_matrix: dict[str, Any] = Field(..., description="{labels: [...], matrix: [[...]]}")


class RetrainRequest(BaseModel):
    include_uploaded: bool = Field(True, description="Incluir los archivos de entrenamiento subidos.")
    activate: bool = Field(True, description="Activar la nueva versión al terminar.")


class TrainingFileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    original_name: str
    file_format: str
    rows_count: int
    categories_count: int
    uploaded_at: datetime


class RetrainResponse(BaseModel):
    message: str
    activated: bool
    new_model: ModelVersionOut
    previous_active_model: Optional[ModelVersionOut] = None
    files_used: list[TrainingFileOut]


class ExampleOut(BaseModel):
    category: str
    subcategory: str
    symptoms: str
    step_1: str
    result_1: str
    step_2: str
    result_2: str


class ErrorOut(BaseModel):
    detail: str
