"""Modelo de datos relacional (ver db/schema.sql y db/er_diagram.md)."""
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ModelVersion(Base):
    """Cada entrenamiento genera una versión del modelo (control de cambios)."""
    __tablename__ = "model_versions"
    __table_args__ = (
        # como máximo un modelo activo a la vez (índice único parcial)
        Index("ux_model_versions_single_active", "is_active", unique=True,
              sqlite_where=text("is_active = 1"), postgresql_where=text("is_active")),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    algorithm: Mapped[str] = mapped_column(String(120), nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)  # 'initial' | 'retrain'
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    f1_macro: Mapped[float] = mapped_column(Float, nullable=False)
    balanced_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    training_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    test_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    best_params: Mapped[dict] = mapped_column(JSON, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, nullable=False)  # métricas por clase, matriz de confusión, CV
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")
    files: Mapped[list["TrainingFile"]] = relationship(
        secondary="model_training_files", back_populates="models"
    )


class Prediction(Base):
    """Historial: datos enviados y resultado de cada predicción."""
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), nullable=False, index=True)
    symptoms: Mapped[str] = mapped_column(Text, nullable=False)
    step_1: Mapped[str] = mapped_column(Text, nullable=False, default="")
    result_1: Mapped[str] = mapped_column(Text, nullable=False, default="")
    step_2: Mapped[str] = mapped_column(Text, nullable=False, default="")
    result_2: Mapped[str] = mapped_column(Text, nullable=False, default="")
    predicted_category: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    top_predictions: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now, index=True)

    model_version: Mapped[ModelVersion] = relationship(back_populates="predictions")


class TrainingFile(Base):
    """Archivos de entrenamiento enviados por el usuario."""
    __tablename__ = "training_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_format: Mapped[str] = mapped_column(String(10), nullable=False)  # 'csv' | 'json'
    rows_count: Mapped[int] = mapped_column(Integer, nullable=False)
    categories_count: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)

    models: Mapped[list[ModelVersion]] = relationship(
        secondary="model_training_files", back_populates="files"
    )


class ModelTrainingFile(Base):
    """Tabla puente: qué archivos se usaron para entrenar cada versión."""
    __tablename__ = "model_training_files"

    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), primary_key=True)
    training_file_id: Mapped[int] = mapped_column(ForeignKey("training_files.id"), primary_key=True)
