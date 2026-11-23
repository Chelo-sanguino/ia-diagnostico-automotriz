"""Configuración central (variables de entorno con valores por defecto)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "storage"))
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_DIR = DATA_DIR / "models"
BASE_DATASET = BASE_DIR / "data" / "automotive_faults_aktc_obike_et_al.json"
FRONTEND_DIR = BASE_DIR / "frontend"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'app.db'}")

# Parámetros del entrenamiento (idénticos a los del notebook)
RANDOM_STATE = 42
MIN_COMPONENTS_PER_CLASS = 4   # clases con menos componentes distintos se descartan
OUTER_SPLITS = 4               # 1 de 4 grupos => ~25 % de componentes para test
INNER_SPLITS = 3               # CV agrupada dentro del GridSearch
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

for _d in (DATA_DIR, UPLOAD_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
