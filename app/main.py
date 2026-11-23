"""Punto de entrada FastAPI: API REST + interfaz web estática."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import models  # noqa: F401  (registra las tablas)
from app.config import FRONTEND_DIR
from app.database import Base, SessionLocal, engine
from app.routers import api
from app.services import ensure_initial_model

DESCRIPTION = """
Servicio de IA que **clasifica el sistema del vehículo afectado** (ABS, refrigeración, combustible, …)
a partir de los síntomas y de los resultados de dos pasos de diagnóstico.

Metodología: **CRISP-ML(Q)**. Modelo: TF-IDF + Regresión Logística, evaluado con partición
agrupada por componente (sin fuga de información).
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        ensure_initial_model(db)  # primer arranque: entrena con el dataset base
    yield


app = FastAPI(
    title="API de Diagnóstico Automotriz con IA", version="1.0.0", description=DESCRIPTION, lifespan=lifespan,
    openapi_tags=[
        {"name": "Predicción", "description": "Clasificar y consultar el historial."},
        {"name": "Modelo", "description": "Métricas, versiones y reentrenamiento."},
        {"name": "Entrenamiento", "description": "Archivos de entrenamiento."},
        {"name": "Sistema", "description": "Salud del servicio."},
    ],
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(api.router)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/health", tags=["Sistema"], summary="Estado del servicio")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
