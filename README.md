# Diagnóstico Automotriz con IA

Servicio de IA (FastAPI) + cliente web que **clasifica el sistema del vehículo afectado**
(ABS, Cooling, Fuel, Electrical, …) a partir de los síntomas y de los resultados de dos pasos de diagnóstico.
Metodología **CRISP-ML(Q)** (cronograma en `docs/Cronograma_CRISP-ML-Q.xlsx`).

- **Tipo de problema:** clasificación supervisada multiclase (11 clases).
- **Modelo:** TF-IDF + Regresión Logística (`class_weight=balanced`), hiperparámetros por GridSearch (F1 macro).
- **Evaluación:** partición agrupada por componente (`StratifiedGroupKFold`) → se mide sobre componentes **no vistos**.

## Ejecutar
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```
- Interfaz web: http://localhost:8000
- Swagger UI: http://localhost:8000/docs  ·  Documento exportado: `docs/openapi.json`

El primer arranque entrena el modelo inicial con `data/` (~10 s) y lo registra en la base de datos.
Con Docker: `docker build -t ia-auto . && docker run -p 8000:8000 -v ia-data:/data ia-auto`.

## Endpoints (`/api/v1`)
| Requisito | Método y ruta |
|---|---|
| Enviar datos y obtener predicción | `POST /predict` |
| Historial de predicciones | `GET /predictions` |
| Métricas del modelo | `GET /metrics` |
| Reentrenar | `POST /retrain` |
| Enviar archivo de entrenamiento | `POST /training-files` (+ `GET /training-files`) |
| Extra: versiones / rollback / ejemplos | `GET /models`, `POST /models/{id}/activate`, `GET /examples` |

Formato del archivo de entrenamiento: `.csv` con `category, symptoms, step_1, result_1, step_2, result_2`
(opcional `subcategory`, `component_id`) o el `.json` con el formato original (`data/automotive_faults_aktc_obike_et_al.json`).
Ejemplo rápido: `data/automotive_faults_tabular.csv`.

## Estructura
```
app/            API (main, routers, services, models, schemas) y ML (app/ml)
frontend/       Cliente web (HTML/CSS/JS sin dependencias)
db/             schema.sql y diagrama ER (er_diagram.md / .png)
docs/           openapi.json, cronograma, matriz de confusión
notebooks/      Experimentación (fases 1-4 de CRISP-ML(Q))
tests/          pytest (API, BD, validaciones)
```

## Resultados (componentes no vistos)
| Métrica | Hold-out | CV agrupada anidada (4 folds) |
|---|---|---|
| Accuracy | 0.493 | 0.543 ± 0.080 |
| F1 macro | 0.489 | 0.457 ± 0.085 |
| Balanced accuracy | 0.501 | 0.489 ± 0.075 |

Referencia: azar ≈ 0.09 y clase mayoritaria = 0.159 de accuracy. Las clases con pocos componentes
(Steering, Wheels & Tires) se excluyen por no poder separarse por grupo.
