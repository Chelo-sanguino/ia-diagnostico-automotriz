import io
import json
import sqlite3
from pathlib import Path

from app.config import BASE_DIR

SAMPLE = {
    "symptoms": "ABS warning light on",
    "step_1": "Check ABS fuse", "result_1": "Blown",
    "step_2": "Inspect wiring to ABS module", "result_2": "Good wiring",
}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


def test_predict_and_history(client):
    r = client.post("/api/v1/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_category"] == "ABS System"
    assert 0 <= body["confidence"] <= 1 and len(body["top_predictions"]) == 3
    hist = client.get("/api/v1/predictions?limit=5").json()
    assert hist["total"] >= 1
    assert hist["items"][0]["id"] == body["id"]
    assert hist["items"][0]["symptoms"] == SAMPLE["symptoms"]


def test_predict_validation(client):
    assert client.post("/api/v1/predict", json={"symptoms": ""}).status_code == 422


def test_metrics(client):
    m = client.get("/api/v1/metrics").json()
    assert 0 < m["f1_macro"] <= 1 and 0 < m["accuracy"] <= 1
    assert "ABS System" in m["classes"]
    assert len(m["confusion_matrix"]["matrix"]) == len(m["classes"])
    assert m["nested_cv"]["folds"] == 4


def test_upload_rejects_bad_files(client):
    r = client.post("/api/v1/training-files", files={"file": ("x.txt", b"hola", "text/plain")})
    assert r.status_code == 422
    r = client.post("/api/v1/training-files", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 422 and "Faltan columnas" in r.json()["detail"]


def test_upload_retrain_and_rollback(client):
    csv = io.BytesIO(
        b"category,subcategory,symptoms,step_1,result_1,step_2,result_2\n"
        b"ABS System,Nuevo sensor,ABS light flickers,Check sensor,Loose,Check wiring,Good\n"
        b"ABS System,Nuevo sensor,ABS light flickers,Check sensor,Tight,Check wiring,Faulty\n"
    )
    up = client.post("/api/v1/training-files", files={"file": ("nuevo.csv", csv, "text/csv")})
    assert up.status_code == 201 and up.json()["rows_count"] == 2
    assert len(client.get("/api/v1/training-files").json()) >= 1

    before = client.get("/api/v1/models").json()
    rt = client.post("/api/v1/retrain", json={"include_uploaded": True, "activate": True})
    assert rt.status_code == 200, rt.text
    new = rt.json()["new_model"]
    assert new["is_active"] and len(rt.json()["files_used"]) == 1
    assert len(client.get("/api/v1/models").json()) == len(before) + 1
    assert client.get("/api/v1/metrics").json()["model_version"] == new["id"]

    # rollback a la versión anterior
    old_id = rt.json()["previous_active_model"]["id"]
    assert client.post(f"/api/v1/models/{old_id}/activate").json()["is_active"]
    assert client.get("/api/v1/metrics").json()["model_version"] == old_id
    assert client.post("/api/v1/models/99999/activate").status_code == 404


def test_examples(client):
    ex = client.get("/api/v1/examples?n=3").json()
    assert len(ex) == 3 and "symptoms" in ex[0]


def test_schema_sql_matches_orm():
    """db/schema.sql debe crear las mismas tablas que el ORM."""
    sql = (BASE_DIR / "db" / "schema.sql").read_text(encoding="utf-8")
    con = sqlite3.connect(":memory:")
    con.executescript(sql)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert {"model_versions", "predictions", "training_files", "model_training_files"} <= tables


def test_openapi_documented(client):
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for p in ["/api/v1/predict", "/api/v1/predictions", "/api/v1/metrics", "/api/v1/retrain", "/api/v1/training-files"]:
        assert p in paths
