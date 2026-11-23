import os
import tempfile

# Aislar el almacenamiento de pruebas ANTES de importar la app
_tmp = tempfile.mkdtemp(prefix="ia_tests_")
os.environ["DATA_DIR"] = _tmp
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:  # dispara lifespan => entrena el modelo inicial
        yield c
