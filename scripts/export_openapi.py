"""Exporta el documento Swagger/OpenAPI a docs/openapi.json (sin arrancar el servidor)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402

out = ROOT / "docs" / "openapi.json"
out.write_text(json.dumps(app.openapi(), indent=2, ensure_ascii=False), encoding="utf-8")
print(f"OpenAPI exportado a {out}")
