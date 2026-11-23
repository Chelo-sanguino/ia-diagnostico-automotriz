"""Carga, validación y preparación de datos (fases 1 y 2 de CRISP-ML(Q))."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.config import MIN_COMPONENTS_PER_CLASS

REQUIRED_COLUMNS = ["category", "symptoms", "step_1", "result_1", "step_2", "result_2"]
TEXT_COLUMNS = REQUIRED_COLUMNS + ["subcategory"]


class DatasetError(ValueError):
    """El archivo de entrenamiento no cumple el formato esperado."""


def json_to_dataframe(raw: list[dict]) -> pd.DataFrame:
    """Expande el JSON jerárquico original a una tabla (misma lógica del notebook)."""
    rows = []
    for idx, item in enumerate(raw):
        try:
            cat, subcat = item["category"], item["subcategory"]
            syms = item["symptoms"]
            step1, step2 = item["diagnosis_steps"][0], item["diagnosis_steps"][1]
        except (KeyError, IndexError, TypeError) as exc:
            raise DatasetError(
                f"Registro #{idx + 1} inválido: se esperan 'category', 'subcategory', "
                f"'symptoms' y dos 'diagnosis_steps'."
            ) from exc
        variants = syms + [" + ".join(syms)] if len(syms) > 1 else syms
        for s_var in variants:
            for r1 in step1["result"]:
                for r2 in step2["result"]:
                    is_fault = (r1 == step1["result"][0]) or (r2 == step2["result"][0])
                    rows.append({
                        "component_id": idx + 1,
                        "category": cat,
                        "subcategory": subcat,
                        "symptoms": s_var,
                        "step_1": step1["step"],
                        "result_1": r1,
                        "step_2": step2["step"],
                        "result_2": r2,
                        "diagnostic_status": "Falla Confirmada" if is_fault else "Sin Falla / Normal",
                    })
    return pd.DataFrame(rows)


def load_base_dataset(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        return json_to_dataframe(json.load(fh))


def load_training_file(path: Path, id_offset: int = 0) -> pd.DataFrame:
    """Lee un archivo subido por el usuario (.json con el formato original o .csv tabular).

    ``id_offset`` desplaza los ``component_id`` para que no choquen con los del
    dataset base ni con los de otros archivos (evita mezclar grupos distintos).
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if not isinstance(raw, list):
                raise DatasetError("El JSON debe ser una lista de componentes.")
            df = json_to_dataframe(raw)
        elif suffix == ".csv":
            df = pd.read_csv(path, encoding="utf-8")
        else:
            raise DatasetError("Formato no soportado: use .csv o .json")
    except (json.JSONDecodeError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetError(f"No se pudo leer el archivo: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(
            f"Faltan columnas obligatorias: {', '.join(missing)}. "
            f"Se requieren: {', '.join(REQUIRED_COLUMNS)} (opcionales: component_id, subcategory)."
        )

    df = df.copy()
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    df = df.dropna(subset=["category", "symptoms"])
    df = df[(df["category"] != "") & (df["symptoms"] != "")]
    for col in ["step_1", "result_1", "step_2", "result_2"]:
        df[col] = df[col].fillna("")
    if df.empty:
        raise DatasetError("El archivo no contiene filas válidas (category y symptoms son obligatorios).")

    # Agrupación por componente: component_id > subcategory > categoría+paso 1
    if "component_id" in df.columns and df["component_id"].notna().all():
        comp = pd.factorize(df["component_id"].astype(str))[0] + 1
    else:
        key_col = df["subcategory"].fillna("") if "subcategory" in df.columns else df["step_1"]
        comp = pd.factorize(df["category"] + "|" + key_col)[0] + 1
    df["component_id"] = comp + id_offset
    if "subcategory" not in df.columns:
        df["subcategory"] = ""
    return df.reset_index(drop=True)


def build_text(symptoms: str, step_1: str, result_1: str, step_2: str, result_2: str) -> str:
    """Texto único que consume el modelo (misma plantilla que en el entrenamiento)."""
    return f"{symptoms or ''} | {step_1 or ''}: {result_1 or ''} | {step_2 or ''}: {result_2 or ''}"


def build_diagnostic_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["symptoms"].fillna("") + " | " + df["step_1"].fillna("") + ": " + df["result_1"].fillna("")
        + " | " + df["step_2"].fillna("") + ": " + df["result_2"].fillna("")
    )


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina clases con pocos componentes (no se pueden separar por grupo) y duplicados."""
    df = df.copy()
    df["text"] = build_diagnostic_text(df)
    comps = df.groupby("category")["component_id"].nunique()
    valid = comps[comps >= MIN_COMPONENTS_PER_CLASS].index
    df = df[df["category"].isin(valid)]
    df = df.drop_duplicates(subset=["text", "category"])
    return df.reset_index(drop=True)
