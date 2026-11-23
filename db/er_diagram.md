# Diagrama entidad-relación

```mermaid
erDiagram
    MODEL_VERSIONS ||--o{ PREDICTIONS : "genera"
    MODEL_VERSIONS ||--o{ MODEL_TRAINING_FILES : "se entrena con"
    TRAINING_FILES ||--o{ MODEL_TRAINING_FILES : "alimenta"

    MODEL_VERSIONS {
        int      id PK
        string   algorithm
        string   source "initial | retrain"
        string   artifact_path
        float    accuracy
        float    f1_macro
        float    balanced_accuracy
        int      training_samples
        int      test_samples
        json     best_params
        json     details "por clase, matriz de confusión, CV"
        boolean  is_active "único índice parcial"
        datetime created_at
    }
    PREDICTIONS {
        int      id PK
        int      model_version_id FK
        text     symptoms
        text     step_1
        text     result_1
        text     step_2
        text     result_2
        string   predicted_category
        float    confidence
        json     top_predictions
        datetime created_at
    }
    TRAINING_FILES {
        int      id PK
        string   original_name
        string   stored_path
        string   file_format "csv | json"
        int      rows_count
        int      categories_count
        datetime uploaded_at
    }
    MODEL_TRAINING_FILES {
        int model_version_id PK, FK
        int training_file_id PK, FK
    }
```

Script de creación: [`schema.sql`](schema.sql) (SQLite; comentarios indican los tipos equivalentes en PostgreSQL).
La aplicación crea las mismas tablas al arrancar mediante SQLAlchemy (`app/models.py`).
