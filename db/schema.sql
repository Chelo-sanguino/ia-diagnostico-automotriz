-- =============================================================================
-- Esquema de base de datos - Servicio de IA de diagnóstico automotriz
-- Motor por defecto: SQLite (compatible con PostgreSQL cambiando los tipos
-- indicados en los comentarios). La aplicación también crea estas tablas
-- automáticamente al arrancar (SQLAlchemy: Base.metadata.create_all).
--
-- Uso:  sqlite3 storage/app.db < db/schema.sql
-- =============================================================================
PRAGMA foreign_keys = ON;

-- Versiones del modelo: cada entrenamiento / reentrenamiento genera una fila
CREATE TABLE IF NOT EXISTS model_versions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,   -- PostgreSQL: SERIAL
    algorithm          VARCHAR(120) NOT NULL,
    source             VARCHAR(20)  NOT NULL CHECK (source IN ('initial', 'retrain')),
    artifact_path      VARCHAR(500) NOT NULL,               -- archivo .joblib del pipeline
    accuracy           REAL    NOT NULL,
    f1_macro           REAL    NOT NULL,
    balanced_accuracy  REAL    NOT NULL,
    training_samples   INTEGER NOT NULL,
    test_samples       INTEGER NOT NULL,
    best_params        JSON    NOT NULL,                    -- PostgreSQL: JSONB
    details            JSON    NOT NULL,                    -- métricas por clase, matriz de confusión, CV anidada
    is_active          BOOLEAN NOT NULL DEFAULT 0,          -- solo una versión activa a la vez
    created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Garantiza a nivel de BD que exista como máximo UN modelo activo
CREATE UNIQUE INDEX IF NOT EXISTS ux_model_versions_single_active
    ON model_versions (is_active) WHERE is_active = 1;

-- Historial de predicciones: datos enviados + resultado
CREATE TABLE IF NOT EXISTS predictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version_id   INTEGER NOT NULL REFERENCES model_versions (id),
    symptoms           TEXT    NOT NULL,
    step_1             TEXT    NOT NULL DEFAULT '',
    result_1           TEXT    NOT NULL DEFAULT '',
    step_2             TEXT    NOT NULL DEFAULT '',
    result_2           TEXT    NOT NULL DEFAULT '',
    predicted_category VARCHAR(100) NOT NULL,
    confidence         REAL    NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    top_predictions    JSON    NOT NULL,                    -- [{category, probability}, ...]
    created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_predictions_model_version_id ON predictions (model_version_id);
CREATE INDEX IF NOT EXISTS ix_predictions_created_at       ON predictions (created_at);

-- Archivos de entrenamiento enviados por el usuario
CREATE TABLE IF NOT EXISTS training_files (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    original_name      VARCHAR(255) NOT NULL,
    stored_path        VARCHAR(500) NOT NULL,
    file_format        VARCHAR(10)  NOT NULL CHECK (file_format IN ('csv', 'json')),
    rows_count         INTEGER NOT NULL,
    categories_count   INTEGER NOT NULL,
    uploaded_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Relación N:M: qué archivos intervinieron en el entrenamiento de cada versión
CREATE TABLE IF NOT EXISTS model_training_files (
    model_version_id   INTEGER NOT NULL REFERENCES model_versions (id),
    training_file_id   INTEGER NOT NULL REFERENCES training_files (id),
    PRIMARY KEY (model_version_id, training_file_id)
);
