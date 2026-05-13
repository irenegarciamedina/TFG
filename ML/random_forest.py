"""
ML/random_forest.py
---------------------------------------------------------------------------
Random Forest para predicción de glucosa en el horizonte de 40 minutos.
Ahora carga y concatena todos los ficheros de INPUT_FILES (multi-paciente).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

from ML.config import (
    INPUT_FILES, PLOT_RF, REPORT_FILE, OUTPUT_DIR,
    GLUCOSE_COL, FEATURES, FEATURES_OPCIONALES,
    HORIZON_STEPS, HORIZON_MIN,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES, RF_RANDOM_STATE,
    TRAIN_RATIO,
)


# ---------------------------------------------------------------------------
# CARGA DE DATOS  (uno o varios pacientes)
# ---------------------------------------------------------------------------

def cargar_datos() -> tuple[pd.DataFrame, list]:
    """
    Carga y concatena todos los CSV preprocesados de INPUT_FILES.
    Añade una columna 'patient_id' para poder distinguirlos si hace falta.
    """
    if not INPUT_FILES:
        raise FileNotFoundError(
            "No se encontraron ficheros preprocesados. "
            "Ejecuta primero el preprocesamiento (main.py)."
        )

    print(f"\n[RF] Cargando {len(INPUT_FILES)} fichero(s)...")
    frames = []
    for path in INPUT_FILES:
        df_p = pd.read_csv(path, index_col=0, parse_dates=True)
        df_p["patient_id"] = os.path.basename(path).replace("_preprocessing.csv", "")
        frames.append(df_p)
        print(f"      -> {os.path.basename(path)}: {len(df_p):,} registros")

    df = pd.concat(frames, ignore_index=False)
    df = df.sort_index()   # ordenamos por timestamp global

    # Features disponibles (comunes a todos los pacientes)
    features_disponibles = list(FEATURES)
    for col in FEATURES_OPCIONALES:
        if col in df.columns:
            features_disponibles.append(col)
            print(f"      -> Feature opcional encontrada: '{col}' ✓")

    print(f"\n      -> Total registros : {len(df):,}")
    print(f"      -> Features activas: {len(features_disponibles)}: {features_disponibles}")
    return df, features_disponibles


# ---------------------------------------------------------------------------
# CONSTRUCCIÓN DE X, y
# ---------------------------------------------------------------------------

def construir_xy(df: pd.DataFrame, features: list) -> tuple:
    """
    Construye X e y respetando la separación entre pacientes:
    el target de cada fila es la glucosa de HORIZON_STEPS pasos después,
    pero NO cruza el límite entre pacientes distintos.
    """
    X_list, y_list = [], []
    for _, grupo in df.groupby("patient_id", sort=False):
        g = grupo[GLUCOSE_COL].values
        feat = grupo[features].values
        if len(g) <= HORIZON_STEPS:
            continue
        X_list.append(feat[:-HORIZON_STEPS])
        y_list.append(g[HORIZON_STEPS:])

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


# ---------------------------------------------------------------------------
# DIVISIÓN TEMPORAL
# ---------------------------------------------------------------------------

def dividir_temporal(X: np.ndarray, y: np.ndarray) -> tuple:
    """División temporal global (primeros TRAIN_RATIO% → train)."""
    n_train = int(len(X) * TRAIN_RATIO)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


# ---------------------------------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------------------------------

def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    print(f"\n[RF] Entrenando Random Forest ({RF_N_ESTIMATORS} árboles, {len(X_train):,} muestras)...")
    rf = RandomForestRegressor(
        n_estimators     = RF_N_ESTIMATORS,
        max_depth        = RF_MAX_DEPTH,
        min_samples_leaf = RF_MIN_SAMPLES,
        random_state     = RF_RANDOM_STATE,
        n_jobs           = -1,
    )
    rf.fit(X_train, y_train)
    print("      -> Entrenamiento completado")
    return rf


# ---------------------------------------------------------------------------
# EVALUACIÓN
# ---------------------------------------------------------------------------

def evaluar(
    rf: RandomForestRegressor,
    X_train, X_test, y_train, y_test,
    features: list,
    df: pd.DataFrame,
) -> dict:
    y_pred_train = rf.predict(X_train)
    y_pred_test  = rf.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test   = mean_absolute_error(y_test, y_pred_test)
    r2_test    = r2_score(y_test, y_pred_test)

    print(f"\n[RF] Métricas de rendimiento (horizonte {HORIZON_MIN} min, {len(INPUT_FILES)} pacientes):")
    print(f"      RMSE train : {rmse_train:.2f} mg/dL")
    print(f"      RMSE test  : {rmse_test:.2f}  mg/dL")
    print(f"      MAE  test  : {mae_test:.2f}  mg/dL")
    print(f"      R²   test  : {r2_test:.4f}")

    importancias_mdi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    print("\n[RF] Calculando importancia por permutación...")
    perm = permutation_importance(
        rf, X_test, y_test,
        n_repeats=15, random_state=RF_RANDOM_STATE, n_jobs=-1,
    )
    importancias_perm = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
    perm_std          = pd.Series(perm.importances_std,  index=features)

    print("\n[RF] Ranking de importancia (permutación):")
    for feat, val in importancias_perm.items():
        barra = "█" * int(val * 400)
        print(f"      {feat:<28} {val:.4f}  {barra}")

    return {
        "rmse_train"        : rmse_train,
        "rmse_test"         : rmse_test,
        "mae_test"          : mae_test,
        "r2_test"           : r2_test,
        "importancias_perm" : importancias_perm,
        "importancias_mdi"  : importancias_mdi,
        "perm_std"          : perm_std,
        "y_test"            : y_test,
        "y_pred_test"       : y_pred_test,
        "features"          : features,
        "df"                : df,
        "n_pacientes"       : len(INPUT_FILES),
    }


# ---------------------------------------------------------------------------
# ORQUESTADOR
# ---------------------------------------------------------------------------

def ejecutar_random_forest() -> dict:
    df, features = cargar_datos()

    faltantes = [f for f in features if f not in df.columns]
    if faltantes:
        print(f"\n[RF] ⚠ Features no encontradas, se omiten: {faltantes}")
        features = [f for f in features if f in df.columns]

    X, y = construir_xy(df, features)
    X_train, X_test, y_train, y_test = dividir_temporal(X, y)
    print(f"\n[RF] Muestras — train: {len(X_train):,}  |  test: {len(X_test):,}")

    rf_model = train_rf(X_train, y_train)
    metricas = evaluar(rf_model, X_train, X_test, y_train, y_test, features, df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from ML.visualizacion import generar_dashboard_rf, escribir_reporte_rf
    generar_dashboard_rf(metricas, df)
    escribir_reporte_rf(metricas)

    return {"rf_model": rf_model, "features": features, "metricas": metricas}