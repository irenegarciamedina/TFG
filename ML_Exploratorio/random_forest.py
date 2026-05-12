import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

from ML_Exploratorio.config import (
    INPUT_FILE, PLOT_RF, REPORT_FILE, OUTPUT_DIR,
    GLUCOSE_COL, FEATURES, FEATURES_OPCIONALES,
    HORIZON_STEPS, HORIZON_MIN,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES, RF_RANDOM_STATE,
    TRAIN_RATIO,
)

def cargar_datos() -> pd.DataFrame:
    print(f"\n[RF] Cargando datos desde: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

    # añade las features opcionales si estas existen

    features_disponibles = list(FEATURES)
    for col in FEATURES_OPCIONALES:
        if col in df.columns:
            features_disponibles.append(col)
            print(f"      -> Feature opcional encontrada: '{col}' ✓")

    print(f"      -> Features activas ({len(features_disponibles)}): {features_disponibles}")
    print(f"      -> Registros cargados: {len(df):,}  |  Período: {df.index.min().date()} → {df.index.max().date()}")
    return df, features_disponibles


def construir_xy(df: pd.DataFrame, features: list) -> tuple:
    X = df[features].values[:-HORIZON_STEPS]
    y = df[GLUCOSE_COL].values[HORIZON_STEPS:]
    return X, y

# predice glucosa [t + HORIZON_STEPS] a partir de t
# forma más directa para evaluar qué variables en el presente predicen la glucosa futura

# df: DataFrame
# features: columnas usadas como predictores
# x: np.ndarray shape (m_muestras, n_feautres)
# y: np.ndarray shape (n_muestras, )


def dividir_temporal(X: np.ndarray, y: np.ndarray) -> tuple:
    n_train = int(len(X) * TRAIN_RATIO)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]

# división para el train
# no se usa train_test_split(shuffle=True) porque se trata de una serie temporal
# si se mezcla se dan fugas de información
# el modelo varía datos del futuro en el entrenamiento y los resultados no serían realistas

def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    print(f"\n[RF] Entrenando Random Forest ({RF_N_ESTIMATORS} árboles)...")
    rf = RandomForestRegressor(
        n_estimators  = RF_N_ESTIMATORS,
        max_depth     = RF_MAX_DEPTH,
        min_samples_leaf = RF_MIN_SAMPLES,
        random_state  = RF_RANDOM_STATE,
        n_jobs        = -1,    # usa todos los núcleos disponibles
    )
    rf.fit(X_train, y_train)
    print(f"      -> Entrenamiento completado  |  OOB score: {'N/A (oob_score=False)'}")
    return rf

# parámetros definidos en el ML_Exploratorio.config

def evaluar(
    rf: RandomForestRegressor,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    features: list,
    df: pd.DataFrame,
) -> dict:
    y_pred_train = rf.predict(X_train)
    y_pred_test  = rf.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test   = mean_absolute_error(y_test, y_pred_test)
    r2_test    = r2_score(y_test, y_pred_test)

    print(f"\n[RF] Métricas de rendimiento (horizonte {HORIZON_MIN} min):")
    print(f"      RMSE train : {rmse_train:.2f} mg/dL")
    print(f"      RMSE test  : {rmse_test:.2f}  mg/dL  ← línea base para la LSTM")
    print(f"      MAE  test  : {mae_test:.2f}  mg/dL")
    print(f"      R²   test  : {r2_test:.4f}")

    # Importancia por Mean Decrease in Impurity
    importancias_mdi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    # Importancia por Permutación (más robusta que MDI)
    print("\n[RF] Calculando importancia por permutación (puede tardar ~30 s)...")
    perm = permutation_importance(
        rf, X_test, y_test,
        n_repeats=15, random_state=RF_RANDOM_STATE, n_jobs=-1,
    )
    importancias_perm = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
    perm_std          = pd.Series(perm.importances_std,  index=features)

    print("\n[RF] Ranking de importancia (permutación — más fiable):")
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
    }

def ejecutar_random_forest() -> dict:
    df, features = cargar_datos()

    # Verificar que las features obligatorias existen en el CSV
    faltantes = [f for f in features if f not in df.columns]
    if faltantes:
        print(f"\n[RF] ⚠ Features no encontradas en el CSV, se omiten: {faltantes}")
        features = [f for f in features if f in df.columns]

    X, y = construir_xy(df, features)
    X_train, X_test, y_train, y_test = dividir_temporal(X, y)
    print(f"\n[RF] Muestras — train: {len(X_train):,}  |  test: {len(X_test):,}")

    rf_model = train_rf(X_train, y_train)
    metricas = evaluar(rf_model, X_train, X_test, y_train, y_test, features, df)

    # Crear carpeta de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from ML_Exploratorio.visualizacion import generar_dashboard_rf, escribir_reporte_rf
    generar_dashboard_rf(metricas, df)
    escribir_reporte_rf(metricas)

    return {"rf_model": rf_model, "features": features, "metricas": metricas}