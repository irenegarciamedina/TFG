# Random Forest — predicción de glucosa a 40 minutos.
# - Entrena con TRAIN_FILES (20 pacientes)
# - Evalúa con TEST_FILES  (5 pacientes nunca vistos)

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold

from ML.config import (
    TRAIN_FILES, TEST_FILES,
    PLOT_RF, REPORT_FILE, OUTPUT_DIR,
    GLUCOSE_COL, FEATURES, FEATURES_OPCIONALES,
    HORIZON_STEPS, HORIZON_MIN,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES, RF_RANDOM_STATE,
)


# CARGA DE DATOS

def _cargar_grupo(paths: list, etiqueta: str) -> tuple:
    if not paths:
        raise FileNotFoundError(f"No hay ficheros para el grupo '{etiqueta}'.")

    print(f"\n[RF] Cargando {len(paths)} fichero(s) de {etiqueta}...")
    frames = []
    for path in paths:
        df_p = pd.read_csv(path, index_col=0, parse_dates=True)
        df_p["patient_id"] = os.path.basename(path).replace("_preprocessing.csv", "")
        frames.append(df_p)
        print(f"      -> {os.path.basename(path)}: {len(df_p):,} registros")

    df = pd.concat(frames, ignore_index=False)

    features_disp = list(FEATURES)
    for col in FEATURES_OPCIONALES:
        if col in df.columns:
            features_disp.append(col)

    print(f"      -> Total {etiqueta}: {len(df):,} registros | features: {len(features_disp)}")
    return df, features_disp


# CONSTRUCCIÓN DE X, y  (sin cruzar límites entre pacientes)

def _construir_xy(df: pd.DataFrame, features: list) -> tuple:
    X_list, y_list = [], []
    for _, grupo in df.groupby("patient_id", sort=False):
        g    = grupo[GLUCOSE_COL].values
        feat = grupo[features].values
        if len(g) <= HORIZON_STEPS:
            continue
        X_list.append(feat[:-HORIZON_STEPS])
        y_list.append(g[HORIZON_STEPS:])
    return np.vstack(X_list), np.concatenate(y_list)


# ENTRENAMIENTO

def _train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:

    #Entrena el Random Forest con búsqueda de hiperparámetros con GridSearchCV.

    # n_estimators     : número de árboles (300 es suficiente, más no mejora significativamente)
    # max_depth        : None = los nodos crecen hasta hojas puras, limitar reduce varianza
    # min_samples_leaf : mínimo de muestras en hoja que actúa como regularización implícita

    # La validación cruzada usa 3 folds temporales (TimeSeriesSplit no aplica bien aquí
    # porque los datos son de distintos pacientes, se usa KFold shuffle=False para
    # respetar el orden temporal dentro de cada fold).

    print(f"\n[RF] Búsqueda de hiperparámetros (GridSearchCV) con {len(X_train):,} muestras...")

    param_grid = {
        "n_estimators"    : [100, 300],
        "max_depth"       : [None, 20],
        "min_samples_leaf": [5, 10],
    }

    base_model = RandomForestRegressor(
        random_state = RF_RANDOM_STATE,
        n_jobs       = -1,
    )

    cv = KFold(n_splits=3, shuffle=False)   # shuffle=False respeta orden temporal
    gs = GridSearchCV(
        estimator  = base_model,
        param_grid = param_grid,
        cv         = cv,
        scoring    = "neg_root_mean_squared_error",
        n_jobs     = -1,
        verbose    = 1,
        refit      = True,
    )
    gs.fit(X_train, y_train)

    best = gs.best_params_
    print(f"      -> Mejores hiperparámetros: {best}")
    print(f"      -> RMSE CV (train): {-gs.best_score_:.2f} mg/dL")
    print("      -> Reentrenando con todos los datos de train...")
    return gs.best_estimator_


# EVALUACIÓN

def _evaluar(
    model: RandomForestRegressor,
    X_train, X_test, y_train, y_test,
    features: list,
    df_test: pd.DataFrame,
) -> dict:
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    mae_test   = mean_absolute_error(y_test, y_pred_test)
    r2_test    = r2_score(y_test, y_pred_test)

    print(f"\n[RF] Métricas (horizonte {HORIZON_MIN} min | "f"train={len(TRAIN_FILES)} pac., test={len(TEST_FILES)} pac.):")
    print(f"      RMSE train : {rmse_train:.2f} mg/dL")
    print(f"      RMSE test  : {rmse_test:.2f}  mg/dL")
    print(f"      MAE  test  : {mae_test:.2f}  mg/dL")
    print(f"      R²   test  : {r2_test:.4f}")

    importancias_mdi = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    print("\n[RF] Calculando importancia por permutación:")
    perm = permutation_importance(
        model, X_test, y_test,
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
        "df"                : df_test,
        "n_train_pacientes" : len(TRAIN_FILES),
        "n_test_pacientes"  : len(TEST_FILES),
    }



# PUNTO DE ENTRADA

def ejecutar_random_forest() -> dict:
    df_train, features_train = _cargar_grupo(TRAIN_FILES, "TRAIN")
    df_test,  features_test  = _cargar_grupo(TEST_FILES,  "TEST")

    # Usar solo features presentes en ambos grupos
    features = [f for f in features_train if f in df_test.columns]
    faltantes = [f for f in FEATURES if f not in features]
    if faltantes:
        print(f"\n[RF] ⚠ Features ausentes en algún grupo, omitidas: {faltantes}")

    X_train, y_train = _construir_xy(df_train, features)
    X_test,  y_test  = _construir_xy(df_test,  features)

    print(f"\n[RF] Muestras — train: {len(X_train):,}  |  test: {len(X_test):,}")

    rf_model = _train_rf(X_train, y_train)
    metricas = _evaluar(rf_model, X_train, X_test, y_train, y_test, features, df_test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from ML.visualizacion import generar_dashboard_rf, escribir_reporte_rf
    generar_dashboard_rf(metricas, df_test)
    escribir_reporte_rf(metricas)

    return {"rf_model": rf_model, "features": features, "metricas": metricas}