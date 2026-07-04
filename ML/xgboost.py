import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBRegressor

from ML.config import (
    TRAIN_FILES, TEST_FILES,
    OUTPUT_DIR, REPORT_FILE,
    GLUCOSE_COL, FEATURES, FEATURES_OPCIONALES,
    HORIZON_STEPS, HORIZON_MIN,
)


# CARGA DE DATOS

def _cargar_grupo(paths: list, etiqueta: str) -> tuple:
    if not paths:
        raise FileNotFoundError(f"No hay ficheros para el grupo '{etiqueta}'.")

    print(f"\n[XGB] Cargando {len(paths)} fichero(s) de {etiqueta}...")
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

def _train_xgb(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:

    # Entrena el XGBoost con búsqueda de hiperparámetros con GridSearchCV.

    # n_estimators   : número de árboles de boosting
    # max_depth      : profundidad máxima de cada árbol base
    # learning_rate  : tasa de aprendizaje (shrinkage)
    # subsample      : fracción de muestras por árbol (regularización estocástica)

    print(f"\n[XGB] Búsqueda de hiperparámetros (GridSearchCV) con {len(X_train):,} muestras...")

    param_grid = {
        "n_estimators"  : [200, 400],
        "max_depth"     : [4, 6],
        "learning_rate" : [0.05, 0.1],
        "subsample"     : [0.8],
    }

    base_model = XGBRegressor(
        objective    = "reg:squarederror",
        random_state = 42,
        n_jobs       = -1,
        tree_method  = "hist",   # más rápido para datasets medianos/grandes
        verbosity    = 0,
    )

    cv = KFold(n_splits=3, shuffle=False)
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
    return gs.best_estimator_


# EVALUACIÓN

def _evaluar(
    model: XGBRegressor,
    X_train, X_test, y_train, y_test,
    features: list,
) -> dict:
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    mae_test   = mean_absolute_error(y_test, y_pred_test)
    r2_test    = r2_score(y_test, y_pred_test)

    print(f"\n[XGB] Métricas (horizonte {HORIZON_MIN} min | " f"train={len(TRAIN_FILES)} pac., test={len(TEST_FILES)} pac.):")
    print(f"      RMSE train : {rmse_train:.2f} mg/dL")
    print(f"      RMSE test  : {rmse_test:.2f}  mg/dL")
    print(f"      MAE  test  : {mae_test:.2f}  mg/dL")
    print(f"      R²   test  : {r2_test:.4f}")

    # Importancia de features (gain = ganancia media por split)
    importancias = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    print("\n[XGB] Ranking de importancia (gain):")
    for feat, val in importancias.items():
        barra = "█" * int(val * 400)
        print(f"      {feat:<28} {val:.4f}  {barra}")

    return {
        "rmse_train"   : rmse_train,
        "rmse_test"    : rmse_test,
        "mae_test"     : mae_test,
        "r2_test"      : r2_test,
        "importancias" : importancias,
        "y_test"       : y_test,
        "y_pred_test"  : y_pred_test,
        "features"     : features,
        "n_train_pacientes": len(TRAIN_FILES),
        "n_test_pacientes" : len(TEST_FILES),
    }

# PUNTO DE ENTRADA

def ejecutar_xgboost() -> dict:
    df_train, features_train = _cargar_grupo(TRAIN_FILES, "TRAIN")
    df_test,  features_test  = _cargar_grupo(TEST_FILES,  "TEST")

    features   = [f for f in features_train if f in df_test.columns]
    faltantes  = [f for f in FEATURES if f not in features]
    if faltantes:
        print(f"\n[XGB] ⚠ Features ausentes en algún grupo, omitidas: {faltantes}")

    X_train, y_train = _construir_xy(df_train, features)
    X_test,  y_test  = _construir_xy(df_test,  features)

    print(f"\n[XGB] Muestras — train: {len(X_train):,}  |  test: {len(X_test):,}")

    xgb_model = _train_xgb(X_train, y_train)
    metricas  = _evaluar(xgb_model, X_train, X_test, y_train, y_test, features)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from ML.visualizacion import generar_dashboard_xgb, escribir_reporte_xgb
    generar_dashboard_xgb(metricas, df_test)
    escribir_reporte_xgb(metricas)

    return {"xgb_model": xgb_model, "features": features, "metricas": metricas}
