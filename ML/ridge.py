import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

    print(f"\n[RIDGE] Cargando {len(paths)} fichero(s) de {etiqueta}...")
    frames = []
    for path in paths:
        df_p = pd.read_csv(path, index_col=0, parse_dates=True)
        df_p["patient_id"] = os.path.basename(path).replace("_preprocessing.csv", "")
        frames.append(df_p)
        print(f"        -> {os.path.basename(path)}: {len(df_p):,} registros")

    df = pd.concat(frames, ignore_index=False)

    features_disp = list(FEATURES)
    for col in FEATURES_OPCIONALES:
        if col in df.columns:
            features_disp.append(col)

    print(f"        -> Total {etiqueta}: {len(df):,} registros | features: {len(features_disp)}")
    return df, features_disp


# COSNTRUCCIÓN DE X, y

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

def _train_ridge(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:

    # Entrenamiento se da con búsqueda de alpha mediante RidgeCV (validación cruzada generalizada)
    # Las features son escaladas para que la regularización L2 actúe de forma equitativa en todas ellas

    # alpha es quien controla la intensidad de la penalización:
    # valor pequeño: el modelo es más complejo y tiene riesgo de sobreajuste
    # valor mayor: los coeficientes son más pequeños y cuenta con mayor regularización

    print(f"\n[RIDGE] Búsqueda de alpha (RidgeCV) con {len(X_train):,} muestras...")

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]

    # RidgeCV elige el mejor alpha por Efficient Leave-One-Out CV
    ridge_cv = RidgeCV(alphas=alphas, fit_intercept=True)
    scaler   = StandardScaler()

    X_scaled = scaler.fit_transform(X_train)
    ridge_cv.fit(X_scaled, y_train)

    print(f"        -> Alpha óptimo: {ridge_cv.alpha_:.4f}")

    # Pipeline para que test se escale automáticamente con los parámetros de train
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=ridge_cv.alpha_, fit_intercept=True)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# EVALUACIÓN

def _evaluar(
    model: Pipeline,
    X_train, X_test, y_train, y_test,
    features: list,
) -> dict:
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    mae_test   = mean_absolute_error(y_test, y_pred_test)
    r2_test    = r2_score(y_test, y_pred_test)

    print(f"\n[RIDGE] Métricas (horizonte {HORIZON_MIN} min | "f"train={len(TRAIN_FILES)} pac., test={len(TEST_FILES)} pac.):")
    print(f"        RMSE train : {rmse_train:.2f} mg/dL")
    print(f"        RMSE test  : {rmse_test:.2f}  mg/dL")
    print(f"        MAE  test  : {mae_test:.2f}  mg/dL")
    print(f"        R²   test  : {r2_test:.4f}")

    # Coeficientes normalizados como proxy de importancia
    coefs = model.named_steps["ridge"].coef_
    importancias = pd.Series(
        np.abs(coefs), index=features
    ).sort_values(ascending=False)

    print("\n[RIDGE] Coeficientes (|β| normalizados — proxy de importancia):")
    for feat, val in importancias.items():
        barra = "█" * int(val / importancias.max() * 30)
        print(f"        {feat:<28} {val:.4f}  {barra}")

    return {
        "rmse_train"   : rmse_train,
        "rmse_test"    : rmse_test,
        "mae_test"     : mae_test,
        "r2_test"      : r2_test,
        "importancias" : importancias,
        "coefs_raw"    : pd.Series(coefs, index=features),
        "alpha"        : model.named_steps["ridge"].alpha,
        "y_test"       : y_test,
        "y_pred_test"  : y_pred_test,
        "features"     : features,
        "n_train_pacientes": len(TRAIN_FILES),
        "n_test_pacientes" : len(TEST_FILES),
    }


# PUNTO DE ENTRADA

def ejecutar_ridge() -> dict:
    df_train, features_train = _cargar_grupo(TRAIN_FILES, "TRAIN")
    df_test,  features_test  = _cargar_grupo(TEST_FILES,  "TEST")

    features  = [f for f in features_train if f in df_test.columns]
    faltantes = [f for f in FEATURES if f not in features]
    if faltantes:
        print(f"\n[RIDGE] ⚠ Features ausentes en algún grupo, omitidas: {faltantes}")

    X_train, y_train = _construir_xy(df_train, features)
    X_test,  y_test  = _construir_xy(df_test,  features)

    print(f"\n[RIDGE] Muestras — train: {len(X_train):,}  |  test: {len(X_test):,}")

    ridge_model = _train_ridge(X_train, y_train)
    metricas    = _evaluar(ridge_model, X_train, X_test, y_train, y_test, features)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from ML.visualizacion import generar_dashboard_ridge, escribir_reporte_ridge
    generar_dashboard_ridge(metricas, df_test)
    escribir_reporte_ridge(metricas)

    return {"ridge_model": ridge_model, "features": features, "metricas": metricas}