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
