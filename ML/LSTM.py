import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from ML.config import (
    TRAIN_FILES, TEST_FILES,
    OUTPUT_DIR, REPORT_FILE,
    GLUCOSE_COL, FEATURES, FEATURES_OPCIONALES,
    HORIZON_STEPS, HORIZON_MIN,
)

# Longitud de la ventana de entrada (pasos pasados usados para predecir).
# 12 pasos × 5 min = 60 min de histórico.
LOOKBACK_STEPS = 12

# Hiperparámetros de entrenamiento
LSTM_UNITS      = 64
DROPOUT_RATE    = 0.2
BATCH_SIZE      = 64
MAX_EPOCHS      = 60
PATIENCE        = 8
LEARNING_RATE   = 1e-3
RANDOM_STATE    = 42


# CARGA DE DATOS

def _cargar_grupo(paths: list, etiqueta: str) -> tuple:
    if not paths:
        raise FileNotFoundError(f"No hay ficheros para el grupo '{etiqueta}'.")

    print(f"\n[LSTM] Cargando {len(paths)} fichero(s) de {etiqueta}...")
    frames = []
    for path in paths:
        df_p = pd.read_csv(path, index_col=0, parse_dates=True)
        df_p["patient_id"] = os.path.basename(path).replace("_preprocessing.csv", "")
        frames.append(df_p)
        print(f"       -> {os.path.basename(path)}: {len(df_p):,} registros")

    df = pd.concat(frames, ignore_index=False)

    features_disp = list(FEATURES)
    for col in FEATURES_OPCIONALES:
        if col in df.columns:
            features_disp.append(col)

    print(f"       -> Total {etiqueta}: {len(df):,} registros | features: {len(features_disp)}")
    return df, features_disp


# CONSTRUCCIÓN DE SECUENCIAS. VENTANAS DESLIZANTES SIN CRUZAR LÍMITES ENTRE PACIENTES

def _construir_secuencias(df: pd.DataFrame, features: list) -> tuple:

    # Para cada paciente construye ventanas deslizantes de longitud
    # LOOKBACK_STEPS sobre 'features', con target = glucosa HORIZON_STEPS
    # pasos por delante del final de la ventana.

    # X resultante: (n_muestras, LOOKBACK_STEPS, n_features)
    # y resultante: (n_muestras,)
    
    X_list, y_list = [], []
    min_len = LOOKBACK_STEPS + HORIZON_STEPS

    for _, grupo in df.groupby("patient_id", sort=False):
        g    = grupo[GLUCOSE_COL].values
        feat = grupo[features].values
        n    = len(grupo)
        if n <= min_len:
            continue

        # Ventana [i, i+LOOKBACK) predice glucosa en i+LOOKBACK-1+HORIZON_STEPS
        n_muestras = n - LOOKBACK_STEPS - HORIZON_STEPS + 1
        for i in range(n_muestras):
            ventana = feat[i: i + LOOKBACK_STEPS]
            target  = g[i + LOOKBACK_STEPS - 1 + HORIZON_STEPS]
            X_list.append(ventana)
            y_list.append(target)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y

# CONSTRUCCIÓN DEL MODELO

def _build_model(n_steps: int, n_features: int) -> Sequential:

    # Arquitectura LSTM apilada con regularización por Dropout:
    # LSTM(64, return_sequences=True) -> Dropout -> LSTM(32) -> Dropout -> Dense(1)

    # return_sequences=True en la primera capa permite que la segunda LSTM
    # siga viendo la secuencia completa, capturando dependencias a distintas
    # escalas temporales dentro de la ventana de 60 min.

    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=(n_steps, n_features)),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS // 2),
        Dropout(DROPOUT_RATE),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model

# ENTRENAMIENTO

def _train_lstm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> tuple:
    
    # Entrena el LSTM con EarlyStopping sobre la pérdida de validación.

    # X_val/y_val: partición interna del propio TRAIN (últimos pacientes
    # de train, nunca el test final) usada solo para decidir cuándo parar
    # el entrenamiento y evitar sobreajuste.
    
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    n_steps, n_features = X_train.shape[1], X_train.shape[2]
    model = _build_model(n_steps, n_features)

    print(f"\n[LSTM] Arquitectura: LSTM({LSTM_UNITS}) -> Dropout -> "
          f"LSTM({LSTM_UNITS // 2}) -> Dropout -> Dense(16) -> Dense(1)")
    print(f"[LSTM] Entrenando con {len(X_train):,} secuencias "f"(ventana={n_steps} pasos × {n_features} features)...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    n_epochs_reales = len(history.history["loss"])
    print(f"[LSTM] Entrenamiento detenido en epoch {n_epochs_reales}/{MAX_EPOCHS} "f"(EarlyStopping, patience={PATIENCE})")

    return model, history

# EVALUACIÓN

def _evaluar(
    model: Sequential,
    X_train, X_test, y_train, y_test,
    history,
) -> dict:
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_test  = model.predict(X_test,  verbose=0).flatten()

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    mae_test   = mean_absolute_error(y_test, y_pred_test)
    r2_test    = r2_score(y_test, y_pred_test)

    print(f"\n[LSTM] Métricas (horizonte {HORIZON_MIN} min | "f"train={len(TRAIN_FILES)} pac., test={len(TEST_FILES)} pac.):")
    print(f"       RMSE train : {rmse_train:.2f} mg/dL")
    print(f"       RMSE test  : {rmse_test:.2f}  mg/dL")
    print(f"       MAE  test  : {mae_test:.2f}  mg/dL")
    print(f"       R²   test  : {r2_test:.4f}")

    return {
        "rmse_train"     : rmse_train,
        "rmse_test"      : rmse_test,
        "mae_test"       : mae_test,
        "r2_test"        : r2_test,
        "y_test"         : y_test,
        "y_pred_test"    : y_pred_test,
        "loss_history"   : history.history["loss"],
        "val_loss_history": history.history["val_loss"],
        "n_epochs"       : len(history.history["loss"]),
        "lookback_steps" : LOOKBACK_STEPS,
        "n_train_pacientes": len(TRAIN_FILES),
        "n_test_pacientes" : len(TEST_FILES),
    }

# PUNTO DE ENTRADA

def ejecutar_lstm() -> dict:
    df_train_full, features_train = _cargar_grupo(TRAIN_FILES, "TRAIN")
    df_test,       features_test  = _cargar_grupo(TEST_FILES,  "TEST")

    features  = [f for f in features_train if f in df_test.columns]
    faltantes = [f for f in FEATURES if f not in features]
    if faltantes:
        print(f"\n[LSTM] ⚠ Features ausentes en algún grupo, omitidas: {faltantes}")

    # Partición interna de validación: últimos 4 pacientes de TRAIN_FILES
    # (nunca se toca TEST_FILES). Sirve solo para EarlyStopping.
    n_val_pacientes = max(1, len(TRAIN_FILES) // 5)
    pacientes_train_ids = [
        os.path.basename(p).replace("_preprocessing.csv", "") for p in TRAIN_FILES
    ]
    val_ids   = set(pacientes_train_ids[-n_val_pacientes:])
    train_ids = set(pacientes_train_ids[:-n_val_pacientes])

    df_fit = df_train_full[df_train_full["patient_id"].isin(train_ids)]
    df_val = df_train_full[df_train_full["patient_id"].isin(val_ids)]

    print(f"\n[LSTM] Partición interna — fit: {len(train_ids)} pac. | "f"val (EarlyStopping): {len(val_ids)} pac.")

    # Escalado: se ajusta SOLO con datos de fit para evitar fuga de información
    scaler = StandardScaler()
    scaler.fit(df_fit[features].values)

    def _escalar(df_):
        df_copia = df_.copy()
        df_copia[features] = scaler.transform(df_copia[features].values)
        return df_copia

    df_fit_s   = _escalar(df_fit)
    df_val_s   = _escalar(df_val)
    df_test_s  = _escalar(df_test)

    X_fit,   y_fit   = _construir_secuencias(df_fit_s,  features)
    X_val,   y_val   = _construir_secuencias(df_val_s,  features)
    X_test,  y_test  = _construir_secuencias(df_test_s, features)

    # X_train completo (fit + val) para evaluar el modelo final
    X_train_full = np.concatenate([X_fit, X_val], axis=0)
    y_train_full = np.concatenate([y_fit, y_val], axis=0)

    print(f"\n[LSTM] Secuencias — fit: {len(X_fit):,}  |  val: {len(X_val):,}  |  "f"test: {len(X_test):,}  (lookback={LOOKBACK_STEPS} pasos = "
          f"{LOOKBACK_STEPS * 5} min)")

    if len(X_fit) < 50 or len(X_test) < 10:
        print("[LSTM] ⚠ Insuficientes secuencias para entrenar/evaluar.")
        return {}

    model, history = _train_lstm(X_fit, y_fit, X_val, y_val)
    metricas = _evaluar(model, X_train_full, X_test, y_train_full, y_test, history)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from ML.visualizacion import generar_dashboard_lstm, escribir_reporte_lstm
    generar_dashboard_lstm(metricas, df_test)
    escribir_reporte_lstm(metricas)

    return {"lstm_model": model, "features": features, "metricas": metricas}