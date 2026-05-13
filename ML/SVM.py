# SVM para clasificación de caídas bruscas de glucosa.

import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc)
from sklearn.pipeline import Pipeline

from ML.config import (
    INPUT_FILES, PLOT_SVM, REPORT_FILE, OUTPUT_DIR,
    GLUCOSE_COL, FEATURES, FEATURES_OPCIONALES,
    HYPO_THRESHOLD, DROP_STEPS, DROP_THRESHOLD,
    SVM_C, SVM_KERNEL, SVM_GAMMA,
    TRAIN_RATIO,
)


# ETIQUETADO DE CAÍDAS BRUSCAS

def etiquetar_caidas(df: pd.DataFrame, features: list) -> tuple:

    # Detecta caídas bruscas dentro de un DataFrame de un único paciente.
    # No se mezclan ventanas entre pacientes.

    g = df[GLUCOSE_COL].values
    delta_ventana = np.array([
        g[i] - g[max(0, i - DROP_STEPS)] for i in range(len(g))
    ])

    mask_caida = delta_ventana < DROP_THRESHOLD
    etiquetas, indices_evento, X_eventos = [], [], []

    for i in np.where(mask_caida)[0]:
        if i + DROP_STEPS >= len(g):
            continue
        es_real = int(g[i + 1] < HYPO_THRESHOLD and g[i + 2] < HYPO_THRESHOLD)
        etiquetas.append(es_real)
        indices_evento.append(df.index[i])
        X_eventos.append(df[features].iloc[i].values)

    return np.array(X_eventos), np.array(etiquetas), indices_evento


def etiquetar_todos_pacientes(df_global: pd.DataFrame, features: list) -> tuple:

    # Llama a etiquetar_caidas para cada paciente por separado y combina.
    # Esto evita que la ventana deslizante cruce el límite entre pacientes.
    
    X_total, y_total, idx_total = [], [], []
    for pid, grupo in df_global.groupby("patient_id", sort=False):
        X_p, y_p, idx_p = etiquetar_caidas(grupo, features)
        if len(X_p) > 0:
            X_total.append(X_p)
            y_total.append(y_p)
            idx_total.extend(idx_p)
    if not X_total:
        return np.array([]), np.array([]), []
    return np.vstack(X_total), np.concatenate(y_total), idx_total


# ENTRENAMIENTO

def entrenar_svm(X_train, y_train):
    print(f"\n[SVM] Entrenando SVM (kernel={SVM_KERNEL}, C={SVM_C})...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA, probability=True, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# EVALUACIÓN

def evaluar_svm(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    v_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    v_recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    v_especif   = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    v_f1        = (2 * v_precision * v_recall / (v_precision + v_recall) if (v_precision + v_recall) > 0 else 0.0)

    dict_report = classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["Ruido", "Caída real"],
        output_dict=True,
        zero_division=0,
    )

    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = np.array([0]), np.array([0]), 0.0

    return {
        "y_pred"        : y_pred,
        "y_prob"        : y_prob,
        "cm"            : cm,
        "report"        : dict_report,
        "fpr"           : fpr,
        "tpr"           : tpr,
        "roc_auc"       : roc_auc,
        "sensibilidad"  : v_recall,
        "especificidad" : v_especif,
        "precision"     : v_precision,
        "f1"            : v_f1,
    }


# PUNTO DE ENTRADA

def ejecutar_svm() -> dict:
    if not INPUT_FILES:
        print("[SVM] ⚠ No hay ficheros preprocesados disponibles.")
        return {}

    print(f"\n[SVM] Cargando {len(INPUT_FILES)} fichero(s)...")
    frames = []
    for path in INPUT_FILES:
        df_p = pd.read_csv(path, index_col=0, parse_dates=True)
        df_p["patient_id"] = os.path.basename(path).replace("_preprocessing.csv", "")
        frames.append(df_p)
    df = pd.concat(frames, ignore_index=False).sort_index()

    features = list(FEATURES) + [f for f in FEATURES_OPCIONALES if f in df.columns]

    X, y, indices = etiquetar_todos_pacientes(df, features)

    if len(X) < 10:
        print("[SVM] ⚠ Insuficientes eventos detectados para entrenar.")
        return {}

    print(f"[SVM] Eventos detectados: {len(X)}  (caídas reales: {y.sum()}, ruido: {(y==0).sum()})")

    # División TEMPORAL (no aleatoria) — igual que RF
    n_train = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    idx_test = indices[n_train:]

    # Comprobación mínima de clases
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("[SVM] Una de las particiones no tiene ambas clases. " "Aumentar el número de pacientes o revisar umbrales.")
        return {}

    pipeline = entrenar_svm(X_train, y_train)
    metricas = evaluar_svm(pipeline, X_test, y_test)
    metricas.update({
        "y_test"       : y_test,
        "indices_test" : idx_test,
        "n_pacientes"  : len(INPUT_FILES),
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from ML.visualizacion import generar_dashboard_svm, escribir_reporte_svm
    print("[SVM] Generando dashboard y reporte...")
    generar_dashboard_svm(metricas, df)
    escribir_reporte_svm(metricas)

    return {"svm_pipeline": pipeline, "metricas": metricas}