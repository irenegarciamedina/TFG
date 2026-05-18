# SVM para clasificación de caídas bruscas de glucosa.
# Validación: Leave-One-Patient-Out (LOPO) — el estándar en datos clínicos.

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
)


# ---------------------------------------------------------------------------
# ETIQUETADO DE CAÍDAS BRUSCAS
# ---------------------------------------------------------------------------

def etiquetar_caidas(df: pd.DataFrame, features: list) -> tuple:
    """Detecta caídas bruscas dentro de un DataFrame de un único paciente.
    No se mezclan ventanas entre pacientes."""

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
    """Llama a etiquetar_caidas para cada paciente por separado y combina.
    Devuelve también el mapping paciente -> índices de sus eventos."""
    X_total, y_total, idx_total, pac_ids = [], [], [], []
    for pid, grupo in df_global.groupby("patient_id", sort=False):
        X_p, y_p, idx_p = etiquetar_caidas(grupo, features)
        if len(X_p) > 0:
            X_total.append(X_p)
            y_total.append(y_p)
            idx_total.extend(idx_p)
            pac_ids.extend([pid] * len(X_p))
    if not X_total:
        return np.array([]), np.array([]), [], np.array([])
    return (
        np.vstack(X_total),
        np.concatenate(y_total),
        idx_total,
        np.array(pac_ids),
    )


# ---------------------------------------------------------------------------
# VALIDACIÓN LEAVE-ONE-PATIENT-OUT (LOPO)
# ---------------------------------------------------------------------------

def _lopo_cv(X: np.ndarray, y: np.ndarray, pac_ids: np.ndarray) -> dict:
    # Leave-One-Patient-Out cross-validation.
    
    # En cada fold un paciente actúa como test y el resto como train.
    # Evita la fuga de información entre pacientes y es el estándar para evaluar modelos en datos clínicos con múltiples sujetos.

    pacientes = np.unique(pac_ids)
    n_folds   = len(pacientes)
    print(f"\n[SVM] Leave-One-Patient-Out CV  ({n_folds} folds)...")

    y_test_all,  y_pred_all, y_prob_all = [], [], []
    pac_test_all = []

    for fold_i, pac_test in enumerate(pacientes, 1):
        mask_test  = pac_ids == pac_test
        mask_train = ~mask_test

        X_tr, y_tr = X[mask_train], y[mask_train]
        X_te, y_te = X[mask_test],  y[mask_test]

        # Saltar fold si alguna partición no tiene las dos clases

        if len(np.unique(y_tr)) < 2:
            print(f"  [fold {fold_i:>2}/{n_folds}] {pac_test}: SKIP — train sin ambas clases")
            continue
        if len(np.unique(y_te)) < 2:
            print(f"  [fold {fold_i:>2}/{n_folds}] {pac_test}: SKIP — test sin ambas clases")
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA, probability=True, class_weight="balanced", random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]

        acc_fold = (y_pred == y_te).mean()
        print(f"  [fold {fold_i:>2}/{n_folds}] test={pac_test}  " f"n_test={len(y_te):>4}  acc={acc_fold:.3f}  " f"caídas={y_te.sum()}")

        y_test_all.append(y_te)
        y_pred_all.append(y_pred)
        y_prob_all.append(y_prob)
        pac_test_all.extend([pac_test] * len(y_te))

    if not y_test_all:
        return {}

    y_test = np.concatenate(y_test_all)
    y_pred = np.concatenate(y_pred_all)
    y_prob = np.concatenate(y_prob_all)

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "n_folds_usados": len(y_test_all),
        "n_folds_total":  n_folds,
    }



# EVALUACIÓN AGREGADA

def evaluar_svm(y_test: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
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

    X, y, indices, pac_ids = etiquetar_todos_pacientes(df, features)

    if len(X) < 10:
        print("[SVM] ⚠ Insuficientes eventos detectados para entrenar.")
        return {}

    print(f"[SVM] Eventos detectados: {len(X)}  " f"(caídas reales: {y.sum()}, ruido: {(y==0).sum()})  " f"pacientes: {len(np.unique(pac_ids))}")

    # LOPO cross-validation 
    lopo = _lopo_cv(X, y, pac_ids)
    if not lopo:
        print("[SVM] ⚠ LOPO CV no produjo resultados válidos.")
        return {}

    metricas = evaluar_svm(lopo["y_test"], lopo["y_pred"], lopo["y_prob"])
    metricas.update({
        "y_test"        : lopo["y_test"],
        "y_pred"        : lopo["y_pred"],
        "y_prob"        : lopo["y_prob"],
        "indices_test"  : indices,          # se usa sólo para la serie temporal
        "n_pacientes"   : len(INPUT_FILES),
        "n_folds_usados": lopo["n_folds_usados"],
        "n_folds_total" : lopo["n_folds_total"],
        "validacion"    : "LOPO",
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from ML.visualizacion import generar_dashboard_svm, escribir_reporte_svm
    print("[SVM] Generando dashboard y reporte...")
    generar_dashboard_svm(metricas, df)
    escribir_reporte_svm(metricas)

    return {"metricas": metricas}