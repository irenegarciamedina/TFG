import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc)

from ML.config import (
    INPUT_FILES, OUTPUT_DIR, REPORT_FILE,
    GLUCOSE_COL, FEATURES, FEATURES_OPCIONALES,
    HYPO_THRESHOLD, DROP_STEPS, DROP_THRESHOLD,
)

from ML.SVM import etiquetar_todos_pacientes


# VALIDACIÓN LOPO

def _lopo_cv(X: np.ndarray, y: np.ndarray, pac_ids: np.ndarray) -> dict:
    
    pacientes = np.unique(pac_ids)
    n_folds   = len(pacientes)
    print(f"\n[LR] Leave-One-Patient-Out CV  ({n_folds} folds)...")

    y_test_all, y_pred_all, y_prob_all = [], [], []

    for fold_i, pac_test in enumerate(pacientes, 1):
        mask_test  = pac_ids == pac_test
        mask_train = ~mask_test

        X_tr, y_tr = X[mask_train], y[mask_train]
        X_te, y_te = X[mask_test],  y[mask_test]

        if len(np.unique(y_tr)) < 2:
            print(f"  [fold {fold_i:>2}/{n_folds}] {pac_test}: SKIP — train sin ambas clases")
            continue
        if len(np.unique(y_te)) < 2:
            print(f"  [fold {fold_i:>2}/{n_folds}] {pac_test}: SKIP — test sin ambas clases")
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C            = 1.0,
                class_weight = "balanced",   # compensa desbalance ~16:1
                max_iter     = 1000,
                solver       = "lbfgs",
                random_state = 42,
            )),
        ])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]

        acc_fold = (y_pred == y_te).mean()
        print(f"  [fold {fold_i:>2}/{n_folds}] test={pac_test}  "f"n_test={len(y_te):>4}  acc={acc_fold:.3f}  "f"caídas={y_te.sum()}")

        y_test_all.append(y_te)
        y_pred_all.append(y_pred)
        y_prob_all.append(y_prob)

    if not y_test_all:
        return {}

    return {
        "y_test"         : np.concatenate(y_test_all),
        "y_pred"         : np.concatenate(y_pred_all),
        "y_prob"         : np.concatenate(y_prob_all),
        "n_folds_usados" : len(y_test_all),
        "n_folds_total"  : n_folds,
    }

# EVALUACIÓN

def _evaluar_lr(y_test, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    especif   = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

    dict_report = classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["Ruido", "Caída real"],
        output_dict=True,
        zero_division=0,
    )

    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = np.array([0]), np.array([0]), 0.0

    print(f"\n[LR] Métricas agregadas LOPO:")
    print(f"      Sensibilidad  : {recall:.4f}")
    print(f"      Especificidad : {especif:.4f}")
    print(f"      Precisión     : {precision:.4f}")
    print(f"      F1-score      : {f1:.4f}")
    print(f"      AUC-ROC       : {roc_auc:.4f}")

    return {
        "cm"            : cm,
        "report"        : dict_report,
        "fpr"           : fpr,
        "tpr"           : tpr,
        "roc_auc"       : roc_auc,
        "sensibilidad"  : recall,
        "especificidad" : especif,
        "precision"     : precision,
        "f1"            : f1,
    }


# PUNTO DE ENTRADA

def ejecutar_logistic_regression() -> dict:
    if not INPUT_FILES:
        print("[LR] ⚠ No hay ficheros preprocesados disponibles.")
        return {}

    print(f"\n[LR] Cargando {len(INPUT_FILES)} fichero(s)...")
    frames = []
    for path in INPUT_FILES:
        df_p = pd.read_csv(path, index_col=0, parse_dates=True)
        df_p["patient_id"] = os.path.basename(path).replace("_preprocessing.csv", "")
        frames.append(df_p)
    df = pd.concat(frames, ignore_index=False).sort_index()

    features = list(FEATURES) + [f for f in FEATURES_OPCIONALES if f in df.columns]

    X, y, indices, pac_ids = etiquetar_todos_pacientes(df, features)

    if len(X) < 10:
        print("[LR] ⚠ Insuficientes eventos detectados para entrenar.")
        return {}

    print(f"[LR] Eventos detectados: {len(X)}  "f"(caídas reales: {y.sum()}, ruido: {(y==0).sum()})  "f"pacientes: {len(np.unique(pac_ids))}")

    lopo = _lopo_cv(X, y, pac_ids)
    if not lopo:
        print("[LR] ⚠ LOPO CV no produjo resultados válidos.")
        return {}

    metricas = _evaluar_lr(lopo["y_test"], lopo["y_pred"], lopo["y_prob"])
    metricas.update({
        "y_test"         : lopo["y_test"],
        "y_pred"         : lopo["y_pred"],
        "y_prob"         : lopo["y_prob"],
        "indices_test"   : indices,
        "n_pacientes"    : len(INPUT_FILES),
        "n_folds_usados" : lopo["n_folds_usados"],
        "n_folds_total"  : lopo["n_folds_total"],
        "validacion"     : "LOPO",
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from ML.visualizacion import generar_dashboard_lr, escribir_reporte_lr
    print("[LR] Generando dashboard y reporte...")
    generar_dashboard_lr(metricas, df)
    escribir_reporte_lr(metricas)

    return {"metricas": metricas}