import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from ML.config import (
    INPUT_FILE, PLOT_SVM, REPORT_FILE, OUTPUT_DIR,
    GLUCOSE_COL, FEATURES, FEATURES_OPCIONALES,
    HYPO_THRESHOLD, DROP_STEPS, DROP_THRESHOLD,
    SVM_C, SVM_KERNEL, SVM_GAMMA,
    TRAIN_RATIO,
)

def etiquetar_caidas(df: pd.DataFrame, features: list) -> tuple:
    """Detecta y etiqueta eventos de caídas bruscas."""
    g = df[GLUCOSE_COL].values
    delta_ventana = np.array([
        g[i] - g[max(0, i - DROP_STEPS)] for i in range(len(g))
    ])

    mask_caida = delta_ventana < DROP_THRESHOLD
    etiquetas, indices_evento, X_eventos = [], [], []

    for i in np.where(mask_caida)[0]:
        if i + DROP_STEPS >= len(g): continue
        # Clase 1: Real (sigue bajo), Clase 0: Ruido (se recupera)
        es_real = int(g[i + 1] < HYPO_THRESHOLD and g[i + 2] < HYPO_THRESHOLD)
        etiquetas.append(es_real)
        indices_evento.append(df.index[i])
        X_eventos.append(df[features].iloc[i].values)

    return np.array(X_eventos), np.array(etiquetas), indices_evento

def entrenar_svm(X_train, y_train):
    """Pipeline con escalado y clasificador."""
    print(f"\n[SVM] Entrenando SVM (kernel={SVM_KERNEL}, C={SVM_C})...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA, probability=True, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# Pipeline StandardScaler. SVC
# el escalado sirve para que las magnitudes grandes no dominen sobre las features pequeñas
# el hiperplano de separación sería incorrecto



def evaluar_svm(pipeline, X_test, y_test):
    """Calcula métricas de clasificación de forma robusta para el reporte."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Aseguramos matriz 2x2 para evitar errores si falta una clase
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 1. Calculamos valores numéricos individuales con nombres claros
    v_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    v_recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    v_especif   = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    v_f1        = float(2 * (v_precision * v_recall) / (v_precision + v_recall)) if (v_precision + v_recall) > 0 else 0.0

    # 2. Generamos el DICCIONARIO del reporte (lo que visualizacion.py necesita)
    # Es crucial que se llame 'report' en el return final
    dict_report = classification_report(
        y_test, 
        y_pred, 
        labels=[0, 1], 
        target_names=["Ruido", "Caída real"], 
        output_dict=True,
        zero_division=0
    )
    
    # 3. Manejo de curva ROC
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = np.array([0]), np.array([0]), 0.0
    
    # Retornamos el diccionario con las claves exactas que pide tu TFG
    return {
        "y_pred": y_pred, 
        "y_prob": y_prob, 
        "cm": cm,
        "report": dict_report,       # <--- Esto arregla el AttributeError
        "fpr": fpr, 
        "tpr": tpr, 
        "roc_auc": roc_auc,
        "sensibilidad": v_recall,
        "especificidad": v_especif,
        "r": dict_report,               # Valor numérico para etiquetas cortas
        "p": v_precision,
        "f1": v_f1
    }

def ejecutar_svm() -> dict:
    """Orquestador del pipeline SVM con división estratificada correcta."""
    print(f"\n[SVM] Cargando datos desde: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    
    features = list(FEATURES) + [f for f in FEATURES_OPCIONALES if f in df.columns]
    X, y, indices = etiquetar_caidas(df, features)
    
    if len(X) < 10:
        print("[SVM] ⚠ Insuficientes eventos detectados para entrenar.")
        return {}

    # --- CORRECCIÓN AQUÍ: Eliminamos la división manual y usamos SOLO train_test_split ---
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        test_size=(1 - TRAIN_RATIO),
        stratify=y,   # Asegura que haya caídas reales en el test
        random_state=42
    )

    # Entrenar y Evaluar
    pipeline = entrenar_svm(X_train, y_train)
    metricas = evaluar_svm(pipeline, X_test, y_test)
    
    # Inyectar datos necesarios para visualización usando los índices correctos
    metricas.update({
        "y_test": y_test, 
        "indices_test": idx_test  # <--- Usamos idx_test de la división estratificada
    })

    # Importación local para evitar ciclos y generar reportes
    from ML.visualizacion import generar_dashboard_svm, escribir_reporte_svm
    
    print("[SVM] Generando dashboard y reporte...")
    generar_dashboard_svm(metricas, df)
    escribir_reporte_svm(metricas)

    return {"svm_pipeline": pipeline, "metricas": metricas}