import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ML.config import OUTPUT_DIR, HORIZON_MIN

C1 = "#2980B9"
C2 = "#E74C3C"
C3 = "#27AE60"
C4 = "#9B59B6"
PALETA = [C1, C2, C3, C4]

# TABLA COMPARATIVA DE REGRESORES: RF, XGB, RIDGE, LSTM

def comparar_regresores(resultados: dict) -> pd.DataFrame:
    """
    resultados = {
        "Random Forest" : metricas_rf,
        "XGBoost"       : metricas_xgb,
        "Ridge"         : metricas_ridge,
        "LSTM"          : metricas_lstm,
    }
    """
    filas = []
    for nombre, m in resultados.items():
        if m:
            filas.append({
                "Modelo"      : nombre,
                "RMSE (mg/dL)": round(m.get("rmse_test", float("nan")), 2),
                "MAE (mg/dL)" : round(m.get("mae_test",  float("nan")), 2),
                "R²"          : round(m.get("r2_test",   float("nan")), 4),
            })

    df = pd.DataFrame(filas).set_index("Modelo")
    print("\n[COMP] Comparativa Regresores:")
    print(df.to_string())
    return df


# TABLA COMPARATIVA DE CLASIFICADORES: SVM, GBT, LR

def comparar_clasificadores(resultados: dict) -> pd.DataFrame:
    """
    resultados = {
        "SVM"                 : metricas_svm,
        "Gradient Boosting"   : metricas_gbt,
        "Logistic Regression" : metricas_lr,
    }
    """
    filas = []
    for nombre, m in resultados.items():
        if m:
            filas.append({
                "Modelo"          : nombre,
                "Sensibilidad"    : round(m.get("sensibilidad",  float("nan")), 4),
                "Especificidad"   : round(m.get("especificidad", float("nan")), 4),
                "Precisión"       : round(m.get("precision",     float("nan")), 4),
                "F1-score"        : round(m.get("f1",            float("nan")), 4),
                "AUC-ROC"         : round(m.get("roc_auc",       float("nan")), 4),
            })

    df = pd.DataFrame(filas).set_index("Modelo")
    print("\n[COMP] Comparativa Clasificadores:")
    print(df.to_string())
    return df


# GRÁFICO COMPARATIVO

def generar_figura_comparativa(
    res_regresores: dict,
    res_clasificadores: dict,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    C1 = "#2980B9"
    C2 = "#E74C3C"
    C3 = "#27AE60"
    C4 = "#9B59B6"
    PALETA = [C1, C2, C3, C4]

    plt.rcParams.update({
        "font.size"       : 13,
        "axes.titlesize"  : 14,
        "axes.labelsize"  : 13,
        "xtick.labelsize" : 11,
        "ytick.labelsize" : 11,
        "legend.fontsize" : 11,
        "figure.titlesize": 16,
    })

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # RMSE TEST

    ax0 = fig.add_subplot(gs[0, 0])
    reg_nombres = list(res_regresores.keys())
    rmse_vals   = [res_regresores[n].get("rmse_test", 0) for n in reg_nombres]
    bars = ax0.bar(reg_nombres, rmse_vals, color=PALETA[:len(reg_nombres)], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, rmse_vals):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax0.set_ylabel("RMSE (mg/dL)")
    ax0.set_title(f"RMSE en test — horizonte {HORIZON_MIN} min")
    ax0.set_ylim(0, max(rmse_vals) * 1.25 if rmse_vals else 1)
    ax0.axhline(y=min(rmse_vals) if rmse_vals else 0,
                color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    
    
    # R² TEST

    ax1 = fig.add_subplot(gs[0, 1])
    r2_vals = [res_regresores[n].get("r2_test", 0) for n in reg_nombres]
    bars = ax1.bar(reg_nombres, r2_vals, color=PALETA[:len(reg_nombres)], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, r2_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("R²")
    ax1.set_title(f"R² en test — horizonte {HORIZON_MIN} min")
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)


    # F1 AUC-ROC

    ax2 = fig.add_subplot(gs[1, 0])
    clas_nombres = list(res_clasificadores.keys())
    f1_vals   = [res_clasificadores[n].get("f1",       0) for n in clas_nombres]
    auc_vals  = [res_clasificadores[n].get("roc_auc",  0) for n in clas_nombres]

    x      = np.arange(len(clas_nombres))
    ancho  = 0.35
    b1 = ax2.bar(x - ancho / 2, f1_vals,  ancho, label="F1-score", color=C1, alpha=0.85, edgecolor="white")
    b2 = ax2.bar(x + ancho / 2, auc_vals, ancho, label="AUC-ROC",  color=C2, alpha=0.85, edgecolor="white")
    for bar, val in zip(list(b1) + list(b2), f1_vals + auc_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(clas_nombres, rotation=12)
    ax2.set_ylabel("Puntuación")
    ax2.set_title("F1-score y AUC-ROC — Clasificadores (LOPO)")
    ax2.set_ylim(0, 1.15)
    ax2.legend()


    # CURVAS ROC

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Azar (AUC = 0.50)")
    for i, (nombre, m) in enumerate(res_clasificadores.items()):
        if m and "fpr" in m and "tpr" in m:
            ax3.plot(m["fpr"], m["tpr"], linewidth=2, color=PALETA[i], label=f"{nombre} (AUC={m.get('roc_auc', 0):.3f})")
    ax3.set_xlabel("Tasa de Falsos Positivos (1 − Especificidad)")
    ax3.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    ax3.set_title("Curvas ROC — Clasificadores (LOPO)")
    ax3.legend(loc="lower right")
    ax3.set_xlim([-0.02, 1.02])
    ax3.set_ylim([-0.02, 1.05])

    fig.suptitle("Comparativa de Modelos — Predicción de Glucosa (CGM - HUPA-UCM)", fontweight="bold")

    ruta = os.path.join(OUTPUT_DIR, "Comparativa_modelos.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[COMP] Figura comparativa guardada en: {ruta}")


# COMPARATIVA REAL VS PREDICHO SERIE TEMPORAL TODOS LOS REGRESORES

def generar_figura_real_vs_predicho(
    res_regresores: dict,
    n_muestras: int = 300,
) -> None:
    
    # Genera una figura comparativa de la serie temporal real vs. predicha para todos los regresores
    # superppuestos en la misma gráfica

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reg_nombres = [n for n in res_regresores if res_regresores[n].get("y_test") is not None]

    plt.rcParams.update({
        "font.size"       : 12,
        "axes.titlesize"  : 13,
        "axes.labelsize"  : 12,
        "xtick.labelsize" : 10,
        "ytick.labelsize" : 10,
        "legend.fontsize" : 10,
        "figure.titlesize": 16,
    })

    fig, ax = plt.subplots(figsize=(16, 7))

    # "Real" se toma de un único modelo de referencia: al provenir todos del
    # mismo conjunto de test, la curva real es (prácticamente) la misma para
    # todos; pintarla una sola vez evita saturar la gráfica con líneas
    # oscuras repetidas.
    nombre_ref = reg_nombres[0]
    y_test_ref = np.asarray(res_regresores[nombre_ref]["y_test"])
    n = min(n_muestras, len(y_test_ref))
    idx = np.arange(n)
    ax.plot(idx, y_test_ref[:n], color="#2C3E50", linewidth=2.4, label="Real", zorder=10)

    estilos = ["--", "-.", ":", "--"]
    for i, nombre in enumerate(reg_nombres):
        m = res_regresores[nombre]
        y_pred = np.asarray(m["y_pred_test"])

        # los índices reales están avanzados en LOOKBACK_STEPS - 1
        if "LSTM" in nombre.upper():
            lookback_offset = 11 # LOOKBACK_STEPS - 1 (12 - 1)

            # se construye el gráfico moviendo su eje X hacia la derecha para alinearla con el "real" de referencia
            eje_x = np.arange(lookback_offset, len(y_pred) + lookback_offset)
        else:
            eje_x = np.arange(len(y_pred))

        n_i = min(n_muestras, len(y_pred))
        ax.plot(
            np.arange(n_i), y_pred[:n_i],
            color=PALETA[i % len(PALETA)],
            linewidth=1.6,
            linestyle=estilos[i % len(estilos)],
            alpha=0.9,
            label=f"{nombre} (RMSE={m.get('rmse_test', 0):.2f}, R²={m.get('r2_test', 0):.3f})",
        )

    ax.set_xlabel("Muestra (test)")
    ax.set_ylabel("Glucosa (mg/dL)")
    ax.set_title(f"Comparativa Real vs. Predicho — Todos los Regresores (horizonte {HORIZON_MIN} min)", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.25)

    fig.tight_layout()

    ruta = os.path.join(OUTPUT_DIR, "Comparativa_real_vs_predicho.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[COMP] Figura real vs. predicho (regresores) guardada en: {ruta}") 


    # REPORTE TEXTO

def escribir_reporte_comparativa(
            
    df_reg: pd.DataFrame,
    df_clas: pd.DataFrame,
    report_file: str,
) -> None:
    sep  = "=" * 68
    sep2 = "-" * 68

    with open(report_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n{sep}\n")
        f.write("  COMPARATIVA DE MODELOS\n")
        f.write(f"{sep}\n\n")

        f.write("REGRESORES — Predicción de glucosa a 40 min\n")
        f.write(f"{sep2}\n")
        f.write(df_reg.to_string())
        f.write(f"\n  → Mejor RMSE : {df_reg['RMSE (mg/dL)'].idxmin()}\n")
        f.write(f"  → Mejor R²   : {df_reg['R²'].idxmax()}\n")

        f.write(f"\n\nCLASIFICADORES — Detección de caídas bruscas (LOPO)\n")
        f.write(f"{sep2}\n")
        f.write(df_clas.to_string())
        f.write(f"\n  → Mejor F1      : {df_clas['F1-score'].idxmax()}\n")
        f.write(f"  → Mejor AUC-ROC : {df_clas['AUC-ROC'].idxmax()}\n")
        f.write(f"\n{sep}\n")

    print(f"[COMP] Reporte comparativo añadido a: {report_file}")

