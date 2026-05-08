import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from config import GLUCOSE_COL, PLOT_FILE, REPORT_FILE, INPUT_FILE, OUTPUT_FILE, ROLLING_WINDOW


# CÁLCULO DE LAS MÉTRICAS

def calcular_metricas_clinicas(df):
    g = df[GLUCOSE_COL]
    return {
        "TiR"   : ((g >= 70) & (g <= 180)).mean() * 100,
        "TBR_1" : ((g >= 54) & (g < 70)).mean()   * 100,
        "TBR_2" : (g < 54).mean()                  * 100,
        "TAR_1" : ((g > 180) & (g <= 250)).mean()  * 100,
        "TAR_2" : (g > 250).mean()                  * 100,
        "media" : g.mean(),
        "sd"    : g.std(),
        "cv"    : (g.std() / g.mean() * 100) if g.mean() != 0 else 0,
        "gmi"   : 3.31 + 0.02392 * g.mean(),
        "dias"  : (df.index.max() - df.index.min()).days,
    }

def generar_diagnostico(df):
    metricas = calcular_metricas_clinicas(df)
    

    # CONFIGURACIÓN DEL ESTILO

    print(f"\n[+] Restaurando dashboard completo en: {PLOT_FILE}")
    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    C_RAW, C_CLEAN, C_VERDE = "#A8C8E8", "#C0392B", "#27AE60"


    # Serie temporal 

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df["glucose_raw"], color=C_RAW, lw=0.7, alpha=0.6, label="Señal cruda")
    ax1.plot(df.index, df[GLUCOSE_COL], color=C_CLEAN, lw=1.3, label="Señal suavizada")
    ax1.axhspan(70, 180, alpha=0.07, color=C_VERDE)
    ax1.axhline(180, color="#E67E22", ls="--", lw=0.9, alpha=0.8, label="Límites TiR")
    ax1.axhline(70,  color="#E67E22", ls="--", lw=0.9, alpha=0.8)
    ax1.set_title(f"Serie glucémica completa ({metricas['dias']} días)", fontweight="bold")
    ax1.set_ylabel("Glucosa (mg/dL)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.2)


    # 2. Suavizado de 48 horas

    ax2 = fig.add_subplot(gs[1, 0])
    z0 = df.index.min()
    z1 = z0 + pd.Timedelta(hours=48)
    dz = df.loc[z0:z1]
    ax2.plot(dz.index, dz["glucose_raw"], color=C_RAW, lw=0.9, alpha=0.6, label="Cruda")
    ax2.plot(dz.index, dz[GLUCOSE_COL], color=C_CLEAN, lw=1.4, label="Suavizada")
    ax2.axhspan(70, 180, alpha=0.07, color=C_VERDE)
    ax2.set_title("Zoom -- Primeras 48h (Efecto del suavizado)", fontweight="bold", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.2)


    # 3. Histograma de densidad

    ax3 = fig.add_subplot(gs[1, 1])
    bins = np.linspace(30, 420, 75)
    ax3.hist(df["glucose_raw"], bins=bins, color=C_RAW, alpha=0.6, density=True, label="Cruda")
    ax3.hist(df[GLUCOSE_COL],  bins=bins, color=C_CLEAN, alpha=0.65, density=True, label="Suavizada")
    ax3.axvline(70, color="#E67E22", ls="--", lw=1.1, label="Límites TiR")
    ax3.axvline(180, color="#E67E22", ls="--", lw=1.1)
    ax3.set_title("Distribución de glucosa: cruda vs. suavizada", fontweight="bold", fontsize=9)
    ax3.set_ylabel("Densidad")
    ax3.legend(fontsize=7)


    # 4. Reloj Cicadiano (Seno/Coseno)

    ax4 = fig.add_subplot(gs[2, 0])
    sc = ax4.scatter(df["time_hour_sin"], df["time_hour_cos"], c=df.index.hour + df.index.minute/60, cmap="twilight", s=4, alpha=0.5)
    plt.colorbar(sc, ax=ax4, label="Hora del día")
    ax4.set_title("Codificación cíclica (Evita discontinuidad 23h->0h)", fontweight="bold", fontsize=9)
    ax4.set_aspect("equal")


    # 5. Métricas AGP con objetivos clínicos

    ax5 = fig.add_subplot(gs[2, 1])
    categorias = ["TBR-2\n(<54)", "TBR-1\n(54-70)", "TiR\n(70-180)", "TAR-1\n(180-250)", "TAR-2\n(>250)"]
    valores = [metricas["TBR_2"], metricas["TBR_1"], metricas["TiR"], metricas["TAR_1"], metricas["TAR_2"]]
    colores = ["#C0392B", "#E67E22", "#27AE60", "#F39C12", "#8E44AD"]
    bars = ax5.bar(categorias, valores, color=colores, alpha=0.8)
    for bar in bars:
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5, f"{bar.get_height():.1f}%", ha='center', fontweight='bold', size=8)
    ax5.set_title(f"Métricas AGP -- GMI: {metricas['gmi']:.2f}% | CV: {metricas['cv']:.1f}%", fontweight="bold", fontsize=9)
    ax5.set_ylim(0, max(valores) + 10)

    plt.suptitle("Fase 1 -- Preprocesamiento de la Señal Glucémica\nTFG: Predicción con LSTM -- Irene García Medina", fontsize=12, fontweight="bold", y=1.02)
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()

    # GENERAR TXT 
    
    estado_dm = "inestable (CV > 36%)" if metricas["cv"] > 36 else "estable (CV <= 36%)"
    reporte = f"""
================================================================================
REPORTE FASE 1 -- PREPROCESAMIENTO DE LA SEÑAL
================================================================================
Paciente : {INPUT_FILE}
Duración : {metricas['dias']} días | {len(df):,} registros

MÉTRICAS CLÍNICAS (AGP):
  TiR (70-180 mg/dL) : {metricas['TiR']:>6.1f}% (Objetivo >= 70%)
  TBR (< 70 mg/dL)   : {metricas['TBR_1']+metricas['TBR_2']:>6.1f}% (Objetivo < 4%)
  Media Glucosa      : {metricas['media']:.1f} mg/dL
  Variabilidad (CV)  : {metricas['cv']:.1f}% -> Diabetes {estado_dm}
  GMI (HbA1c est.)   : {metricas['gmi']:.2f}%

PROCESAMIENTO APLICADO:
  1. Rango corregido (40-400 mg/dL)
  2. Suavizado (Ventana móvil {ROLLING_WINDOW*5} min)
  3. Tiempo cíclico (Sen/Cos 24h y 7 días)
================================================================================
"""
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(reporte)
    print(f"[+] Reporte guardado: {REPORT_FILE}")