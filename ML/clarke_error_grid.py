# Clarke Error Grid CEG: Utilizado en evaluaciones clínicas de predicciones de glucosa

# ZONAS:
# A - Diferencia clínica aceptable: <20% en rango de hipoglucemia
# B - Error >20% sin consecuencias clínicas muy relevantes
# C - Tratamiento corrector innecesario
# D - Fallo en la detección de hipoglucemia o hiperglucemia peligrosa
# E - Tratamiento opuesto al necesario

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# CLASIFICACIÓN DE PUNTOS EN ZONAS

def clasificar_zonas(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # Asigna cada par (real, predicho) a una zona CEG (A–E).

    #  y_real : array de glucosa real en mg/dL
    # y_pred : array de glucosa predicha en mg/dL

    # Devuelve
    # zonas : array de caracteres 'A'–'E', misma longitud que las entradas

    n     = len(y_real)
    zonas = np.empty(n, dtype="U1")

    for i in range(n):
        r = float(y_real[i])
        p = float(y_pred[i])

        # Zona E - tratamiento contraindicado
        # Hipoglucemia real + predicción de hiperglucemia, o viceversa

        if (r <= 70 and p >= 180) or (r >= 180 and p <= 70):
            zonas[i] = "E"

        # Zona D - fallo de detección
        # Debe evaluarse ANTES del criterio ±20 % de la zona A, porque un punto
        # con r=65 y p=78 cumple el ±20 % pero representa un fallo de detección
        # de hipoglucemia y debe clasificarse como D (Clarke 1987).
        # Real hipo + predicción en rango normal/alto

        elif r <= 70 and p > 70 and p <= 180:
            zonas[i] = "D"
        # Real hiperglucemia grave + predicción en rango aceptable

        elif r >= 240 and p >= 70 and p <= 180:
            zonas[i] = "D"

        # Zona C - tratamiento innecesario
        # También debe evaluarse antes del ±20 % de zona A.
        # Glucosa real en rango pero predicción dispara corrección

        elif 70 <= r <= 180 and p > r * 1.20 and p >= 180:
            zonas[i] = "C"
        elif 70 <= r <= 180 and p < r * 0.80 and p <= 70:
            zonas[i] = "C"

        # Zona A - clínicamente aceptable
        # Ambos en rango hipo, o diferencia relativa < 20 %

        elif r < 70 and p < 70:
            zonas[i] = "A"
        elif abs(p - r) / max(r, 1e-6) <= 0.20:
            zonas[i] = "A"

        # Zona B - error sin consecuencia clínica
        # Todo lo que está fuera de A pero no llega a C, D, E

        else:
            zonas[i] = "B"

    return zonas


def calcular_porcentajes(zonas: np.ndarray) -> dict:
    """Devuelve el porcentaje de puntos en cada zona."""
    n = len(zonas)
    return {z: float((zonas == z).sum()) / n * 100 for z in "ABCDE"}


# ---------------------------------------------------------------------------
# GENERAR DASHBOARD
# ---------------------------------------------------------------------------

def _dibujar_ceg(ax, y_real: np.ndarray, y_pred: np.ndarray, zonas: np.ndarray, pct: dict, titulo: str, n_pacientes_test: int) -> None:

    COL = {
        "A": "#27AE60",   # verde
        "B": "#2980B9",   # azul
        "C": "#E67E22",   # naranja
        "D": "#E74C3C",   # rojo
        "E": "#8E44AD",   # morado
    }
    BG_COL = {
        "A": "#D5E8D4",
        "B": "#DAE8FC",
        "C": "#FFE6CC",
        "D": "#F8CECC",
        "E": "#E1D5E7",
    }

    ax.set_facecolor("#F8F9FA")

    # Banda de zona A (± 20 %)

    x_ref = np.linspace(0, 400, 400)
    ax.fill_between(x_ref,
                    np.clip(x_ref * 0.80, 0, 400),
                    np.clip(x_ref * 1.20, 0, 400),
                    alpha=0.12, color=BG_COL["A"], zorder=0, label="_nolegend_")

    # Zona hipo-hipo (ambos < 70)

    ax.fill_between([0, 70], [0, 0], [70, 70],
                    alpha=0.12, color=BG_COL["A"], zorder=0)

    # Líneas de referencia
    ax.plot(x_ref, x_ref,        color="black",   lw=1.4, ls="-",  zorder=5)
    ax.plot(x_ref, x_ref * 1.20, color="#555555", lw=0.9, ls="--", zorder=5, label="_nolegend_")
    ax.plot(x_ref, x_ref * 0.80, color="#555555", lw=0.9, ls="--", zorder=5, label="_nolegend_")

    # Líneas clínicas (70 y 180 mg/dL)
    for val in [70, 180]:
        ax.axvline(val, color="#E67E22", lw=0.7, ls=":",  alpha=0.6, zorder=4)
        ax.axhline(val, color="#E67E22", lw=0.7, ls=":",  alpha=0.6, zorder=4)

    # Puntos coloreados por zona
    for zona in "ABCDE":
        mask = zonas == zona
        if mask.any():
            ax.scatter(
                y_real[mask], y_pred[mask],
                c=COL[zona], s=5, alpha=0.35,
                label=f"Zona {zona}  {pct[zona]:.1f}%  (n={mask.sum():,})",
                zorder=3,
            )

    # Etiquetas de zona en el gráfico
    etiq_pos = {
        "A": (30,  350), "B": (340, 30),
        "C": (160, 380), "D": (380, 130),
        "E": (380, 50),
    }
    for zona, (xp, yp) in etiq_pos.items():
        if (zonas == zona).any():
            ax.text(xp, yp, zona, fontsize=14, fontweight="bold",
                    color=COL[zona], alpha=0.6, zorder=6)

    # Decoración
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Glucosa real (mg/dL)",     fontsize=11)
    ax.set_ylabel("Glucosa predicha (mg/dL)", fontsize=11)
    ax.set_title(
        f"Clarke Error Grid — Random Forest  (horizonte 40 min)\n"
        f"{titulo}  |  n = {len(y_real):,} puntos  |  {n_pacientes_test} pacientes test",
        fontweight="bold", fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92, title="Zonas CEG")
    ax.grid(alpha=0.15)
    ax.set_aspect("equal")


def _dibujar_barras(ax, pct: dict) -> None:
    """Dibuja un gráfico de barras con el porcentaje por zona."""
    COL = {
        "A": "#27AE60", "B": "#2980B9", "C": "#E67E22",
        "D": "#E74C3C", "E": "#8E44AD",
    }
    zonas  = list("ABCDE")
    vals   = [pct[z] for z in zonas]
    colores = [COL[z] for z in zonas]

    bars = ax.bar(zonas, vals, color=colores, alpha=0.82, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Línea de referencia en 95 % para zona A
    ax.axhline(95, color="#27AE60", ls="--", lw=1.0, alpha=0.7, label="Objetivo zona A ≥ 95%")
    ax.set_ylim(0, 110)
    ax.set_xlabel("Zona CEG", fontsize=10)
    ax.set_ylabel("Porcentaje de predicciones (%)", fontsize=10)
    ax.set_title("Distribución por zonas", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_facecolor("#F8F9FA")


# FUNCIÓN PRINCIPAL EXPORTADA

def generar_clarke_error_grid(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    test_files: list = None,
) -> dict:
    # Genera la Clarke Error Grid completa y la guarda en PLOT_CEG.

    # y_real      : array de glucosa real (mg/dL), set de test
    # y_pred      : array de glucosa predicha (mg/dL), set de test
    # test_files  : lista de rutas de los ficheros del grupo test (para el título)

    # Devuelve
    # pct : dict con porcentaje en cada zona {'A': x, 'B': x, ...}

    from ML.config import PLOT_CEG, OUTPUT_DIR, HORIZON_MIN

    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Eliminar NaNs
    mask   = np.isfinite(y_real) & np.isfinite(y_pred)
    y_real = y_real[mask]
    y_pred = y_pred[mask]

    zonas = clasificar_zonas(y_real, y_pred)
    pct   = calcular_porcentajes(zonas)

    # Título con IDs de pacientes test
    if test_files:
        ids = [os.path.basename(p).replace("_preprocessing.csv", "") for p in test_files]
        titulo = "Test: " + ", ".join(ids)
        n_test = len(test_files)
    else:
        titulo = "Set de test"
        n_test = 0

    # Figura: CEG + barras
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("white")

    _dibujar_ceg(axes[0], y_real, y_pred, zonas, pct, titulo, n_test)
    _dibujar_barras(axes[1], pct)

    ab_pct = pct["A"] + pct["B"]
    fig.suptitle(
        f"Evaluación Clínica — Clarke Error Grid\n"
        f"Zonas seguras A+B: {ab_pct:.1f}%  |  "
        f"Zona A (ideal ≥ 95%): {pct['A']:.1f}%",
        fontsize=13, fontweight="bold", y=1.01,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_CEG, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[CEG] Clarke Error Grid guardada: {PLOT_CEG}")
    print(f"[CEG] Distribución de zonas (n={len(y_real):,} predicciones):")
    for z in "ABCDE":
        n_z   = int((zonas == z).sum())
        barra = "█" * int(pct[z] / 2)
        print(f"      Zona {z}: {pct[z]:5.1f}%  ({n_z:,})  {barra}")
    print(f"[CEG] Zonas A+B (clínicamente seguras): {ab_pct:.1f}%")

    return pct