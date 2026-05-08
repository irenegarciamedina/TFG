"""
paso1_rango_sensor.py — Paso 1: Corrección del rango operativo del sensor CGM.

Responsabilidad única:
  Detectar y corregir valores de glucosa que están fuera del rango físico
  del sensor FreeStyle Libre 2, diferenciando entre saturación real del
  sensor (capear) y errores técnicos bajo el mínimo (interpolar).
"""

import numpy as np
import pandas as pd
from config import GLUCOSE_COL, SENSOR_MAX, SENSOR_MIN


def corregir_rango_sensor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige los valores de glucosa fuera del rango operativo del sensor.

    El análisis de HUPA0001P revela 8 valores superiores a 400 mg/dL
    (máximo hasta 444 mg/dL), todos en el episodio nocturno del
    23 de junio a las 3:00–5:00h.

    Criterio para valores > SENSOR_MAX (400 mg/dL):
      Se capean en 400 mg/dL. NO se eliminan porque son lecturas de
      saturación del sensor durante una hiperglucemia severa real.
      El FreeStyle Libre 2 no puede reportar valores superiores a 400;
      cuando la glucosa real supera ese umbral, el sensor devuelve su
      máximo. Eliminarlos ocultaría un evento crítico que el modelo
      LSTM debe aprender a anticipar (Fenómeno del Amanecer agravado).

    Criterio para valores < SENSOR_MIN (40 mg/dL):
      Se convierten a NaN y se interpolan linealmente. Por debajo de
      40 mg/dL la señal eléctrica del sensor es infiable — a diferencia
      del caso superior, aquí no hay certeza de que el valor sea real.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la columna GLUCOSE_COL cargada desde carga.py.

    Retorna
    -------
    pd.DataFrame
        DataFrame con la columna GLUCOSE_COL corregida.
    """
    print("\n[1/3] Corrección del rango del sensor CGM...")

    mask_sobre = df[GLUCOSE_COL] > SENSOR_MAX
    mask_bajo  = df[GLUCOSE_COL] < SENSOR_MIN

    n_sobre = int(mask_sobre.sum())
    n_bajo  = int(mask_bajo.sum())

    # --- Saturación por encima del rango (capear, no eliminar) ---
    if n_sobre > 0:
        print(f"      -> {n_sobre} valor(es) > {SENSOR_MAX} mg/dL (saturación del sensor):")
        for ts, val in df.loc[mask_sobre, GLUCOSE_COL].items():
            print(f"             {ts}  {val:.1f} mg/dL  ->  capeado a {SENSOR_MAX}")
        df.loc[mask_sobre, GLUCOSE_COL] = SENSOR_MAX
    else:
        print(f"      -> Sin valores > {SENSOR_MAX} mg/dL")

    # --- Fuera de rango inferior (interpolar) ---
    if n_bajo > 0:
        print(f"      -> {n_bajo} valor(es) < {SENSOR_MIN} mg/dL — interpolados linealmente")
        df.loc[mask_bajo, GLUCOSE_COL] = np.nan
        df[GLUCOSE_COL] = df[GLUCOSE_COL].interpolate(method="linear")
    else:
        print(f"      -> Sin valores < {SENSOR_MIN} mg/dL")

    return df