"""
paso2_suavizado.py — Paso 2: Suavizado de la señal glucémica (denoising).

Responsabilidad única:
  Aplicar una media móvil centrada sobre la columna de glucosa para eliminar
  el ruido de alta frecuencia del sensor CGM, preservando la señal original
  en una columna auxiliar para comparación posterior.
"""

import pandas as pd
from config import GLUCOSE_COL, ROLLING_WINDOW


def suavizar_senal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica una media móvil centrada de 15 minutos sobre la señal de glucosa.

    El análisis de HUPA0001P muestra un delta medio entre lecturas
    consecutivas de 16.5 mg/dL, con 305 saltos superiores a 50 mg/dL.
    Parte de esta variabilidad es ruido del sensor (inflamación inicial
    alrededor del filamento, compresión al dormir, variabilidad enzimática).

    Por qué media móvil CENTRADA y no simple:
      El CGM introduce un retraso biológico de 5–12 min por medir glucosa
      intersticial en lugar de sangre capilar. Una media simple (mirando
      solo hacia atrás) agravaría ese retraso. La versión centrada es
      simétrica y no desplaza la curva en el tiempo.

    Ventana de 3 intervalos (= 15 minutos):
      Atenúa el ruido sin borrar la morfología de los picos postprandiales,
      cuya duración típica es de 1–2 horas.

    La señal original se preserva en 'glucose_raw' para poder incluirla
    en las gráficas comparativas de la memoria del TFG.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con GLUCOSE_COL corregida por paso1_rango_sensor.py.

    Retorna
    -------
    pd.DataFrame
        DataFrame con GLUCOSE_COL suavizada y nueva columna 'glucose_raw'.
    """
    print(f"\n[2/3] Suavizado — Media Móvil Centrada "
          f"(ventana = {ROLLING_WINDOW} × 5 min = {ROLLING_WINDOW * 5} min)...")

    # Guardar señal original antes de transformar
    df["glucose_raw"] = df[GLUCOSE_COL].copy()

    df[GLUCOSE_COL] = (
        df[GLUCOSE_COL]
        .rolling(window=ROLLING_WINDOW, center=True, min_periods=1)
        .mean()
    )

    # Métricas de efectividad del suavizado
    var_antes   = df["glucose_raw"].var()
    var_despues = df[GLUCOSE_COL].var()
    reduccion   = (var_antes - var_despues) / var_antes * 100

    print(f"      -> Varianza antes:    {var_antes:.2f} (mg/dL)²")
    print(f"      -> Varianza después:  {var_despues:.2f} (mg/dL)²")
    print(f"      -> Reducción de ruido: {reduccion:.1f}%")

    return df