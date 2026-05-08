"""
paso3_tiempo_ciclico.py — Paso 3: Codificación cíclica del tiempo.

Responsabilidad única:
  Transformar el índice temporal en variables trigonométricas (seno/coseno)
  que permiten a la red LSTM aprender patrones circadianos y semanales sin
  discontinuidades numéricas en los límites del día o de la semana.
"""

import numpy as np
import pandas as pd


def codificar_tiempo_ciclico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade cuatro columnas de codificación cíclica del tiempo al DataFrame.

    Problema que resuelve:
      Si la hora se codifica como un entero (0–23), la red percibe una
      distancia de 23 entre las 23:55h y las 00:00h, cuando en realidad
      están separadas por 5 minutos. Esto impide aprender el Fenómeno del
      Amanecer (liberación de GH y cortisol entre las 3:00 y 5:00h) y la
      Sensibilidad Nocturna a la insulina, que son exactamente los patrones
      que se analizan en el Estado del Arte del TFG.

    Solución — transformación trigonométrica:
      Representar cada momento como un punto sobre un círculo unitario:

          hora_sin = sin(2π · hora / T)
          hora_cos = cos(2π · hora / T)

      Se necesitan AMBAS coordenadas porque una sola produce ambigüedad:
      las 6h y las 18h tienen el mismo seno, pero distinto coseno.

    Columnas generadas:
      - time_hour_sin / time_hour_cos  →  período circadiano (T = 24h)
      - time_dow_sin  / time_dow_cos   →  período semanal (T = 7 días)
        El día de la semana captura diferencias de actividad y dieta entre
        días laborables y fin de semana, relevantes para la predicción.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con índice DatetimeIndex después del suavizado.

    Retorna
    -------
    pd.DataFrame
        DataFrame con cuatro columnas adicionales de codificación cíclica.
    """
    print("\n[3/3] Codificación cíclica del tiempo (ritmos circadianos)...")

    t = df.index

    # Hora del día con minutos (0.0 – 23.983...)
    horas = t.hour + t.minute / 60.0
    df["time_hour_sin"] = np.sin(2 * np.pi * horas / 24)
    df["time_hour_cos"] = np.cos(2 * np.pi * horas / 24)

    # Día de la semana (0 = lunes, 6 = domingo)
    dow = t.dayofweek.astype(float)
    df["time_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["time_dow_cos"] = np.cos(2 * np.pi * dow / 7)

    print(f"      -> Columnas añadidas: time_hour_sin, time_hour_cos, "
          f"time_dow_sin, time_dow_cos")
    print(f"      -> Rango hour_sin: [{df['time_hour_sin'].min():.3f}, "
          f"{df['time_hour_sin'].max():.3f}]")
    print(f"      -> Rango hour_cos: [{df['time_hour_cos'].min():.3f}, "
          f"{df['time_hour_cos'].max():.3f}]")

    return df