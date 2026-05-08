import numpy as np
import pandas as pd


# TIEMPO CÍCLICO: Expresar el tiempo como un reloj con un único valor para cada momento del día

def codificar_tiempo_ciclico(df: pd.DataFrame) -> pd.DataFrame:
    
    print("\n[3/3] Codificación cíclica del tiempo...")

    t = df.index


    # Hora del día con minutos

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