import numpy as np
import pandas as pd
from config import GLUCOSE_COL, SENSOR_MAX, SENSOR_MIN


# CORREGIR RANGO (outliners)

def corregir_rango_sensor(df: pd.DataFrame) -> pd.DataFrame:

    print("\n[1/3] Corrección del rango del sensor CGM...")

    mask_sobre = df[GLUCOSE_COL] > SENSOR_MAX
    mask_bajo  = df[GLUCOSE_COL] < SENSOR_MIN

    n_sobre = int(mask_sobre.sum())
    n_bajo  = int(mask_bajo.sum())

    # Corrección de saturación por encima del rango 

    if n_sobre > 0:
        print(f"      -> {n_sobre} valor(es) > {SENSOR_MAX} mg/dL (saturación del sensor):")
        for ts, val in df.loc[mask_sobre, GLUCOSE_COL].items():
            print(f"             {ts}  {val:.1f} mg/dL  ->  capeado a {SENSOR_MAX}")
        df.loc[mask_sobre, GLUCOSE_COL] = SENSOR_MAX
    else:
        print(f"      -> Sin valores > {SENSOR_MAX} mg/dL")

    # Interpolación de fuera de rango inferior
    if n_bajo > 0:
        print(f"      -> {n_bajo} valor(es) < {SENSOR_MIN} mg/dL — interpolados linealmente")
        df.loc[mask_bajo, GLUCOSE_COL] = np.nan
        df[GLUCOSE_COL] = df[GLUCOSE_COL].interpolate(method="linear")
    else:
        print(f"      -> Sin valores < {SENSOR_MIN} mg/dL")

    return df