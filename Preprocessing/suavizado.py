import pandas as pd
from config import GLUCOSE_COL, ROLLING_WINDOW


# SUAVIZAR SEÑAL (evitar dientes de sierra)

def suavizar_senal(df: pd.DataFrame) -> pd.DataFrame:
    
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