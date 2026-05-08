import numpy as np
import pandas as pd

def compute_cob(df, carb_col='carb_input', interval_min=5, absorption_min=180):
    """
    Calcula los Carbohidratos Activos (COB).
    Modelo: absorción exponencial con tiempo medio de absorción.
    absorption_min: tiempo hasta absorber ~63% de los carbohidratos (minutos)
    """
    n_steps = (absorption_min * 3) // interval_min  # ventana = 3x tiempo medio
    t = np.arange(n_steps) * interval_min
    
    # Curva de absorción: diferencial de una exponencial (tipo campana)
    tau = absorption_min / np.log(2)  # constante de tiempo
    absorption_curve = (t / tau**2) * np.exp(-t / tau)
    absorption_curve = np.maximum(absorption_curve, 0)
    
    if absorption_curve.sum() > 0:
        absorption_curve /= absorption_curve.sum()
    
    carbs = df[carb_col].fillna(0).values
    cob = np.convolve(carbs, absorption_curve, mode='full')[:len(carbs)]
    
    df['cob'] = cob
    return df