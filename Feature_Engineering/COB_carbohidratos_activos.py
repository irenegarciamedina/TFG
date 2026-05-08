import numpy as np
import pandas as pd

# modelo: Absorción exponencial con tiempo medio de absorción
# absorption min: tiempo que toma absorber unos 2/3 de los carbohidratos en minutos

def compute_cob(df, carb_col='carb_input', interval_min=5, absorption_min=180):
    
    n_steps = (absorption_min * 3) // interval_min  # ventana = 3x tiempo medio
    t = np.arange(n_steps) * interval_min
    
    # Curva de absorción: diferencial de una exponencial (tipo campana)ç
    
    tau = absorption_min / np.log(2)  # constante de tiempo
    absorption_curve = (t / tau**2) * np.exp(-t / tau)
    absorption_curve = np.maximum(absorption_curve, 0)
    
    if absorption_curve.sum() > 0:
        absorption_curve /= absorption_curve.sum()
    
    carbs = df[carb_col].fillna(0).values
    cob = np.convolve(carbs, absorption_curve, mode='full')[:len(carbs)]
    
    df['cob'] = cob
    return df
