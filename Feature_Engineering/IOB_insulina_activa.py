import numpy as np
import pandas as pd

# peak min: tiempo pico de acción em minutos
# duration min: duracción total de acción en minuto

def compute_iob(df, bolus_col='bolus_volume_delivered', interval_min=5, peak_min=75, duration_min=240):
   
    # Curva de actividad de insulina (triangular normalizada)

    n_steps = duration_min // interval_min
    t = np.arange(n_steps) * interval_min
    
    # Perfil bilineal: sube hasta peak, baja hasta duration

    activity = np.where(
        t <= peak_min,
        t / peak_min,
        (duration_min - t) / (duration_min - peak_min)
    )
    activity = np.maximum(activity, 0)
    activity /= activity.sum()  # normalizar para que la integral = 1
    
    # IOB = convolución del bolus con la curva de actividad
    
    bolus = df[bolus_col].fillna(0).values
    iob = np.convolve(bolus, activity, mode='full')[:len(bolus)]
    
    df['iob'] = iob
    return df

# El modelo de actividad utilizado es el bilinial como un aproximación simplificada a los modelos de Hovorka o Walsh