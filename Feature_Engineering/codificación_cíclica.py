import numpy as np
import pandas as pd

def encode_time_cyclically(df, time_col='time'):
    """
    Convierte la columna de tiempo en componentes sen/cos para
    que la LSTM entienda la ciclicidad (medianoche ≈ mediodía en distancia).
    """
    # Asegurar que time_col es datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Segundos del día (0 a 86400)
    seconds_in_day = 24 * 60 * 60
    day_seconds = (df[time_col].dt.hour * 3600 + 
                   df[time_col].dt.minute * 60 + 
                   df[time_col].dt.second)
    
    df['time_sin'] = np.sin(2 * np.pi * day_seconds / seconds_in_day)
    df['time_cos'] = np.cos(2 * np.pi * day_seconds / seconds_in_day)
    
    # También útil: día de la semana (captura patrones lunes vs fin de semana)
    df['weekday_sin'] = np.sin(2 * np.pi * df[time_col].dt.dayofweek / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df[time_col].dt.dayofweek / 7)
    
    return df