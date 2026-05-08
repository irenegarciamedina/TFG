"""
carga.py — Paso 0: Carga y exploración inicial del CSV del dataset HUPA-UCM.

Responsabilidad única:
  Leer el archivo CSV, parsear el índice temporal y mostrar un diagnóstico
  inicial de la serie. No aplica ninguna transformación sobre los datos.
"""

import pandas as pd
from config import INPUT_FILE, TIME_COL


def cargar_datos(filepath: str = INPUT_FILE) -> pd.DataFrame:
    """
    Lee el CSV del dataset HUPA-UCM y devuelve un DataFrame indexado por tiempo.

    El dataset usa punto y coma (;) como separador. La columna 'time' se
    convierte a datetime y se establece como índice para facilitar las
    operaciones temporales en los pasos siguientes.

    Parámetros
    ----------
    filepath : str
        Ruta al archivo CSV del paciente (por defecto la definida en config.py).

    Retorna
    -------
    pd.DataFrame
        DataFrame ordenado cronológicamente con índice DatetimeIndex.
    """
    print("=" * 68)
    print("FASE 1 -- PREPROCESAMIENTO  |  Dataset HUPA-UCM")
    print("=" * 68)
    print(f"\n[0] Cargando archivo: {filepath}")

    df = pd.read_csv(filepath, sep=";")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.set_index(TIME_COL).sort_index()

    # Diagnóstico del archivo cargado
    intervalo_ok = df.index.to_series().diff().dropna().nunique() == 1
    print(f"    Registros  : {len(df):,}")
    print(f"    Columnas   : {list(df.columns)}")
    print(f"    Rango      : {df.index.min().date()}  ->  {df.index.max().date()}")
    print(f"    Duración   : {df.index.max() - df.index.min()}")
    print(f"    Grid 5 min : {'OK' if intervalo_ok else 'IRREGULAR -- revisar'}")
    print(f"    Nulos      : {df.isna().sum().sum()} (total en todas las columnas)")

    return df