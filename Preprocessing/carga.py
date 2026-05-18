import pandas as pd
from config import INPUT_FILE, TIME_COL

# CARGAR LOS DATOS

def cargar_datos(filepath: str = INPUT_FILE) -> pd.DataFrame:
    print("=" * 68)
    print("PREPROCESSING |  Dataset HUPA-UCM")
    print("=" * 68)
    print(f"\n[0] Cargando archivo: {filepath}")

    df = pd.read_csv(filepath, sep=";")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.set_index(TIME_COL).sort_index()

    intervalo_ok = df.index.to_series().diff().dropna().nunique() == 1
    print(f"    Registros  : {len(df):,}")
    print(f"    Columnas   : {list(df.columns)}")
    print(f"    Rango      : {df.index.min().date()}  ->  {df.index.max().date()}")
    print(f"    Duración   : {df.index.max() - df.index.min()}")
    print(f"    Grid 5 min : {'OK' if intervalo_ok else 'IRREGULAR -- revisar'}")
    print(f"    Nulos      : {df.isna().sum().sum()} (total en todas las columnas)")

    return df