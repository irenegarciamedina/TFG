"""
config.py — Configuración centralizada de la Fase 1
Todos los módulos importan sus constantes desde aquí.
Para cambiar cualquier parámetro basta con editar este único archivo.
"""

import sys

# ------------------------------------------------------------------------------
# RUTAS DE ARCHIVOS
# ------------------------------------------------------------------------------
INPUT_FILE  = sys.argv[1] if len(sys.argv) > 1 else "Datos/HUPA0001P.csv"
OUTPUT_FILE = INPUT_FILE.replace(".csv", "_preprocessing.csv")
REPORT_FILE = "Preprocessing/output/Preprocessing.txt"
PLOT_FILE   = "Preprocessing/output/Preprocessing.png"

# ------------------------------------------------------------------------------
# COLUMNAS DEL DATASET HUPA-UCM
# ------------------------------------------------------------------------------
TIME_COL    = "time"
GLUCOSE_COL = "glucose"

# ------------------------------------------------------------------------------
# PARÁMETROS DEL SENSOR FREESTYLE LIBRE 2 (Abbott)
# ------------------------------------------------------------------------------
SENSOR_MAX = 400.0   # mg/dL — límite superior certificado del sensor
SENSOR_MIN = 40.0    # mg/dL — límite inferior certificado del sensor

# ------------------------------------------------------------------------------
# PARÁMETROS DE SUAVIZADO
# ------------------------------------------------------------------------------
ROLLING_WINDOW = 3   # intervalos × 5 min = ventana de 15 min centrada