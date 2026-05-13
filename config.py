"""
config.py  (raíz del proyecto)
---------------------------------------------------------------------------
Configuración global del pipeline de preprocesamiento.
"""

import sys
import os

# ---------------------------------------------------------------------------
# DIRECTORIO DE DATOS
# ---------------------------------------------------------------------------
DATOS_DIR = "Datos"

# ---------------------------------------------------------------------------
# RUTAS DE ARCHIVOS
# Cuando se ejecuta sobre un único paciente (modo compatibilidad o bucle interno)
# estas variables se sobreescriben en main.py antes de cada llamada.
# ---------------------------------------------------------------------------
INPUT_FILE  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATOS_DIR, "HUPA0001P.csv")
OUTPUT_FILE = INPUT_FILE.replace(".csv", "_preprocessing.csv")

# Patrón para nombrar las salidas de cada paciente (usado en main.py)
OUTPUT_FILE_PATTERN = os.path.join(DATOS_DIR, "{patient}_preprocessing.csv")

# ---------------------------------------------------------------------------
# RUTAS DE SALIDA DEL PREPROCESAMIENTO
# Ahora incluyen el nombre del paciente para no sobrescribirse entre pacientes.
# ---------------------------------------------------------------------------
REPORT_FILE = os.path.join("Preprocessing", "output", "Preprocessing.txt")
PLOT_FILE   = os.path.join("Preprocessing", "output", "Preprocessing.png")

# ---------------------------------------------------------------------------
# COLUMNAS DEL DATASET HUPA-UCM
# ---------------------------------------------------------------------------
TIME_COL    = "time"
GLUCOSE_COL = "glucose"

# ---------------------------------------------------------------------------
# PARÁMETROS DEL SENSOR FREESTYLE LIBRE 2 (Abbott)
# ---------------------------------------------------------------------------
SENSOR_MAX = 400.0   # mg/dL — límite superior certificado del sensor
SENSOR_MIN = 40.0    # mg/dL — límite inferior certificado del sensor

# ---------------------------------------------------------------------------
# PARÁMETROS DE SUAVIZADO
# ---------------------------------------------------------------------------
ROLLING_WINDOW = 3   # intervalos × 5 min = ventana de 15 min centrada