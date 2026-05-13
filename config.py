import os

# DIRECTORIO DE DATOS
DATOS_DIR = "Datos"

# ---------------------------------------------------------------------------
# RUTAS DE ARCHIVOS
# estas variables se sobreescriben en main.py antes de cada llamada.
# NOTA: no se lee sys.argv aquí para evitar efectos secundarios al importar el módulo desde contextos distintos a la línea de comandos

INPUT_FILE  = os.path.join(DATOS_DIR, "HUPA0001P.csv")
OUTPUT_FILE = INPUT_FILE.replace(".csv", "_preprocessing.csv")

# Patrón para nombrar las salidas de cada paciente (usado en main.py)
OUTPUT_FILE_PATTERN = os.path.join(DATOS_DIR, "{patient}_preprocessing.csv")

# ---------------------------------------------------------------------------
# RUTAS DE SALIDA DEL PREPROCESAMIENTO
# ---------------------------------------------------------------------------
# Reporte acumulado: todos los pacientes, uno debajo del otro
REPORT_FILE = os.path.join("Preprocessing", "output", "Preprocessing.txt")
# PNG individual por paciente: se construye en visualizacion.generar_diagnostico
# pasando el patient_id; el patrón es Preprocessing/output/<patient_id>.png
PLOT_FILE   = os.path.join("Preprocessing", "output", "Preprocessing.png")   # fallback legacy

def plot_file_paciente(patient_id: str) -> str:
    """Devuelve la ruta del PNG de diagnóstico para un paciente concreto."""
    return os.path.join("Preprocessing", "output", f"{patient_id}.png")

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