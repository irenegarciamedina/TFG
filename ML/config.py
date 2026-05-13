"""
ML/config.py
---------------------------------------------------------------------------
Configuración de la fase de Machine Learning.
Ahora soporta múltiples pacientes mediante INPUT_FILES (lista de rutas).
"""

import os
import glob

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------

# Lista de ficheros preprocesados.
# - Se rellena desde main.py cuando se procesan varios pacientes.
# - Por compatibilidad, si se ejecuta ML directamente se cargan todos los que
#   existan en la carpeta Datos/.
_patron_default = os.path.join("Datos", "*_preprocessing.csv")
INPUT_FILES = sorted(glob.glob(_patron_default))   # puede estar vacía si aún no se ha preprocesado

# Compatibilidad con código antiguo que usaba INPUT_FILE (singular)
INPUT_FILE = INPUT_FILES[0] if INPUT_FILES else os.path.join("Datos", "HUPA0001P_preprocessing.csv")

# Salidas de esta fase
OUTPUT_DIR    = os.path.join("ML", "output")
PLOT_RF       = os.path.join(OUTPUT_DIR, "RF_importancia_features.png")
PLOT_SVM      = os.path.join(OUTPUT_DIR, "SVM_clasificacion.png")
PLOT_BASELINE = os.path.join(OUTPUT_DIR, "Baseline_comparativa.png")
REPORT_FILE   = os.path.join(OUTPUT_DIR, "ML_reporte.txt")

# ---------------------------------------------------------------------------
# COLUMNA OBJETIVO
# ---------------------------------------------------------------------------
GLUCOSE_COL = "glucose"

# ---------------------------------------------------------------------------
# HORIZONTE DE PREDICCIÓN
# ---------------------------------------------------------------------------
HORIZON_STEPS = 8          # 8 × 5 min = 40 minutos
HORIZON_MIN   = HORIZON_STEPS * 5

# ---------------------------------------------------------------------------
# FEATURES DEL MODELO
# ---------------------------------------------------------------------------
FEATURES = [
    'glucose',
    'iob',
    'cob',
    'heart_rate',
    'basal_rate',
    'time_hour_cos',
    'time_hour_sin',
    'steps',
    'time_dow_sin'
]

FEATURES_OPCIONALES = [
    "bolus_volume_delivered",
    "carb_input",
]

TARGET    = 'glucose_target'
HORIZONTE = 8

# ---------------------------------------------------------------------------
# HIPERPARÁMETROS — RANDOM FOREST
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS  = 300
RF_MAX_DEPTH     = None
RF_MIN_SAMPLES   = 10
RF_RANDOM_STATE  = 42

# ---------------------------------------------------------------------------
# HIPERPARÁMETROS — SVM
# ---------------------------------------------------------------------------
SVM_C      = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA  = "scale"

HYPO_THRESHOLD = 70      # mg/dL
DROP_STEPS     = 3
DROP_THRESHOLD = -15     # mg/dL en 15 min

# ---------------------------------------------------------------------------
# DIVISIÓN TRAIN / TEST  (división temporal, no aleatoria)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.80