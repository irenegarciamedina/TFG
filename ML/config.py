
# División TRAIN/TEST por paciente:
# 20 entrenan, 5 evalúan.

import os
import glob

# RUTAS
_patron_default = os.path.join("Datos", "*_preprocessing.csv")
INPUT_FILES = sorted(glob.glob(_patron_default))

# División por paciente: 20 train / 5 test
# main.py sobreescribe estas listas después del preprocesamiento
N_TRAIN_PATIENTS = 20
TRAIN_FILES = INPUT_FILES[:N_TRAIN_PATIENTS]
TEST_FILES  = INPUT_FILES[N_TRAIN_PATIENTS:]

INPUT_FILE = INPUT_FILES[0] if INPUT_FILES else os.path.join("Datos", "HUPA0001P_preprocessing.csv")

# Salidas
OUTPUT_DIR    = os.path.join("ML", "output")
PLOT_RF       = os.path.join(OUTPUT_DIR, "RF_importancia_features.png")
PLOT_SVM      = os.path.join(OUTPUT_DIR, "SVM_clasificacion.png")
PLOT_CEG      = os.path.join(OUTPUT_DIR, "RF_clarke_error_grid.png")
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
SVM_C            = 1.0
SVM_KERNEL       = "rbf"
SVM_GAMMA        = "scale"
SVM_CLASS_WEIGHT = "balanced"   # compensa el desbalance ~16:1 ruido/caída

HYPO_THRESHOLD = 70
DROP_STEPS     = 3
DROP_THRESHOLD = -15

# ---------------------------------------------------------------------------
# DIVISIÓN TRAIN / TEST
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.80  