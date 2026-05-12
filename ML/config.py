import os

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------

INPUT_FILE    = os.path.join("Datos", "HUPA0001P_preprocessing.csv")

# Salidas de esta fase
OUTPUT_DIR    = os.path.join("ML", "output")
PLOT_RF       = os.path.join(OUTPUT_DIR, "RF_importancia_features.png")
PLOT_SVM      = os.path.join(OUTPUT_DIR, "SVM_clasificacion.png")
PLOT_BASELINE = os.path.join(OUTPUT_DIR, "Baseline_comparativa.png")
REPORT_FILE   = os.path.join(OUTPUT_DIR, "ML_reporte.txt")

# ---------------------------------------------------------------------------
# COLUMNA OBJETIVO
# ---------------------------------------------------------------------------

GLUCOSE_COL = "glucose"          # columna de glucosa limpia (misma de fase 1)

# ---------------------------------------------------------------------------
# HORIZONTE DE PREDICCIÓN
# ---------------------------------------------------------------------------

HORIZON_STEPS = 8                # 8 × 5 min = 40 minutos
HORIZON_MIN   = HORIZON_STEPS * 5

# como el sensor mide cada 5 minutosy hay 8 pasos, el total es de 40 minutos
# es el horizonte clínico estándar para las alertas de hipoglucemia 

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

# Features opcionales (se añaden si existen en el CSV)
FEATURES_OPCIONALES = [
    "bolus_volume_delivered",   # dosis de bolo registrada
    "carb_input",               # gramos de carbohidratos registrados
]

TARGET = 'glucose_target'
HORIZONTE = 8  # 40 minutos

# lista que se usará para los modelos

# ---------------------------------------------------------------------------
# HIPERPARÁMETROS — RANDOM FOREST
# ---------------------------------------------------------------------------

RF_N_ESTIMATORS  = 300   # cuantos más árboles haya, más estable será pero mas lento también
RF_MAX_DEPTH     = None  # none para que crezca hasta las hojas más bajas. Es los más adecuado porque hay pocas features
RF_MIN_SAMPLES   = 10    # mínimo de muestras por hoja. Evita un sobreajuste
RF_RANDOM_STATE  = 42    # semilla para reproducibilidad

# ---------------------------------------------------------------------------
# HIPERPARÁMETROS — SVM
# ---------------------------------------------------------------------------

SVM_C        = 1.0      # parámetro de regularización
SVM_KERNEL   = "rbf"    # kernel gaussiano (adecuado para datos no lineales)
SVM_GAMMA    = "scale"  # escala automática según el número de features

# Umbral para clasificar una caída como "hipoglucemia inminente"
HYPO_THRESHOLD = 70     # mg/dL — criterio ADA/FDA para TBR-1

# Caída brusca: delta de glucosa en 15 min (3 pasos × 5 min)
DROP_STEPS     = 3      # ventana para calcular la tasa de caída
DROP_THRESHOLD = -15    # mg/dL en 15 min = caída clínicamente relevante

# ---------------------------------------------------------------------------
# DIVISIÓN TRAIN / TEST
# ---------------------------------------------------------------------------
# Se usa división temporal (no aleatoria) porque los datos de glucosa son una serie temporal
# barajar crearía fuga de información del futuro al pasado

TRAIN_RATIO = 0.80      # 80% primeros días → entrenamiento