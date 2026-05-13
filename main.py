import sys
import glob
import os

import Preprocessing.carga as carga
import Preprocessing.rango_sensor as rango_sensor
import Preprocessing.suavizado as suavizado
import Preprocessing.visualizacion as visualizacion
import Feature_Engineering.tiempo_ciclico as tiempo_ciclico
import Feature_Engineering.IOB_insulina_activa as iob
import Feature_Engineering.COB_carbohidratos_activos as cob
import ML.random_forest as rf
import ML.SVM as svm
import ML.clarke_error_grid as ceg

from config import OUTPUT_FILE_PATTERN, DATOS_DIR

# Número de pacientes para entrenamiento. El resto son utilizados para test
N_TRAIN_PATIENTS = 20


# ---------------------------------------------------------------------------
# PREPROCESAMIENTO DE UN ÚNICO PACIENTE
# ---------------------------------------------------------------------------

def preprocesar_paciente(input_path: str) -> str:
    import config as cfg
    cfg.INPUT_FILE  = input_path
    cfg.OUTPUT_FILE = input_path.replace(".csv", "_preprocessing.csv")

    patient_id = os.path.splitext(os.path.basename(input_path))[0]

    df = carga.cargar_datos(filepath=input_path)
    df = rango_sensor.corregir_rango_sensor(df)
    df = suavizado.suavizar_senal(df)
    df = tiempo_ciclico.codificar_tiempo_ciclico(df)
    df = iob.compute_iob(df)
    df = cob.compute_cob(df)

    df.to_csv(cfg.OUTPUT_FILE)
    visualizacion.generar_diagnostico(df, patient_id=patient_id)

    print(f"  [OK] {os.path.basename(input_path)} -> {cfg.OUTPUT_FILE}\n")
    return cfg.OUTPUT_FILE


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def main():

    if len(sys.argv) > 1:
        ficheros = [sys.argv[1]]
    else:
        patron   = os.path.join(DATOS_DIR, "HUPA*.csv")
        ficheros = sorted([
            f for f in glob.glob(patron)
            if "_preprocessing" not in f
        ])
        if not ficheros:
            print(f"[ERROR] No se encontraron ficheros CSV en {DATOS_DIR}")
            sys.exit(1)

    print("=" * 68)
    print(f"  Pacientes totales : {len(ficheros)}")
    print(f"  Train             : {min(N_TRAIN_PATIENTS, len(ficheros))} pacientes")
    print(f"  Test              : {max(0, len(ficheros) - N_TRAIN_PATIENTS)} pacientes")
    print("=" * 68)

    # Resetear el reporte acumulado de preprocesamiento para esta ejecución
    import config as cfg_root
    os.makedirs(os.path.dirname(cfg_root.REPORT_FILE), exist_ok=True)
    open(cfg_root.REPORT_FILE, "w").close()

    # PREPROCESAMIENTO

    csv_preprocesados = []
    for i, fichero in enumerate(ficheros, 1):
        print(f"\n{'='*68}")
        print(f"  PACIENTE {i}/{len(ficheros)}: {os.path.basename(fichero)}")
        print(f"{'='*68}")
        try:
            out = preprocesar_paciente(fichero)
            csv_preprocesados.append(out)
        except Exception as e:
            print(f"  [WARN] Error procesando {fichero}: {e}. Se omite.")

    print(f"\n[INFO] Preprocesamiento completado: {len(csv_preprocesados)}/{len(ficheros)} pacientes OK")

    if not csv_preprocesados:
        print("[ERROR] Ningún fichero preprocesado disponible. Abortando ML.")
        sys.exit(1)

    ejecutar_ml(csv_preprocesados)


# ---------------------------------------------------------------------------
# FASE ML  —  división 20 train / 5 test por paciente
# ---------------------------------------------------------------------------

def ejecutar_ml(csv_paths: list):
    import ML.config as ml_cfg

    train_paths = csv_paths[:N_TRAIN_PATIENTS]
    test_paths  = csv_paths[N_TRAIN_PATIENTS:]

    print("\n" + "=" * 68)
    print(f"  FASE ML")
    print(f"  Train : {len(train_paths)} pacientes -> {[os.path.basename(p) for p in train_paths]}")
    print(f"  Test  : {len(test_paths)}  pacientes -> {[os.path.basename(p) for p in test_paths]}")
    print("=" * 68)

    # Actualizamos el config de ML
    ml_cfg.INPUT_FILES  = csv_paths
    ml_cfg.TRAIN_FILES  = train_paths
    ml_cfg.TEST_FILES   = test_paths

    resultado_rf = rf.ejecutar_random_forest()
    svm.ejecutar_svm()

    # Clarke Error Grid (evaluación clínica del Random Forest)
    print("\n" + "=" * 68)
    print("  CLARKE ERROR GRID")
    print("=" * 68)
    metricas_rf = resultado_rf.get("metricas", {})
    y_test_rf   = metricas_rf.get("y_test")
    y_pred_rf   = metricas_rf.get("y_pred_test")

    if y_test_rf is not None and y_pred_rf is not None:
        ceg.generar_clarke_error_grid(y_test_rf, y_pred_rf, test_files=test_paths)
    else:
        print("[WARN] No se pudieron obtener las predicciones del RF para la CEG.")



if __name__ == "__main__":
    main()