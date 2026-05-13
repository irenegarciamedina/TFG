"""
main.py  — Pipeline completo para TODOS los pacientes del dataset HUPA-UCM
---------------------------------------------------------------------------
Uso:
    python main.py                   # procesa todos los pacientes en Datos/
    python main.py Datos/HUPA0001P.csv   # procesa solo ese fichero (modo anterior)
"""

import sys
import glob
import os

import Preprocessing.carga          as carga
import Preprocessing.rango_sensor   as rango_sensor
import Preprocessing.suavizado      as suavizado
import Preprocessing.visualizacion  as visualizacion
import Feature_Engineering.tiempo_ciclico          as tiempo_ciclico
import Feature_Engineering.IOB_insulina_activa     as iob
import Feature_Engineering.COB_carbohidratos_activos as cob
import ML.random_forest as rf
import ML.SVM           as svm

from config import OUTPUT_FILE_PATTERN, DATOS_DIR


# ---------------------------------------------------------------------------
# PREPROCESAMIENTO DE UN ÚNICO PACIENTE
# ---------------------------------------------------------------------------

def preprocesar_paciente(input_path: str) -> str:
    """
    Ejecuta el pipeline de preprocesamiento para un fichero CSV.
    Devuelve la ruta del CSV preprocesado generado.
    """
    # Importamos config aquí para que sys.argv no interfiera en el bucle
    import config as cfg
    cfg.INPUT_FILE  = input_path
    cfg.OUTPUT_FILE = input_path.replace(".csv", "_preprocessing.csv")

    df = carga.cargar_datos(filepath=input_path)
    df = rango_sensor.corregir_rango_sensor(df)
    df = suavizado.suavizar_senal(df)
    df = tiempo_ciclico.codificar_tiempo_ciclico(df)
    df = iob.compute_iob(df)
    df = cob.compute_cob(df)

    df.to_csv(cfg.OUTPUT_FILE)
    visualizacion.generar_diagnostico(df)

    print(f"  [OK] {os.path.basename(input_path)} -> {cfg.OUTPUT_FILE}\n")
    return cfg.OUTPUT_FILE


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def main():

    # ---- Decidir qué ficheros procesar ----
    if len(sys.argv) > 1:
        # Modo compatibilidad: se pasa un fichero concreto
        ficheros = [sys.argv[1]]
    else:
        # Modo completo: todos los CSV de la carpeta Datos/ (excluye los ya preprocesados)
        patron = os.path.join(DATOS_DIR, "HUPA*.csv")
        ficheros = sorted([
            f for f in glob.glob(patron)
            if "_preprocessing" not in f
        ])
        if not ficheros:
            print(f"[ERROR] No se encontraron ficheros CSV en {DATOS_DIR}")
            sys.exit(1)

    print("=" * 68)
    print(f"  Pacientes a procesar: {len(ficheros)}")
    for f in ficheros:
        print(f"    - {os.path.basename(f)}")
    print("=" * 68)

    # ---- Preprocesamiento de cada paciente ----
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

    # ---- Fase ML sobre el conjunto combinado ----
    if not csv_preprocesados:
        print("[ERROR] Ningún fichero preprocesado disponible. Abortando ML.")
        sys.exit(1)

    ejecutar_ml(csv_preprocesados)


# ---------------------------------------------------------------------------
# FASE ML
# ---------------------------------------------------------------------------

def ejecutar_ml(csv_paths: list):
    """
    Entrena Random Forest y SVM combinando todos los CSV preprocesados.
    """
    print("\n" + "=" * 68)
    print(f"  FASE ML — {len(csv_paths)} paciente(s)")
    print("=" * 68)

    # Actualizamos el config de ML para que apunte a la lista de ficheros
    import ML.config as ml_cfg
    ml_cfg.INPUT_FILES = csv_paths   # nueva variable (ver ML/config.py)

    rf.ejecutar_random_forest()
    svm.ejecutar_svm()


if __name__ == "__main__":
    main()