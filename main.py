import Preprocessing.carga as carga
import Preprocessing.paso1_rango_sensor as paso1_rango_sensor
import Preprocessing.paso2_suavizado as paso2_suavizado
import Preprocessing.paso3_tiempo_ciclico as paso3_tiempo_ciclico
import Preprocessing.visualizacion as visualizacion
from config import OUTPUT_FILE

def ejecutar_fase1():
    # Paso 0: Carga
    df = carga.cargar_datos()
    
    # Paso 1: Rango del sensor
    df = paso1_rango_sensor.corregir_rango_sensor(df)
    
    # Paso 2: Suavizado
    df = paso2_suavizado.suavizar_senal(df)
    
    # Paso 3: Tiempo cíclico
    df = paso3_tiempo_ciclico.codificar_tiempo_ciclico(df)
    
    # Guardar resultado final
    df.to_csv(OUTPUT_FILE)

    visualizacion.generar_diagnostico(df)

    print(f"\n[FIN] Proceso completado. Archivo guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    ejecutar_fase1()