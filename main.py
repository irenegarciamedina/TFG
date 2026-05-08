import Preprocessing.carga as carga
import Preprocessing.rango_sensor as rango_sensor
import Preprocessing.suavizado as suavizado
import Preprocessing.tiempo_ciclico as tiempo_ciclico
import Preprocessing.visualizacion as visualizacion
from config import OUTPUT_FILE

def ejecutar_fase1():
    
    # Paso 0: Carga
    df = carga.cargar_datos()
    
    # Paso 1: Rango del sensor
    df = rango_sensor.corregir_rango_sensor(df)
    
    # Paso 2: Suavizado
    df = suavizado.suavizar_senal(df)
    
    # Paso 3: Tiempo cíclico
    df = tiempo_ciclico.codificar_tiempo_ciclico(df)
    
    # Guardar resultado final
    df.to_csv(OUTPUT_FILE)

    visualizacion.generar_diagnostico(df)

    print(f"\n[FIN] Proceso completado. Archivo guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    ejecutar_fase1()