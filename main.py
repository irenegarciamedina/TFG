import Preprocessing.carga as carga
import Preprocessing.rango_sensor as rango_sensor
import Preprocessing.suavizado as suavizado
import Preprocessing.tiempo_ciclico as tiempo_ciclico
import Preprocessing.visualizacion as visualizacion
from config import OUTPUT_FILE

def preprocessing():

    # Carga
    df = carga.cargar_datos()
    
    # Rango del sensor
    df = rango_sensor.corregir_rango_sensor(df)
    
    # Suavizado
    df = suavizado.suavizar_senal(df)
    
    # Tiempo cíclico
    df = tiempo_ciclico.codificar_tiempo_ciclico(df)
    
    # Guardar resultado final
    df.to_csv(OUTPUT_FILE)

    visualizacion.generar_diagnostico(df)

    print(f"\n[FIN] Proceso completado. Archivo guardado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocessing()