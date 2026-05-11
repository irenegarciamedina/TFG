import Preprocessing.carga as carga
import Preprocessing.rango_sensor as rango_sensor
import Preprocessing.suavizado as suavizado
import Feature_Engineering.tiempo_ciclico as tiempo_ciclico
import Preprocessing.visualizacion as visualizacion
import Feature_Engineering.IOB_insulina_activa as iob
import Feature_Engineering.COB_carbohidratos_activos as cob
import ML_Exploratorio.random_forest as rf
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
    
    # Insulina activa IOB
    df = iob.compute_iob(df)

    # Carbohidratos activos COB
    df = cob.compute_cob(df)

    # Guardar resultado final
    df.to_csv(OUTPUT_FILE)

    visualizacion.generar_diagnostico(df)

    print(f"\n[FIN] Proceso completado. Archivo guardado en: {OUTPUT_FILE}")

    ML()

def ML():
    # random forest
    df = rf.ejecutar_random_forest()

if __name__ == "__main__":
    preprocessing()