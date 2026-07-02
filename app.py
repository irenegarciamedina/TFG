from flask import Flask, jsonify, render_template, request, send_from_directory
import pandas as pd
import os
import re
import logging

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATOS_DIR = os.path.join(BASE_DIR, "datos")
ML_OUTPUT_DIR = os.path.join(BASE_DIR, "ML", "output")
PREPROCESSING_OUTPUT_DIR = os.path.join(BASE_DIR, "Preprocessing", "output")

@app.route('/')
def index():

    # renderiza la plantilla HTML que se guardará en la carpeta templates
    return render_template('index.html')


# endpoint para listar todos los pacientes disponibles en la carpeta datos
@app.route('/api/pacientes')
def list_pacientes():
    if not os.path.exists(DATOS_DIR):
        return jsonify([])
    
    # busca archivos que terminen en _preprocessing.csv para identificar los pacientes limpios
    archivos = os.listdir(DATOS_DIR)
    pacientes = []
    for f in archivos:
        if f.endswith('_preprocessing.csv'):
            id_paciente = f.replace('_preprocessing.csv', '')
            pacientes.append({
                "id": id_paciente,
                "archivo": f
            })
    
    # los ordena alfabéticamente
    pacientes.sort(key=lambda x: x['id'])
    return jsonify(pacientes)


# endpoint para obtener los datos temporales de un paciente específico
@app.route('/api/data/<paciente_id>')
def get_data(paciente_id):
    archivo_csv = f"{paciente_id}_preprocessing.csv"
    csv_path = os.path.join(DATOS_DIR, archivo_csv)
    
    if not os.path.exists(csv_path):
        return jsonify({"error": f"Paciente {paciente_id} no encontrado."}), 404
    
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # estructura el payload básico. Si faltan columnas de clasificación, se envían vacías
    payload = {
        "time": df['time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        "glucose": df['glucose'].tolist() if 'glucose' in df.columns else [],
        "iob": df['iob'].tolist() if 'iob' in df.columns else [],
        "cob": df['cob'].tolist() if 'cob' in df.columns else [],
        "heart_rate": df['heart_rate'].tolist() if 'heart_rate' in df.columns else [],
        "steps": df['steps'].tolist() if 'steps' in df.columns else []
    }
    
    return jsonify(payload)


# extrae el segmento de texto de Preprocessing.txt para el paciente solicitado
@app.route('/api/reporte/preprocesamiento/<paciente_id>')
def get_reporte_preprocesamiento(paciente_id):
    path_txt = os.path.join(PREPROCESSING_OUTPUT_DIR, "Preprocessing.txt")
    if not os.path.exists(path_txt):
        return jsonify({"text": "Archivo Preprocessing.txt no encontrado."})
    
    with open(path_txt, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
        
    bloque_lineas = []
    guardando = False
    
    # busca de forma determinista la sección del paciente
    for linea in lineas:
        if f"PACIENTE : {paciente_id}" in linea:
            guardando = True
            bloque_lineas.append("================================================================================\n")
            
        if guardando:
            # Si nos cruzamos con el título del SIGUIENTE paciente, dejamos de acumular
            if "PACIENTE :" in linea and f"PACIENTE : {paciente_id}" not in linea:
                break
            bloque_lineas.append(linea)

    if bloque_lineas:

        # asegura que el bloque cierre estéticamente con una barra
        if not bloque_lineas[-1].startswith("===="):
            bloque_lineas.append("\n================================================================================")
        return jsonify({"text": "".join(bloque_lineas)})
        
    return jsonify({"text": f"================================================================================\nNo se encontró reporte específico para {paciente_id}\n================================================================================"})


# endpoint que lee y envía los bloques estructurados de ML_reporte.txt
@app.route('/api/reporte/ml')
def get_reporte_ml():
    path_txt = os.path.join(ML_OUTPUT_DIR, "ML_reporte.txt")
    if not os.path.exists(path_txt):
        return jsonify({"error": "No encontrado"}), 404
        
    with open(path_txt, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # mapea los identificadores únicos que inician cada bloque real en tu txt
    tags = {
        "rf": "RANDOM FOREST: RANKING DE IMPORTANCIA",
        "xgb": "XGBOOST: PREDICCIÓN DE GLUCOSA",
        "ridge": "RIDGE REGRESSION (BASELINE LINEAL):",
        "lstm": "LSTM: PREDICCIÓN DE GLUCOSA",
        "svm": "SVM: CLASIFICACIÓN DE CAÍDAS BRUSCAS",
        "gbt": "GRADIENT BOOSTING: CLASIFICACIÓN DE CAÍDAS BRUSCAS",
        "lr": "LOGISTIC REGRESSION: CLASIFICACIÓN DE CAÍDAS BRUSCAS",
        "comparativa": "COMPARATIVA DE MODELOS"
    }
    
    lineas = content.splitlines()
    reportes = {}
    
    # extrae cada bloque buscando dónde empieza su clave y dónde empieza cualquier otra clave posterior
    for key, tag in tags.items():
        bloque_lineas = []
        guardando = False
        
        for linea in lineas:
            if tag in linea:
                guardando = True
                bloque_lineas.append("================================================================================\n")
            
            if guardando:

                # comprueba si la línea actual pertenece a otro bloque distinto para detener la captura
                es_otro_tag = any(other_tag in linea for other_key, other_tag in tags.items() if other_key != key)
                if es_otro_tag and tag not in linea:
                    break
                bloque_lineas.append(linea + "\n")
                
        if bloque_lineas:
            if not bloque_lineas[-1].strip().startswith("===="):
                bloque_lineas.append("================================================================================\n")
            reportes[key] = "".join(bloque_lineas)
        else:
            reportes[key] = f"Bloque {tag} no localizado en el reporte."

    # crea el texto de la CEG porque no se incluye en el reporte ML_reporte.txt
    reportes["ceg"] = "================================================================================\nCLARKE ERROR GRID ANALYSIS (CEG)\n================================================================================\nEvaluación de la precisión clínica de las predicciones de glucosa en zonas de riesgo (A, B, C, D, E) para garantizar la seguridad del paciente ante decisiones automatizadas de dosificación."
    
    return jsonify(reportes)


# endpoint para servir de manera segura las imágenes de los modelos y comparativas desde ML/output
@app.route('/api/graficas/<filename>')
def get_ml_graph(filename):

    # protege contra ataques de trayectoria básica
    safe_filename = os.path.basename(filename)
    return send_from_directory(ML_OUTPUT_DIR, safe_filename)


# endpoint para servir los gráficos resultantes del preprocesamiento desde preprocessing/output
@app.route('/api/preprocesamiento/<filename>')
def get_preprocessing_graph(filename):
    safe_filename = os.path.basename(filename)
    return send_from_directory(PREPROCESSING_OUTPUT_DIR, safe_filename)


if __name__ == '__main__':

    # oculta los logs de werkzeug para que no se muestren en la consola
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    # evita duplicar el print con el puerto
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        print(" Por favor, abre en tu navegador: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)