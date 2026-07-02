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
        content = f.read()
    
    # divide el archivo por bloques de pacientes
    bloques = content.split("================================================================================")
    for b in bloques:
        if f"PACIENTE : {paciente_id}" in b:
            return jsonify({"text": "================================================================================\n" + b.strip() + "\n================================================================================"})
            
    return jsonify({"text": f"No se encontró reporte específico para {paciente_id}"})


# endpoint que lee y envía los bloques estructurados de ML_reporte.txt
@app.route('/api/reporte/ml')
def get_reporte_ml():
    path_txt = os.path.join(ML_OUTPUT_DIR, "ML_reporte.txt")
    if not os.path.exists(path_txt):
        return jsonify({"error": "No encontrado"}), 404
        
    with open(path_txt, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # segmenta usando expresiones regulares basadas en los delimitadores
    bloques = re.split(r'={10,}', content)
    
    # Mapeo manual de los bloques del archivo
    reportes = {
        "rf": bloques[2].strip() if len(bloques) > 2 else "",
        "xgb": bloques[4].strip() if len(bloques) > 4 else "",
        "ridge": bloques[6].strip() if len(bloques) > 6 else "",
        "lstm": bloques[8].strip() if len(bloques) > 8 else "",
        "ceg": "Análisis mediante Clarke Error Grid (CEG) para evaluar la precisión clínica de las predicciones de glucosa en zonas de riesgo.",
        "svm": bloques[10].strip() if len(bloques) > 10 else "",
        "lr": bloques[14].strip() if len(bloques) > 14 else "",
        "gbt": bloques[12].strip() if len(bloques) > 12 else "",
        "comparativa": content.split("COMPARATIVA DE MODELOS")[-1].replace("===", "").strip() if "COMPARATIVA DE MODELOS" in content else ""
    }
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