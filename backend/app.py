
import os
import json
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'chama-sos-dashboard-secret-key-2024')
CORS(app)

MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client["bombeiros"]
collection = db["bombeiros"]

model = None
encoders = {}
try:
    import pickle
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    print("Modelo e encoders carregados com sucesso!")
except:
    print("Modelo não encontrado. Execute train_model.py primeiro.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard-stats')
def api_dashboard_stats():
    try:
        total_ocorrencias = collection.count_documents({})
        
        pipeline = [
            {
                '$group': {
                    '_id': None,
                    'tempo_medio': {'$avg': '$Tempo de Resposta (minutos)'},
                    'vitimas_total': {'$sum': '$Número de Vítimas'},
                    'danos_total': {'$sum': '$Danos Estimados (R$)'},
                    'vitimas_media': {'$avg': '$Número de Vítimas'},
                    'danos_medio': {'$avg': '$Danos Estimados (R$)'}
                }
            }
        ]
        
        stats = list(collection.aggregate(pipeline))
        if stats:
            stats = stats[0]
        else:
            stats = {
                'tempo_medio': 0,
                'vitimas_total': 0,
                'danos_total': 0,
                'vitimas_media': 0,
                'danos_medio': 0
            }
        
        pipeline_grupo = [
            {'$group': {'_id': '$Grupo de Ocorrência', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]
        
        pipeline_prioridade = [
            {'$group': {'_id': '$Prioridade', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]
        
        grupos = list(collection.aggregate(pipeline_grupo))
        prioridades = list(collection.aggregate(pipeline_prioridade))
        
        dist_grupo = {item['_id']: item['count'] for item in grupos}
        dist_prioridade = {item['_id']: item['count'] for item in prioridades}
        
        return jsonify({
            'success': True,
            'stats': {
                'total_ocorrencias': total_ocorrencias,
                'tempo_resposta_medio': round(stats['tempo_medio'], 2),
                'total_vitimas': stats['vitimas_total'],
                'danos_totais': round(stats['danos_total'], 2),
                'vitimas_media': round(stats['vitimas_media'], 2),
                'danos_medios': round(stats['danos_medio'], 2)
            },
            'distributions': {
                'grupo': dist_grupo,
                'prioridade': dist_prioridade
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao obter estatísticas: {str(e)}'
        }), 500

@app.route('/api/recent-occurrences')
def api_recent_occurrences():
    try:
        cursor = collection.find({}, {'_id': 0}).sort('Data', -1).limit(20)
        occurrences = list(cursor)
        
        for occ in occurrences:
            if 'Data' in occ and isinstance(occ['Data'], datetime):
                occ['Data'] = occ['Data'].isoformat()
        
        return jsonify({
            'success': True,
            'occurrences': occurrences
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao obter ocorrências: {str(e)}'
        }), 500

@app.route('/api/add-occurrence', methods=['POST'])
def api_add_occurrence():
    try:
        data = request.json
        
        required_fields = ['Grupo de Ocorrência', 'Tipo de Ocorrência', 'Prioridade']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'message': f'Campo obrigatório faltando: {field}'
                }), 400
        
        data['Data'] = datetime.now()
        data['ID'] = collection.count_documents({}) + 1
        
        numeric_fields = ['Tempo de Resposta (minutos)', 'Número de Vítimas', 'Danos Estimados (R$)']
        for field in numeric_fields:
            if field in data:
                data[field] = float(data[field])
        
        result = collection.insert_one(data)
        
        return jsonify({
            'success': True,
            'message': 'Ocorrência adicionada com sucesso!',
            'id': data['ID']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao adicionar ocorrência: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if model is None:
            return jsonify({
                'success': False,
                'message': 'Modelo não treinado. Execute train_model.py primeiro.'
            }), 400
        
        data = request.json
        
        input_data = {}
        for feature in model.feature_names_in_:
            if feature in data:
                if feature in encoders:
                    try:
                        input_data[feature] = encoders[feature].transform([data[feature]])[0]
                    except:
                        input_data[feature] = 0
                else:
                    input_data[feature] = float(data[feature])
            else:
                input_data[feature] = 0
        
        import pandas as pd
        input_df = pd.DataFrame([input_data])
        
        prediction = model.predict(input_df)[0]
        
        interpretation = interpret_prediction(prediction)
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'interpretation': interpretation
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro na predição: {str(e)}'
        }), 500

@app.route('/api/model-info')
def api_model_info():
    if model is None:
        return jsonify({
            'success': False,
            'message': 'Modelo não disponível'
        })
    
    try:
        with open('models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(model.feature_names_in_, model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
        else:
            feature_importance = {}
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'n_features': len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 0
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao obter informações do modelo: {str(e)}'
        }), 500

@app.route('/api/init-database', methods=['POST'])
def api_init_database():
    try:
        from data_generator import generate_sample_data
        count = generate_sample_data()
        return jsonify({
            'success': True,
            'message': f'Banco de dados inicializado com {count} registros de exemplo.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao inicializar banco: {str(e)}'
        }), 500

def interpret_prediction(prediction):
    if prediction <= 15:
        return {
            'level': 'RÁPIDO',
            'color': 'success',
            'icon': '⚡',
            'description': 'Tempo de resposta excelente'
        }
    elif prediction <= 30:
        return {
            'level': 'MÉDIO',
            'color': 'warning',
            'icon': '⏱️',
            'description': 'Tempo de resposta dentro do esperado'
        }
    else:
        return {
            'level': 'LENTO',
            'color': 'danger',
            'icon': '🐌',
            'description': 'Tempo de resposta acima do ideal'
        }

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("Iniciando Dashboard Chama SOS...")
    print(f"MongoDB: {MONGO_URI}")
    print("Acesse: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)