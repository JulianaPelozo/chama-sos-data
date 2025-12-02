# backend/data_generator.py - VERSÃO CORRIGIDA
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient

def init_mongodb():
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
    client = MongoClient(MONGO_URI)
    db = client["chama-sos-bombeiros"]
    return db["chama-sos-bombeiros"]

collection = init_mongodb()

def clear_existing_data():
    if collection.count_documents({}) > 0:
        print(f"⚠️  Removendo {collection.count_documents({})} registros existentes...")
        collection.delete_many({})
        return True
    return False

def generate_sample_data(n_records=2000, clear_first=False):
    
    if clear_first:
        clear_existing_data()
    elif collection.count_documents({}) > 0:
        print(f" Dados já existem ({collection.count_documents({})} registros)")
        print("   Use clear_first=True para limpar ou skip generation")
        return collection.count_documents({})
    
    print(f"Gerando {n_records} registros de exemplo...")
    
    grupos = ['Incêndio', 'Resgate', 'Acidente de Trânsito', 'Desabamento']
    tipos = ['Residencial', 'Comercial', 'Industrial', 'Público']
    prioridades = ['Baixa', 'Média', 'Alta', 'Crítica']
    localizacoes = ['Zona Norte', 'Zona Sul', 'Centro', 'Zona Leste', 'Zona Oeste']
    equipes = ['Equipe A', 'Equipe B', 'Equipe C', 'Equipe D', 'Equipe E']
    
    np.random.seed(42)
    
    data = {
        'ID': range(1, n_records + 1),
        'Grupo de Ocorrência': np.random.choice(grupos, n_records),
        'Tipo de Ocorrência': np.random.choice(tipos, n_records),
        'Prioridade': np.random.choice(prioridades, n_records),
        'Localização': np.random.choice(localizacoes, n_records),
        'Equipe Enviada': np.random.choice(equipes, n_records)
    }
    
    tempo_resposta = []
    numero_vitimas = []
    danos_estimados = []
    
    tempo_por_prioridade = {
        'Baixa': (10, 45),
        'Média': (8, 35),
        'Alta': (5, 25),
        'Crítica': (3, 15)
    }
    
    vitimas_por_grupo = {
        'Incêndio': (0, 8),
        'Resgate': (1, 6),
        'Acidente de Trânsito': (1, 10),
        'Desabamento': (2, 15)
    }
    
    danos_por_tipo = {
        'Residencial': (1000, 20000),
        'Comercial': (5000, 50000),
        'Industrial': (10000, 100000),
        'Público': (2000, 30000)
    }
    
    for i in range(n_records):
        grupo = data['Grupo de Ocorrência'][i]
        tipo = data['Tipo de Ocorrência'][i]
        prioridade = data['Prioridade'][i]
        
        min_tempo, max_tempo = tempo_por_prioridade[prioridade]
        tempo_resposta.append(np.random.randint(min_tempo, max_tempo))
        
        min_vitimas, max_vitimas = vitimas_por_grupo[grupo]
        numero_vitimas.append(np.random.randint(min_vitimas, max_vitimas))
        
        min_danos, max_danos = danos_por_tipo[tipo]
        danos_estimados.append(np.random.randint(min_danos, max_danos))
    
    data['Tempo de Resposta (minutos)'] = tempo_resposta
    data['Número de Vítimas'] = numero_vitimas
    data['Danos Estimados (R$)'] = danos_estimados
    
    start_date = datetime.now() - timedelta(days=180)
    dates = []
    for i in range(n_records):
        random_days = np.random.randint(0, 180)
        random_hours = np.random.randint(0, 24)
        random_minutes = np.random.randint(0, 60)
        date = start_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
        dates.append(date)
    
    data['Data'] = dates
    
    df = pd.DataFrame(data)
    
    collection.delete_many({})
    
    records = df.to_dict('records')
    result = collection.insert_many(records)
    
    print(f"{len(result.inserted_ids)} registros inseridos no MongoDB!")
    
    print("\n   Estatísticas dos Dados Gerados:")
    print(f"   Total de registros: {len(df)}")
    print(f"   Período: {df['Data'].min().strftime('%d/%m/%Y')} a {df['Data'].max().strftime('%d/%m/%Y')}")
    print(f"   Tempo médio de resposta: {df['Tempo de Resposta (minutos)'].mean():.1f} minutos")
    print(f"   Total de vítimas: {df['Número de Vítimas'].sum()}")
    print(f"   Danos totais: R$ {df['Danos Estimados (R$)'].sum():,.2f}")
    
    return len(df)

def export_to_json(filename='data/ocorrencias.json'):
    os.makedirs('data', exist_ok=True)
    
    cursor = collection.find({}, {'_id': 0})
    data = list(cursor)
    
    for item in data:
        if 'Data' in item and isinstance(item['Data'], datetime):
            item['Data'] = item['Data'].isoformat()
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Dados exportados para {filename}")
    return len(data)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Gerador de dados para Chama SOS')
    parser.add_argument('--n-records', type=int, default=2000, help='Número de registros a gerar')
    parser.add_argument('--clear', action='store_true', help='Limpar dados existentes antes de gerar')
    parser.add_argument('--check-only', action='store_true', help='Apenas verificar, não gerar')
    parser.add_argument('--export', action='store_true', help='Exportar para JSON após geração')
    
    args = parser.parse_args()
    
    if args.check_only:
        count = collection.count_documents({})
        print(f"Registros no MongoDB: {count}")
        if count > 0:
            pipeline = [
                {'$group': {
                    '_id': None,
                    'avg_time': {'$avg': '$Tempo de Resposta (minutos)'},
                    'total_victims': {'$sum': '$Número de Vítimas'},
                    'total_damage': {'$sum': '$Danos Estimados (R$)'}
                }}
            ]
            stats = list(collection.aggregate(pipeline))
            if stats:
                stats = stats[0]
                print(f"   Tempo médio: {stats['avg_time']:.1f} min")
                print(f"   Total vítimas: {stats['total_victims']}")
                print(f"   Total danos: R$ {stats['total_damage']:,.2f}")
    else:
        count = generate_sample_data(args.n_records, args.clear)
        
        if args.export:
            export_to_json()
    
    print("\nOperação concluída!")