import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from pymongo import MongoClient

def load_data_from_mongodb():
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
    client = MongoClient(MONGO_URI)
    db = client["chama-sos-bombeiros"]
    collection = db["chama-sos-bombeiros"]
    
    count = collection.count_documents({})
    if count == 0:
        print("Nenhum dado encontrado no MongoDB.")
        print("Execute data_generator.py primeiro ou use o endpoint /api/init-database")
        return None
    
    print(f"Carregando {count} registros do MongoDB...")
    
    cursor = collection.find({}, {'_id': 0})
    df = pd.DataFrame(list(cursor))
    
    print(f"Dados carregados: {df.shape}")
    return df

def prepare_data(df):
    print("\n Preparando dados para treinamento...")
    
    target_variable = "Tempo de Resposta (minutos)"
    
    required_columns = ['Grupo de Ocorrência', 'Tipo de Ocorrência', 
                       'Prioridade', 'Número de Vítimas', 
                       'Danos Estimados (R$)', target_variable]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Colunas faltando: {missing_columns}")
        return None, None, None, None
    
    df = df.dropna(subset=[target_variable])
    
    cat_cols = ['Grupo de Ocorrência', 'Tipo de Ocorrência', 'Prioridade']
    encoders = {}
    
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"   ✓ {col} codificada ({len(le.classes_)} categorias)")
    
    feature_cols = ['Grupo de Ocorrência', 'Tipo de Ocorrência', 
                    'Prioridade', 'Número de Vítimas', 'Danos Estimados (R$)']
    
    X = df[feature_cols]
    y = df[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"   ✓ Features: {list(X.columns)}")
    print(f"   ✓ Target: {target_variable}")
    print(f"   ✓ Treino: {X_train.shape}, Teste: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, encoders, feature_cols

def train_xgboost(X_train, X_test, y_train, y_test):
    print("\nTreinando modelo XGBoost...")
    
    params = {
        'n_estimators': 150,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"Modelo treinado com sucesso!")
    print("\nMétricas de Avaliação:")
    print(f"   MAE:  {mae:.2f} minutos")
    print(f"   MSE:  {mse:.2f}")
    print(f"   RMSE: {rmse:.2f} minutos")
    print(f"   R²:   {r2:.3f}")
    print(f"   R² Treino: {r2_train:.3f}")
    print(f"   MAPE: {mape:.1f}%")
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'r2_train': float(r2_train),
        'mape': float(mape),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test)
    }
    
    return model, metrics

def analyze_feature_importance(model, feature_names):
    print("\nAnálise de Importância das Features:")
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    print("   Rank  Feature                     Importance")
    print("   ----  -------------------------   ----------")
    for i, idx in enumerate(indices):
        feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
        print(f"   {i+1:2d}.   {feature_name:25s}   {importance[idx]:.4f}")
    
    return importance

def save_model(model, encoders, metrics, feature_names):
    print("\n Salvando modelo e arquivos...")
    
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/xgboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✓ Modelo salvo em: {model_path}")
    
    encoders_path = 'models/encoders.pkl'
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"   ✓ Encoders salvos em: {encoders_path}")
    
    metrics_path = 'models/model_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✓ Métricas salvas em: {metrics_path}")
    
    features_info = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'target_variable': 'Tempo de Resposta (minutos)'
    }
    
    features_path = 'models/features_info.json'
    with open(features_path, 'w') as f:
        json.dump(features_info, f, indent=2)
    print(f"   ✓ Informações das features salvas em: {features_path}")

def main():
    
    print("=" * 60)
    print("TREINAMENTO DO MODELO XGBOOST - CHAMA SOS")
    print("=" * 60)
    
    df = load_data_from_mongodb()
    if df is None:
        return
    
    result = prepare_data(df)
    if result[0] is None:
        return
    
    X_train, X_test, y_train, y_test, encoders, feature_names = result
    
    if len(X_train) < 50:
        print("Dados insuficientes para treinamento. Mínimo: 50 amostras")
        return
    
    model, metrics = train_xgboost(X_train, X_test, y_train, y_test)
    
    analyze_feature_importance(model, feature_names)
    
    save_model(model, encoders, metrics, feature_names)
    
    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print("\nPróximos passos:")
    print("1. Execute: python app.py")
    print("2. Acesse: http://localhost:5000")
    print("3. Use o endpoint /api/predict para fazer predições")

if __name__ == '__main__':
    main()