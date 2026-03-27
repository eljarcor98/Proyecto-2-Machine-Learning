import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os

# Configuracion de rutas
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "matches.csv")

def prepare_data():
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra el archivo {DATA_PATH}")
        return None
    
    df = pd.read_csv(DATA_PATH)
    
    # 1. Feature Engineering Basico
    # hs: home shots, as: away shots, hst/ast: shots on target
    # Incluimos cuotas como predictores de la probabilidad "del mercado"
    features = ['hs', 'as_', 'hst', 'ast', 'hf', 'af', 'hc', 'ac', 'b365h', 'b365d', 'b365a']
    
    # Seleccion de caracteristicas y eliminacion de nulos
    X = df[features].dropna()
    y = df.loc[X.index, 'ftr']
    
    return X, y

def train():
    print("--- Iniciando Entrenamiento del Modelo ---")
    prepared = prepare_data()
    if prepared is None: return
    
    X, y = prepared
    
    # Dividir datos (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Partidos para entrenamiento: {X_train.shape[0]}")
    print(f"Partidos para test: {X_test.shape[0]}")
    
    # Entrenar modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Evaluacion
    print("\nMetricas de Evaluacion:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nReporte Detallado:")
    print(classification_report(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    train()
