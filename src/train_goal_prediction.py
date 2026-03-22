import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de rutas
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "matches.csv")

def train_goal_model():
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra el archivo {DATA_PATH}")
        return
    
    matches = pd.read_csv(DATA_PATH)
    
    # --- Preparacion de datos solicitada por el usuario ---
    # Variable objetivo: goles del equipo local (fthg = Full Time Home Goals)
    y = matches["fthg"].astype(float)
    
    # Features numéricas del equipo local
    # hs=shots, hst=shots on target, hc=corners, hf=fouls, hy=yellow, hr=red
    features = ["hs", "hst", "hc", "hf", "hy", "hr"]
    X = matches[features].astype(float)
    
    # Verificar que no hay NaNs
    print(f"NaN en X: {X.isna().sum().sum()}")
    print(f"Shape: X={X.shape}, y={y.shape}")
    # -----------------------------------------------------

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo (Regresion Lineal como punto de partida)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Evaluacion
    print("\nMetricas del Modelo de Prediccion de Goles:")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

    # Coeficientes
    print("\nImportancia de las variables (Coeficientes):")
    for feat, coef in zip(features, model.coef_):
        print(f"{feat}: {coef:.3f}")

if __name__ == "__main__":
    train_goal_model()
