import pandas as pd
import os

# Configuración de rutas
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "matches.csv")

def rapid_exploration():
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra el archivo {DATA_PATH}")
        return
    
    matches = pd.read_csv(DATA_PATH)
    
    # Columnas disponibles
    print("Columnas:", matches.columns.tolist())
    
    # Resultado más común (ftr = Full Time Result: H/D/A)
    print("\nDistribución de resultados:")
    print(matches["ftr"].value_counts())
    
    # Promedio de goles (fthg = Full Time Home Goals, ftag = Away Goals)
    print(f"\nGoles por partido: {(matches['fthg'] + matches['ftag']).mean():.2f}")

if __name__ == "__main__":
    rapid_exploration()
