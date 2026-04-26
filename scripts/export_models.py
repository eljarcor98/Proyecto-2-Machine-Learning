import sys
import pickle
from pathlib import Path

# Configurar rutas para importar desde src
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from app_dash import load_matches, fit_goal_models, fit_xg_model, get_team_profiles

def export():
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("Iniciando exportación de modelos...")
    
    # 1. Cargar datos
    matches = load_matches()
    
    # 2. Entrenar modelos
    print("Entrenando Goal Models...")
    goal_models = fit_goal_models(matches)
    
    print("Entrenando Team Profiles...")
    tp_df_dict = get_team_profiles(matches)
    # Convertir DataFrames a diccionarios para quitar dependencias de pandas/pyarrow en este objeto
    team_profiles = {k: v.to_dict('index') for k, v in tp_df_dict.items()}
    
    print("Entrenando xG Model...")
    xg_model = fit_xg_model()
    
    # 3. Guardar en disco
    print(f"Guardando modelos en {models_dir}...")
    
    with open(models_dir / "goal_models.pkl", "wb") as f:
        pickle.dump(goal_models, f)
        
    with open(models_dir / "xg_model.pkl", "wb") as f:
        pickle.dump(xg_model, f)
        
    with open(models_dir / "team_profiles.pkl", "wb") as f:
        pickle.dump(team_profiles, f)
        
    print("✅ Exportación completada con éxito.")

if __name__ == "__main__":
    export()
