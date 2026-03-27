import requests
import pandas as pd

import os

# Configuración de la API
BASE_URL = "https://premier.72-60-245-2.sslip.io"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Directorio creado: {DATA_DIR}")

def load_matches(save=True):
    print(f"Cargando partidos desde {BASE_URL}...")
    try:
        response = requests.get(f"{BASE_URL}/matches?limit=500")
        response.raise_for_status()
        
        data = response.json()
        if "matches" in data:
            matches = pd.DataFrame(data["matches"])
            print(f"Partidos cargados: {len(matches)}")
            
            if save:
                ensure_data_dir()
                path = os.path.join(DATA_DIR, "matches.csv")
                matches.to_csv(path, index=False)
                print(f"[OK] Datos guardados en: {path}")
            
            return matches
        else:
            print("[ERROR] No se encontro la clave 'matches' en la respuesta.")
            return None
            
    except Exception as e:
        print(f"[ERROR] Error al cargar los datos: {e}")
        return None

def download_all_data():
    """Descarga partidos, jugadores y standings localmente."""
    print("\n--- Iniciando descarga completa de datos ---")
    
    # 1. Partidos
    load_matches(save=True)
    
    # 2. Jugadores
    print("\nDescargando jugadores...")
    players_res = requests.get(f"{BASE_URL}/players?limit=500")
    if players_res.status_code == 200:
        players_data = players_res.json().get("players", [])
        if players_data:
            players = pd.DataFrame(players_data)
            path = os.path.join(DATA_DIR, "players.csv")
            players.to_csv(path, index=False)
            print(f"[OK] Jugadores guardados en: {path}")
        else:
            print("[ERROR] No se encontraron datos de jugadores en la respuesta.")
    else:
        print(f"[ERROR] Fallo en la descarga de jugadores. Status: {players_res.status_code}")
        
    # 3. Standings
    print("\nDescargando standings...")
    standings_res = requests.get(f"{BASE_URL}/standings")
    if standings_res.status_code == 200:
        # Los standings vienen anidados
        standings_data = standings_res.json()
        if isinstance(standings_data, list) and len(standings_data) > 0:
            # Si es la estructura anidada que vimos antes
            if "standings" in standings_data[0]:
                raw_standings = [item["standings"] for item in standings_data]
                df_standings = pd.DataFrame(raw_standings)
            else:
                df_standings = pd.DataFrame(standings_data)
        else:
            df_standings = pd.DataFrame(standings_data)
            
        path = os.path.join(DATA_DIR, "standings.csv")
        df_standings.to_csv(path, index=False)
        print(f"[OK] Standings guardados en: {path}")

if __name__ == "__main__":
    download_all_data()
