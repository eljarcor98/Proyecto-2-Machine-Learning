import requests
import pandas as pd
import json

# Configuración de la API
BASE_URL = "https://premier.72-60-245-2.sslip.io"

def test_connection():
    print(f"Probando conexion a {BASE_URL}...")
    try:
        # 1. Probar endpoint raiz /
        response = requests.get(BASE_URL)
        response.raise_for_status()
        info = response.json()
        print(f"[OK] Conexion exitosa. Version API: {info.get('version')}, Temporada: {info.get('season')}")
        
        # 2. Obtener tabla de posiciones (Standings)
        print("\nConsultando tabla de posiciones...")
        standings_res = requests.get(f"{BASE_URL}/standings")
        standings_res.raise_for_status()
        df_standings = pd.DataFrame(standings_res.json())
        print("[OK] Standings recibidos.")
        print("Columnas disponibles:", df_standings.columns.tolist())
        # Ajustar columnas segun lo que devuelva la API real
        if 'position' in df_standings.columns:
             cols = [c for c in ['position', 'team_name', 'played', 'points'] if c in df_standings.columns]
             print(df_standings[cols].head(5))
        else:
             print(df_standings.head(5))
        
        # 3. Obtener muestra de jugadores
        print("\nConsultando muestra de jugadores...")
        players_res = requests.get(f"{BASE_URL}/players?limit=5")
        players_res.raise_for_status()
        players_data = players_res.json().get('players', [])
        df_players = pd.DataFrame(players_data)
        print("[OK] Datos de jugadores recibidos.")
        if not df_players.empty:
            print(df_players[['web_name', 'team_name', 'now_cost', 'total_points']].head())

        return True
    except Exception as e:
        print(f"[ERROR] Al conectar con la API: {e}")
        return False

if __name__ == "__main__":
    test_connection()
