import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from concurrent.futures import ThreadPoolExecutor
import os

BASE_URL = "https://premier.72-60-245-2.sslip.io"
FIGURE_DIR = r"C:\Users\Arnold's\Documents\Repositorios Machine Learning\taller 2\reports\figures"

def fetch_match_events(match_id):
    try:
        res = requests.get(f"{BASE_URL}/matches/{match_id}/events", timeout=15)
        if res.status_code == 200:
            return res.json().get("events", [])
    except Exception as e:
        print(f"Error en match {match_id}: {e}")
    return []

def main():
    print("--- Iniciando Generación de Leaderboards Estáticos ---")
    
    # 1. Obtener lista de partidos
    print("Obteniendo lista de partidos...")
    matches_res = requests.get(f"{BASE_URL}/matches?limit=500")
    if matches_res.status_code != 200:
        print("Error: No se pudo conectar a la API.")
        return
    
    match_ids = [m["id"] for m in matches_res.json()["matches"]]
    print(f"Partidos encontrados: {len(match_ids)}")
    
    # 2. Descargar todos los eventos en paralelo
    print("Descargando eventos de toda la temporada (esto puede tardar)...")
    all_events = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(fetch_match_events, match_ids))
    
    for events in results:
        all_events.extend(events)
    
    df = pd.DataFrame(all_events)
    print(f"Total de eventos procesados: {len(df)}")
    
    # Asegurar que el directorio de reportes existe
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)

    # 3. Reporte: Top 10 Fallas (Tiros que no son gol)
    print("Generando leaderboard de fallas...")
    shots = df[df["is_shot"] == True].copy()
    misses = shots[shots["is_goal"] == False]
    
    leaderboard_misses = misses.groupby(['player_name', 'team_name']).size().reset_index(name='Total_Fallas')
    leaderboard_misses = leaderboard_misses.sort_values('Total_Fallas', ascending=False).head(10)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=leaderboard_misses, x='Total_Fallas', y='player_name', hue='team_name', palette='Reds_r')
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.title('TOP 10 JUGADORES CON MÁS FALLAS (TIROS SIN GOL) - PREMIER LEAGUE 2025-26', fontsize=14, weight='bold')
    plt.xlabel('Número de Tiros Fallados')
    plt.ylabel('Jugador')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "leaderboard_shots.png"), dpi=300)
    print(f"¡Éxito! Leaderboard de tiros guardado.")

    # 4. Reporte: Top 10 Faltas Cometidas
    print("Generando leaderboard de faltas...")
    fouls = df[df["event_type"] == "Foul"].copy()
    
    leaderboard_fouls = fouls.groupby(['player_name', 'team_name']).size().reset_index(name='Total_Faltas')
    leaderboard_fouls = leaderboard_fouls.sort_values('Total_Faltas', ascending=False).head(10)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=leaderboard_fouls, x='Total_Faltas', y='player_name', hue='team_name', palette='YlOrBr_r')
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.title('TOP 10 JUGADORES CON MÁS FALTAS COMETIDAS - PREMIER LEAGUE 2025-26', fontsize=14, weight='bold')
    plt.xlabel('Número de Faltas')
    plt.ylabel('Jugador')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "leaderboard_fouls.png"), dpi=300)
    print(f"¡Éxito! Leaderboard de faltas guardado.")

    # 5. Reporte: Top 10 Tarjetas Rojas (Jugadores)
    print("Generando leaderboard de tarjetas rojas...")
    # Buscamos eventos de roja
    red_card_events = []
    for e in all_events:
        if e.get("event_type") == "Card":
            for q in e.get("qualifiers", []):
                if q.get("type", {}).get("displayName") in ["Red", "SecondYellow"]:
                    red_card_events.append(e)
                    break
    
    if red_card_events:
        df_reds = pd.DataFrame(red_card_events)
        leaderboard_reds = df_reds.groupby(['player_name', 'team_name']).size().reset_index(name='Total_Rojas')
        leaderboard_reds = leaderboard_reds.sort_values('Total_Rojas', ascending=False).head(10)
        
        plt.figure(figsize=(12, 7))
        sns.barplot(data=leaderboard_reds, x='Total_Rojas', y='player_name', hue='team_name', palette='OrRd_r')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.title('JUGADORES CON MÁS TARJETAS ROJAS - PREMIER LEAGUE 2025-26', fontsize=14, weight='bold')
        plt.xlabel('Número de Tarjetas Rojas')
        plt.ylabel('Jugador')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "leaderboard_reds.png"), dpi=300)
        print(f"¡Éxito! Leaderboard de rojas guardado.")
    else:
        print("No se encontraron tarjetas rojas para el leaderboard.")

    print(f"Reportes generados en: {FIGURE_DIR}")

if __name__ == "__main__":
    main()
