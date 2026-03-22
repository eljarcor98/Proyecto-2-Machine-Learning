import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle
from concurrent.futures import ThreadPoolExecutor
import os

BASE_URL = "https://premier.72-60-245-2.sslip.io"
FIGURE_DIR = r"C:\Users\Arnold's\Documents\Repositorios Machine Learning\taller 2\reports\figures"

def draw_pitch_refined(ax, theme='dark'):
    pitch_color = '#1e1e1e' if theme == 'dark' else 'white'
    line_color = '#cccccc' if theme == 'dark' else 'black'
    
    ax.set_facecolor(pitch_color)
    
    # Campo
    ax.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color=line_color, linewidth=2)
    # Línea central
    ax.plot([50, 50], [0, 100], color=line_color, linewidth=2)
    # Círculo central
    centre_circle = plt.Circle((50, 50), 9.15, color=line_color, fill=False, linewidth=2)
    ax.add_patch(centre_circle)
    # Áreas grandes
    ax.add_patch(Rectangle((0, 21.1), 16.5, 57.8, fill=False, color=line_color, linewidth=2))
    ax.add_patch(Rectangle((83.5, 21.1), 16.5, 57.8, fill=False, color=line_color, linewidth=2))
    # Áreas pequeñas
    ax.add_patch(Rectangle((0, 36.85), 5.5, 26.3, fill=False, color=line_color, linewidth=2))
    ax.add_patch(Rectangle((94.5, 36.85), 5.5, 26.3, fill=False, color=line_color, linewidth=2))
    # Porterías
    ax.plot([-2, 0], [45.2, 45.2], color=line_color, linewidth=3)
    ax.plot([-2, 0], [54.8, 54.8], color=line_color, linewidth=3)
    ax.plot([100, 102], [45.2, 45.2], color=line_color, linewidth=3)
    ax.plot([100, 102], [54.8, 54.8], color=line_color, linewidth=3)
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')
    ax.axis('off')

def fetch_away_shots(match_info):
    match_id = match_info["id"]
    away_team = match_info["away_team"]
    try:
        res = requests.get(f"{BASE_URL}/matches/{match_id}/events", timeout=10)
        if res.status_code == 200:
            events = res.json().get("events", [])
            
            # Filtrar disparos del equipo visitante (usando strip por seguridad)
            away_team_clean = str(away_team).strip().lower()
            away_shots = [e for e in events if e.get("is_shot") and str(e.get("team_name")).strip().lower() == away_team_clean]
            
            for period in ["FirstHalf", "SecondHalf"]:
                period_shots = [s for s in away_shots if s.get("period") == period]
                if not period_shots: continue
                
                avg_x = sum(s["x"] for s in period_shots) / len(period_shots)
                # Si el promedio de X es bajo (<50), invertimos las coordenadas
                should_flip = avg_x < 50
                
                for s in period_shots:
                    if should_flip:
                        s["x_norm"] = 100 - s["x"]
                        s["y_norm"] = 100 - s["y"]
                    else:
                        s["x_norm"] = s["x"]
                        s["y_norm"] = s["y"]
                    normalized_shots.append(s)
            
            return normalized_shots
    except Exception as e:
        print(f"Error fetching match {match_id}: {e}")
    return []

def main():
    print("--- Generando Mapa Global de Disparos Normalizado (VISITANTES) ---")
    
    # 1. Obtener lista de partidos
    print("Obteniendo lista de partidos...")
    matches_res = requests.get(f"{BASE_URL}/matches?limit=500")
    if matches_res.status_code != 200:
        print("Error: No se pudo conectar a la API.")
        return
    
    matches_data = matches_res.json()["matches"]
    print(f"Partidos encontrados: {len(matches_data)}")
    
    # 2. Descargar eventos en paralelo
    print("Descargando y normalizando eventos (291 partidos)...")
    all_shots = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_away_shots, matches_data))
    
    for shots in results:
        all_shots.extend(shots)
    
    df_shots = pd.DataFrame(all_shots)
    if df_shots.empty:
        print("Error: No se acumularon disparos en all_shots.")
        return
    
    print(f"Total de disparos visitantes normalizados: {len(df_shots)}")
    
    print(f"Total de disparos visitantes normalizados: {len(df_shots)}")
    
    # 3. Graficar
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 11))
    draw_pitch_refined(ax, theme='dark')
    
    # Disparos fallados (puntos pequeños, baja opacidad)
    failed = df_shots[df_shots["is_goal"] == False]
    ax.scatter(failed["x_norm"], failed["y_norm"], color="white", s=8, alpha=0.15, label="Tiro Fallado", zorder=2)
    
    # Goles (Estrellas vibrantes)
    goals = df_shots[df_shots["is_goal"] == True]
    ax.scatter(goals["x_norm"], goals["y_norm"], color="#2ecc71", s=100, alpha=0.9, marker="*", edgecolors="white", label=f"GOL ({len(goals)})", zorder=4)
    
    plt.title(f"MAPA GLOBAL DE DISPAROS NORMALIZADO (VISITANTES) - PREMIER LEAGUE 2025-26\n({len(matches_data)} partidos, {len(df_shots)} disparos)", fontsize=18, pad=20, color="white")
    plt.legend(loc="lower right", facecolor="#1e1e1e", edgecolor="white", fontsize=12)
    
    # Nota sobre la normalización
    ax.text(50, -5, "Datos normalizados: Todos los ataques representados hacia la portería derecha", 
            ha='center', fontsize=10, color='#aaaaaa', style='italic')
    
    # Guardar
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
    
    output_path = os.path.join(FIGURE_DIR, "shot_map_global_normalized.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"¡Éxito! El mapa global normalizado ha sido guardado en: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
