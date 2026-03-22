import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from concurrent.futures import ThreadPoolExecutor
import os

BASE_URL = "https://premier.72-60-245-2.sslip.io"
FIGURE_DIR = r"C:\Users\Arnold's\Documents\Repositorios Machine Learning\taller 2\reports\figures"

def draw_pitch_refined(ax):
    pitch_color = '#1e1e1e' 
    line_color = '#cccccc'
    ax.set_facecolor(pitch_color)
    ax.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color=line_color, linewidth=2)
    ax.plot([50, 50], [0, 100], color=line_color, linewidth=2)
    ax.add_patch(Rectangle((0, 21.1), 16.5, 57.8, fill=False, color=line_color, linewidth=1.5))
    ax.add_patch(Rectangle((83.5, 21.1), 16.5, 57.8, fill=False, color=line_color, linewidth=1.5))
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')
    ax.axis('off')

def is_red_card(event):
    if event.get("event_type") != "Card":
        return False
    qualifiers = event.get("qualifiers", [])
    for q in qualifiers:
        if q.get("type", {}).get("displayName") in ["Red", "SecondYellow"]:
            return True
    return False

def fetch_red_cards(match_id):
    try:
        res = requests.get(f"{BASE_URL}/matches/{match_id}/events", timeout=15)
        if res.status_code == 200:
            events = res.json().get("events", [])
            return [e for e in events if is_red_card(e)]
    except:
        pass
    return []

def main():
    print("--- Generando Mapa Global de Tarjetas Rojas ---")
    
    # 1. Obtener partidos
    print("Obteniendo lista de partidos...")
    matches_res = requests.get(f"{BASE_URL}/matches?limit=500")
    match_ids = [m["id"] for m in matches_res.json()["matches"]]
    
    # 2. Descargar eventos
    print(f"Buscando Rojas en {len(match_ids)} partidos...")
    red_cards = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(fetch_red_cards, match_ids))
    
    for r in results:
        red_cards.extend(r)
    
    df = pd.DataFrame(red_cards)
    print(f"Total de tarjetas rojas encontradas: {len(df)}")
    
    if df.empty:
        print("No se encontraron tarjetas rojas en los eventos.")
        return

    # 3. Graficar
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch_refined(ax)
    
    # Graficar cada roja
    for _, row in df.iterrows():
        ax.scatter(row["x"], row["y"], color="#e74c3c", s=150, marker="s", edgecolors="white", label="Tarjeta Roja", zorder=5)
        # Etiqueta opcional con nombre (solo si hay pocos)
        if len(df) < 50:
            ax.text(row["x"]+1, row["y"]+1, f"{row['player_name']}\n({row['minute']}')", fontsize=8, color="white", alpha=0.8)

    plt.title(f"UBICACIÓN DE TARJETAS ROJAS - PREMIER LEAGUE 2025-26\n({len(df)} expulsiones analizadas)", fontsize=16, pad=20)
    
    # Guardar
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
    
    output_path = os.path.join(FIGURE_DIR, "red_cards_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"¡Mapa de Rojas guardado en: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
