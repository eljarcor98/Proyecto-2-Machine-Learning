import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Rectangle, Circle
from concurrent.futures import ThreadPoolExecutor
import os

BASE_URL = "https://premier.72-60-245-2.sslip.io"
FIGURE_DIR = r"C:\Users\Arnold's\Documents\Repositorios Machine Learning\taller 2\reports\figures"

def draw_pitch_dark(ax):
    pitch_color = '#1e1e1e'
    line_color = '#cccccc'
    ax.set_facecolor(pitch_color)
    ax.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color=line_color, linewidth=2)
    ax.plot([50, 50], [0, 100], color=line_color, linewidth=2)
    ax.add_patch(Rectangle((0, 21.1), 16.5, 57.8, fill=False, color=line_color, linewidth=1.5))
    ax.add_patch(Rectangle((83.5, 21.1), 16.5, 57.8, fill=False, color=line_color, linewidth=1.5))
    ax.add_patch(Rectangle((0, 36.8), 5.5, 26.4, fill=False, color=line_color, linewidth=1.2))
    ax.add_patch(Rectangle((94.5, 36.8), 5.5, 26.4, fill=False, color=line_color, linewidth=1.2))
    ax.add_patch(Circle((50, 50), 9.15, fill=False, color=line_color, linewidth=1.5))
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')
    ax.axis('off')

def fetch_normalized_shots(match_info):
    match_id = match_info["id"]
    try:
        res = requests.get(f"{BASE_URL}/matches/{match_id}/events", timeout=15)
        if res.status_code == 200:
            events = res.json().get("events", [])
            shots = [e for e in events if e.get("is_shot")]
            if not shots: return []
            
            # Normalización rápida (atacando a la derecha X=100)
            df = pd.DataFrame(shots)
            for period in ["FirstHalf", "SecondHalf"]:
                p_mask = df["period"] == period
                if p_mask.any():
                    avg_x = df.loc[p_mask, "x"].mean()
                    if avg_x < 50:
                        df.loc[p_mask, "x"] = 100 - df.loc[p_mask, "x"]
                        df.loc[p_mask, "y"] = 100 - df.loc[p_mask, "y"]
            return df.to_dict('records')
    except:
        pass
    return []

def main():
    print("--- Analizando Zonas de Disparo con K-Means (k=4) ---")
    
    # 1. Obtener datos de toda la temporada
    matches_res = requests.get(f"{BASE_URL}/matches?limit=500")
    matches = matches_res.json()["matches"]
    
    print(f"Descargando disparos de {len(matches)} partidos...")
    all_shots = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(fetch_normalized_shots, matches))
    
    for shots in results:
        all_shots.extend(shots)
    
    df = pd.DataFrame(all_shots)
    print(f"Total disparos normalizados: {len(df)}")
    
    # 2. K-Means
    print("Aplicando K-Means...")
    coords = df[['x', 'y']]
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(coords)
    centroids = kmeans.cluster_centers_

    # 3. Visualización
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 10))
    draw_pitch_dark(ax)
    
    # Colores para los clusters
    colors = ['#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
    cluster_names = ["Zona 0", "Zona 1", "Zona 2", "Zona 3"] # Se asignarán nombres lógicos después

    for i in range(4):
        c_df = df[df['cluster'] == i]
        ax.scatter(c_df['x'], c_df['y'], color=colors[i], alpha=0.3, s=20, label=f'Cluster {i}')
        # Dibujar centroide
        ax.scatter(centroids[i][0], centroids[i][1], color='white', s=200, marker='*', edgecolors='black')
        # Etiqueta en el centroide
        ax.text(centroids[i][0], centroids[i][1]+3, f"Zona {i}", color='white', weight='bold', ha='center', fontsize=12)

    plt.title("CLUSTERING DE ZONAS DE DISPARO (K-Means, k=4)\nVisualizando patrones espaciales en toda la temporada", fontsize=18, pad=20)
    
    # Leyenda informativa
    import matplotlib.lines as mlines
    star = mlines.Line2D([], [], color='white', marker='*', linestyle='None', markersize=10, label='Centroide (Zona Media)')
    plt.legend(handles=[star], loc='lower left')

    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
    
    output_path = os.path.join(FIGURE_DIR, "kmeans_shot_zones.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"¡Mapa de Clusters guardado en: {output_path}")
    
    # 4. Análisis de clusters
    print("\nResultados del Clustering:")
    for i in range(4):
        cluster_data = df[df['cluster'] == i]
        avg_x = cluster_data['x'].mean()
        goal_rate = (cluster_data['is_goal'].sum() / len(cluster_data)) * 100
        print(f"Zona {i}: Promedio X={avg_x:.1f}, Efectividad={goal_rate:.1f}%, Total={len(cluster_data)}")

if __name__ == "__main__":
    main()
