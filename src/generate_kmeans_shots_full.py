import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import os

BASE_URL = "https://premier.72-60-245-2.sslip.io"
FIGURE_DIR = r"C:\Users\Arnold's\Documents\Repositorios Machine Learning\taller 2\reports\figures"

def fetch_normalized_shots(match_info):
    match_id = match_info["id"]
    try:
        res = requests.get(f"{BASE_URL}/matches/{match_id}/events", timeout=15)
        if res.status_code == 200:
            events = res.json().get("events", [])
            shots = [e for e in events if e.get("is_shot")]
            if not shots: return []
            
            df = pd.DataFrame(shots)
            for period in ["FirstHalf", "SecondHalf"]:
                p_mask = df["period"] == period
                if p_mask.any():
                    avg_x = df.loc[p_mask, "x"].mean()
                    if avg_x < 50:
                        df.loc[p_mask, "x"] = 100 - df.loc[p_mask, "x"]
                        df.loc[p_mask, "y"] = 100 - df.loc[p_mask, "y"]
            
            # Feature engineering
            df['distance'] = np.sqrt((100 - df['x'])**2 + (50 - df['y'])**2)
            df['angle'] = np.arctan2(abs(50 - df['y']), (100 - df['x']))
            df['period_num'] = df['period'].map({'FirstHalf': 0, 'SecondHalf': 1})
            
            return df[['x', 'y', 'minute', 'distance', 'angle', 'period_num', 'is_goal']].to_dict('records')
    except:
        pass
    return []

def main():
    print("--- Clustering Multivariado con K-Means (k=4) ---")
    
    matches_res = requests.get(f"{BASE_URL}/matches?limit=500")
    matches = matches_res.json()["matches"]
    
    print(f"Descargando datos enriquecidos de {len(matches)} partidos...")
    all_shots = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(fetch_normalized_shots, matches))
    
    for shots in results:
        all_shots.extend(shots)
    
    df = pd.DataFrame(all_shots)
    print(f"Total disparos para análisis: {len(df)}")
    
    # 2. Preprocesamiento
    features = ['x', 'y', 'minute', 'distance', 'angle', 'period_num']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. K-Means
    print("Calculando clusters en 6 dimensiones...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 4. Análisis de Resultados
    print("\nResumen de Clusters (Perfil de Disparo):")
    summary = df.groupby('cluster').agg({
        'minute': 'mean',
        'distance': 'mean',
        'is_goal': 'mean',
        'x': 'count'
    }).rename(columns={'is_goal': 'efectividad', 'x': 'cantidad'})
    summary['efectividad'] *= 100
    print(summary)

    # 5. Visualización (Pairplot de las características más importantes)
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    # Seleccionamos una muestra para el pairplot si es muy grande
    if len(df) > 1000:
        plot_df = df.sample(1000, random_state=42)
    else:
        plot_df = df
        
    g = sns.pairplot(plot_df, vars=['x', 'minute', 'distance'], hue='cluster', palette='viridis', diag_kind='kde')
    g.fig.suptitle("Clustering Multivariado de Situaciones de Tiro (k=4)", y=1.02, fontsize=16)
    
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
        
    output_path = os.path.join(FIGURE_DIR, "kmeans_multi_feature.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n¡Reporte multivariado guardado en: {output_path}")

if __name__ == "__main__":
    main()
