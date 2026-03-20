import requests
import os

BASE = "https://premier.72-60-245-2.sslip.io"
DATA_DIR = r"c:\Users\Arnold's\Documents\Repositorios Machine Learning\Proyecto 2 Machine Learning\data\raw"

endpoints = {
    "players.csv": f"{BASE}/export/players",
    "matches.csv": f"{BASE}/export/matches",
    "events.csv": f"{BASE}/export/events",
    "player_history.csv": f"{BASE}/export/player_history"
}

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

for filename, url in endpoints.items():
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        path = os.path.join(DATA_DIR, filename)
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {path}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

print("All downloads finished.")
