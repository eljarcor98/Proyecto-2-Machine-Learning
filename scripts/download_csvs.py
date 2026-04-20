from pathlib import Path

import requests

BASE = "https://premier.72-60-245-2.sslip.io"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

endpoints = {
    "players.csv": f"{BASE}/export/players",
    "matches.csv": f"{BASE}/export/matches",
    "events.csv": f"{BASE}/export/events",
    "player_history.csv": f"{BASE}/export/player_history"
}

DATA_DIR.mkdir(parents=True, exist_ok=True)

for filename, url in endpoints.items():
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        path = DATA_DIR / filename
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {path}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

print("All downloads finished.")
