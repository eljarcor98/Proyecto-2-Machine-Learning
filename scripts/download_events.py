import time
from pathlib import Path

import requests

BASE = "https://premier.72-60-245-2.sslip.io"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
FILE_PATH = DATA_DIR / "events.csv"
URL = f"{BASE}/export/events"

def download_large_file(url, path):
    print(f"Downloading {url} to {path}...")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                        if chunk:
                            f.write(chunk)
            print(f"Successfully downloaded to {path}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    return False

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    success = download_large_file(URL, FILE_PATH)
    if not success:
        print("Final attempt failed.")
        exit(1)
