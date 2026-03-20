import pandas as pd
import requests
import os
import time

BASE = "https://premier.72-60-245-2.sslip.io"
MATCHES_PATH = r"c:\Users\Arnold's\Documents\Repositorios Machine Learning\Proyecto 2 Machine Learning\data\raw\matches.csv"
OUTPUT_PATH = r"c:\Users\Arnold's\Documents\Repositorios Machine Learning\Proyecto 2 Machine Learning\data\raw\events.csv"

def download_events():
    df_matches = pd.read_csv(MATCHES_PATH)
    match_ids = df_matches['id'].unique()
    
    all_events = []
    total = len(match_ids)
    
    print(f"Starting match-by-match download for {total} matches...")
    
    for i, mid in enumerate(match_ids):
        print(f"[{i+1}/{total}] Downloading events for match {mid}...")
        url = f"{BASE}/matches/{mid}/events?format=csv" # Assuming it supports ?format=csv
        
        try:
            # First try CSV
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and 'id' in response.text:
                # Append to file or list
                with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                    if i == 0:
                        f.write(response.text)
                    else:
                        # Skip header for subsequent matches
                        lines = response.text.splitlines()
                        if len(lines) > 1:
                            f.write("\n" + "\n".join(lines[1:]))
            else:
                # If CSV not supported, try JSON
                url_json = f"{BASE}/matches/{mid}/events"
                resp_json = requests.get(url_json, timeout=30)
                if resp_json.status_code == 200:
                    data = resp_json.json()
                    if 'events' in data:
                        events_df = pd.DataFrame(data['events'])
                        events_df.to_csv(OUTPUT_PATH, mode='a', header=(i==0), index=False, encoding='utf-8')
                else:
                    print(f"Error for match {mid}: {resp_json.status_code}")
            
            # Rate limiting or just to be safe
            # time.sleep(0.1)
            
        except Exception as e:
            print(f"Exception for match {mid}: {e}")
            
    print(f"Finished. Combined events saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    download_events()
