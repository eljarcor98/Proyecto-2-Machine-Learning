import requests
import pandas as pd

BASE_URL = "https://premier.72-60-245-2.sslip.io"

def find_example():
    # Buscaremos en los primeros 10 partidos para ser rápidos
    for match_id in range(1, 21):
        try:
            res = requests.get(f"{BASE_URL}/matches/{match_id}/events")
            if res.status_code == 200:
                events = res.json()["events"]
                for e in events:
                    if e.get("is_goal") and e.get("x") < 20: 
                        print(f"¡Encontrado! Partido ID: {match_id}")
                        print(f"Equipo: {e['team_name']}, Minuto: {e['minute']}, X: {e['x']}, Y: {e['y']}")
                        return
        except:
            continue

if __name__ == "__main__":
    find_example()
