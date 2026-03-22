import pandas as pd
import numpy as np
import json
import os

def build_features():
    print("Iniciando extracción de features...")
    
    # Cargar datos de eventos
    raw_path = 'data/raw/events.csv'
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} no encontrado.")
        return

    # Leer solo eventos de tipo tiro para ser eficientes
    # Usamos un iterador para manejar el archivo grande si es necesario, 
    # pero aquí filtramos por la columna 'is_shot'
    df = pd.read_csv(raw_path)
    df_shots = df[df['is_shot'] == True].copy()
    print(f"Total tiros encontrados: {len(df_shots)}")

    # 1. Features Geométricas
    df_shots['distance'] = np.sqrt((100 - df_shots['x'])**2 + (50 - df_shots['y'])**2)
    df_shots['angle'] = np.arctan2(50 - df_shots['y'], 100 - df_shots['x'])

    # 2. Parseo de Qualifiers
    def parse_qualifiers(q_str):
        if not isinstance(q_str, str) or q_str == '[]':
            return {}
        try:
            # Manejar comillas simples en el string para que sea JSON válido
            q_list = json.loads(q_str.replace("'", '"'))
            return {q.get('type', {}).get('displayName', q.get('type', '')): q.get('value', True) for q in q_list}
        except:
            return {}

    # Extraer qualifiers comunes como columnas booleanas
    qs = df_shots['qualifiers'].apply(parse_qualifiers)
    
    df_shots['is_header'] = qs.apply(lambda x: 'Head' in x).astype(int)
    df_shots['is_big_chance'] = qs.apply(lambda x: 'BigChance' in x).astype(int)
    df_shots['is_penalty'] = qs.apply(lambda x: 'Penalty' in x).astype(int)
    df_shots['is_right_foot'] = qs.apply(lambda x: 'RightFoot' in x).astype(int)
    df_shots['is_left_foot'] = qs.apply(lambda x: 'LeftFoot' in x).astype(int)
    df_shots['is_counter'] = qs.apply(lambda x: 'FastBreak' in x).astype(int)
    df_shots['is_from_corner'] = qs.apply(lambda x: 'FromCorner' in x).astype(int)
    
    # Extraer zona (si existe)
    df_shots['shot_zone'] = qs.apply(lambda x: x.get('Zone', 'Unknown'))

    # Guardar resultado
    output_dir = 'data/processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'shots_features.csv')
    # Columnas finales para el modelo xG
    final_cols = [
        'id', 'match_id', 'player_id', 'team_name', 'x', 'y', 'distance', 'angle',
        'is_header', 'is_right_foot', 'is_left_foot', 'is_big_chance', 
        'is_penalty', 'is_counter', 'is_from_corner', 'shot_zone', 'is_goal'
    ]
    df_shots[final_cols].to_csv(output_path, index=False)
    print(f"Dataset procesado guardado en: {output_path}")

if __name__ == "__main__":
    build_features()
