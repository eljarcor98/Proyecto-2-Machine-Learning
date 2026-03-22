import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Premier League Analytics 2025-26", layout="wide", page_icon="⚽")

BASE_URL = "https://premier.72-60-245-2.sslip.io"

# --- Funciones de Datos ---

@st.cache_data(ttl=3600)
def get_all_matches():
    res = requests.get(f"{BASE_URL}/matches?limit=500")
    if res.status_code == 200:
        return res.json()["matches"]
    return []

def fetch_match_events(match_info):
    match_id = match_info["id"]
    try:
        res = requests.get(f"{BASE_URL}/matches/{match_id}/events", timeout=15)
        if res.status_code == 200:
            events = res.json().get("events", [])
            # Agregamos info del partido a cada evento
            for e in events:
                e["home_team"] = match_info["home_team"]
                e["away_team"] = match_info["away_team"]
            return events
    except:
        pass
    return []

@st.cache_data(ttl=3600)
def get_all_events(matches):
    all_events = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(fetch_match_events, matches))
    for res in results:
        all_events.extend(res)
    return pd.DataFrame(all_events)

# --- Normalización ---

def normalize_events(df):
    if df.empty: return df
    
    # Creamos copias para no afectar el cache
    df = df.copy()
    
    # Determinamos dirección de ataque por (partido, equipo, periodo)
    # Calculamos el promedio de X para cada grupo
    stats = df.groupby(['match_id', 'team_name', 'period'])['x'].mean().reset_index()
    stats['should_flip'] = stats['x'] < 50
    
    df = df.merge(stats[['match_id', 'team_name', 'period', 'should_flip']], on=['match_id', 'team_name', 'period'], how='left')
    
    df['x_norm'] = np.where(df['should_flip'], 100 - df['x'], df['x'])
    df['y_norm'] = np.where(df['should_flip'], 100 - df['y'], df['y'])
    
    return df

# --- Dibujo del Campo (Plotly) ---

def draw_plotly_pitch():
    fig = go.Figure()

    # Líneas del campo
    shapes = [
        # Borde exterior
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color="white", width=2)),
        # Línea central
        dict(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color="white", width=2)),
        # Áreas grandes
        dict(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color="white", width=2)),
        dict(type="rect", x0=83.5, y0=21.1, x1=100, y1=78.9, line=dict(color="white", width=2)),
        # Áreas pequeñas
        dict(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color="white", width=2)),
        dict(type="rect", x0=94.5, y0=36.8, x1=100, y1=63.2, line=dict(color="white", width=2)),
    ]

    fig.update_layout(
        shapes=shapes,
        template="plotly_dark",
        width=800,
        height=600,
        xaxis=dict(range=[-5, 105], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-5, 105], showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="#1e1e1e"
    )
    
    # Círculo central
    fig.add_shape(type="circle", x0=40.85, y0=40.85, x1=59.15, y1=59.15, line_color="white")
    
    return fig

# --- App Streamlit ---

st.title("⚽ Premier League Interactive Dashboard 2025-26")
st.markdown("Analítica avanzada de disparos y faltas capturada directamente desde la API.")

with st.spinner("Cargando datos históricos... esto dura ~10 segundos la primera vez."):
    matches = get_all_matches()
    df_raw = get_all_events(matches)
    df = normalize_events(df_raw)

if df.empty:
    st.error("No se pudieron cargar los datos. Verifica la conexión a la API.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Filtros de Análisis")
event_type = st.sidebar.selectbox("Tipo de Evento", ["Disparos", "Faltas"])
team_filter = st.sidebar.multiselect("Filtrar por Equipo", sorted(df['team_name'].unique()))

# --- Filtrado de Data ---
if event_type == "Disparos":
    plot_df = df[df['is_shot'] == True]
    leaderboard_col = "is_goal"
    leaderboard_title = "Leaderboard: Fallas (Tiros que no fueron gol)"
    # Para disparos, "fallas" son tiros sin gol
    leaderboard_df = plot_df[plot_df['is_goal'] == False].groupby(['player_name', 'team_name']).size().reset_index(name='Fallas')
    leaderboard_df = leaderboard_df.sort_values('Fallas', ascending=False).head(10)
else:
    plot_df = df[df['event_type'] == 'Foul']
    leaderboard_title = "Leaderboard: Faltas Cometidas"
    leaderboard_df = plot_df.groupby(['player_name', 'team_name']).size().reset_index(name='Faltas')
    leaderboard_df = leaderboard_df.sort_values('Faltas', ascending=False).head(10)

if team_filter:
    plot_df = plot_df[plot_df['team_name'].isin(team_filter)]

# --- Visualización ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Mapa Interactivo: {event_type}")
    fig = draw_plotly_pitch()
    
    if not plot_df.empty:
        if event_type == "Disparos":
            # Goles vs No Goles
            goals = plot_df[plot_df['is_goal'] == True]
            misses = plot_df[plot_df['is_goal'] == False]
            
            fig.add_trace(go.Scatter(
                x=misses['x_norm'], y=misses['y_norm'], mode='markers',
                name='Falla', marker=dict(color='white', size=6, opacity=0.3),
                text=misses['player_name'] + " (" + misses['team_name'] + ")",
                hovertemplate="<b>%{text}</b><br>Minuto: %{customdata}<extra></extra>",
                customdata=misses['minute']
            ))
            fig.add_trace(go.Scatter(
                x=goals['x_norm'], y=goals['y_norm'], mode='markers',
                name='GOL', marker=dict(color='#2ecc71', size=12, symbol='star', line=dict(width=1, color='white')),
                text=goals['player_name'] + " (" + goals['team_name'] + ")",
                hovertemplate="<b>%{text}</b><br>Minuto: %{customdata}<extra></extra>",
                customdata=goals['minute']
            ))
        else:
            # Faltas
            fig.add_trace(go.Scatter(
                x=plot_df['x_norm'], y=plot_df['y_norm'], mode='markers',
                name='Falta', marker=dict(color='#e74c3c', size=8, opacity=0.6, symbol='x'),
                text=plot_df['player_name'] + " (" + plot_df['team_name'] + ")",
                hovertemplate="<b>%{text}</b><br>Minuto: %{customdata}<extra></extra>",
                customdata=plot_df['minute']
            ))
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Normalización: Todos los ataques se muestran hacia la portería derecha (X=100).")

with col2:
    st.subheader(leaderboard_title)
    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)
    
    st.subheader("Métricas Rápidas")
    m1, m2 = st.columns(2)
    m1.metric("Total Eventos", len(plot_df))
    if event_type == "Disparos":
        acc = (plot_df['is_goal'].sum() / len(plot_df)) * 100 if len(plot_df) > 0 else 0
        m2.metric("Efectividad", f"{acc:.1f}%")
    else:
        m2.metric("Promedio x Partido", f"{len(plot_df)/len(matches):.1f}")

st.divider()
st.info("💡 Consejo: Usa el lazo o zoom en el mapa para analizar zonas específicas del campo.")
