# Premier League Match Lab

Proyecto 2 de Machine Learning enfocado en análisis de partidos de Premier League, modelado de xG por tiro y predicción del resultado de un partido.

Este proyecto se enfoca en el análisis y modelado de datos de la Premier League para el **Taller 2 de Machine Learning I (2026-I)** de la Universidad Externado de Colombia. El objetivo principal es superar el benchmark de precisión de **Bet365 (49.8%)** utilizando modelos de aprendizaje automático basados en eventos reales de la temporada 2025-26.

## 🎯 Objetivos del Proyecto

1.  **Modelo 1: xG (Expected Goals)** - Regresión Logística para predecir la probabilidad de que un tiro termine en gol (`P(gol | tiro)`).
2.  **Modelo 2: Match Predictor**
    -   **Parte A (Regresión Lineal)**: Predecir el número total de goles de un partido.
    -   **Parte B (Regresión Logística)**: Clasificación multiclase para predecir el resultado (Home, Draw, Away).
3.  **Dashboard Interactivo**: Visualización de mapas de tiros (Shot Maps), predictor de partidos y análisis exploratorio (EDA).

## 📊 Datos y API

Los datos provienen del [API de la Premier League](https://premier.72-60-245-2.sslip.io/).
-   **Jugadores**: Estadísticas de 822 jugadores.
-   **Partidos**: Datos de 291 partidos (de 380) con cuotas de apuestas integradas.
-   **Eventos**: Más de 444,252 eventos detallados (pases, tiros, tarjetas, etc.) con coordenadas `(x, y)`.

El repositorio ahora separa dos problemas distintos:

1. `Shot xG`
   Predice la probabilidad de gol de un tiro individual.
   Modelos incluidos:
   `Regresion lineal polinomica`
   `Regresion logistica binaria`

2. `Match Predictor`
   Predice el comportamiento global de un partido completo.
   Modelos incluidos:
   `Regresion lineal` para `total_goals`
   `Regresion logistica multiclase` para `H / D / A`

## Objetivo del proyecto

Seguir el lineamiento del taller y construir un dashboard reproducible que permita:

- analizar tiros y contexto de gol
- comparar modelos de xG
- estimar goles esperados totales de un partido
- predecir si gana local, empata o gana visitante
- contrastar el predictor de partido con el benchmark de `Bet365`

## Estado actual

### 1. Feature engineering de tiros

Desde `data/raw/events.csv` se construye `data/processed/shots_features.csv` usando [src/features/build_features.py](src/features/build_features.py).

Variables disponibles para xG:

- `distance`
- `angle`
- `is_header`
- `is_big_chance`
- `is_penalty`
- `is_counter`
- `is_right_foot`
- `is_left_foot`
- `is_from_corner`
- `is_volley`
- `is_first_touch`

### 2. Modelos de xG por tiro

Se trabajan dos enfoques sobre tiros:

- `Regresion lineal polinomica`
  usa split `train / test / revalidation`
  compara grados `1, 2 y 3`
  selecciona el mejor por estabilidad en revalidacion

- `Regresion logistica binaria`
  sigue el criterio del PDF del taller
  usa split `70 / 15 / 15`
  aplica `StandardScaler` a `distance` y `angle`
  usa `class_weight="balanced"`
  evalua con `accuracy`, `precision`, `recall`, `f1` y `AUC`

### 3. Match Predictor por partido

Trabaja sobre `data/matches.csv`.

Modelos implementados:

- `Regresion lineal` para predecir `total_goals`
- `Regresion logistica multiclase` para predecir `ftr`:
  `H` = Home win
  `D` = Draw
  `A` = Away win

Features usadas actualmente:

- estadisticas del local: `hs`, `hst`, `hc`, `hf`, `hy`, `hr`
- estadisticas del visitante: `as_`, `ast`, `ac`, `af`, `ay`, `ar`
- odds: `Bet365`, `BetWay`, `Max`, `Average`
- probabilidades implicitas
- arbitro (`referee`)

Benchmark actual considerado:

- `Bet365` para clasificacion del resultado del partido

## Dashboard

El proyecto ahora tiene dos frontends:

- `V1 Streamlit` en [src/dashboard_premier_league.py](src/dashboard_premier_league.py)
- `V2 Dash` en [src/app_dash.py](src/app_dash.py)

### V1 Streamlit

Es la version academica y funcional donde consolidamos:

Incluye estas secciones:

- `Overview`
- `Match Predictor`
- `Shot Lab`
- `Match Replay`
- `Performance`
- `EDA`

### Lo que muestra hoy

- comparacion entre `xG linear` y `xG logistic`
- shot map interactivo
- replay interactivo de partidos usando secuencias de eventos reales
- umbrales para clasificacion de gol
- metricas de overfitting entre splits
- predictor de partido con seleccion de equipos
- input interactivo de `referee` y odds `Bet365`
- comparacion del predictor multiclase contra el benchmark de `Bet365`

### Match Replay

La nueva vista `Match Replay` usa `event data` del archivo `data/raw/events.csv`.

Permite:

- elegir un partido especifico
- filtrar por equipo
- filtrar por jugador
- filtrar por tipo de evento
- limitar por rango de minutos
- reproducir secuencias de acciones sobre el campo
- ver trayectorias usando `x, y, end_x, end_y`

### V2 Dash

La nueva interfaz en Dash esta inspirada en una experiencia tipo sportsbook y apunta a una navegacion mas limpia y visual.

Componentes actuales:

- header del partido con marcador final y contexto
- layout de tres columnas
- panel izquierdo con selector de modo (Historical vs Simulator) y filtros dinámicos
- modo Simulador: permite elegir cualquier equipo y árbitro para predecir el resultado mediante promedios históricos
- replay central con cancha expandida (850px) y animación de eventos minuto a minuto
- panel derecho con eventos recientes y resumen detallado por equipo
- tarjetas de predicción con probabilidades H/D/A, goles esperados y "goal pressure" en tiempo real
- visualización de xG en el hover de cada evento sobre el campo
- capa visual y texturas cargadas desde `assets/app_dash.css`

## Estructura del repositorio

```text
data/
  raw/
  processed/
  matches.csv
  players.csv
  standings.csv
  taller2-ml1-premier-league.pdf

notebooks/
  01_xg_model_logistic_regression.py
  crisp_dm_data_mining.ipynb
  eda_premier_league.ipynb
  taller2_ml_premier_league_final.ipynb
  xg_baseline_naive_bayes.ipynb
  xg_model_linear_regression.ipynb
  xg_threshold_optimization.ipynb

scripts/
  download_csvs.py
  download_events.py
  optimize_threshold.py

assets/
  app_dash.css

src/
  app_dash.py
  dashboard_premier_league.py
  train_goal_prediction.py
  train_model.py
  features/
    build_features.py
```

## Instalación y Reproducibilidad

1.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Descargar Datos**: Ejecutar el script `scripts/download_csvs.py` (actualizado para entornos Linux) para obtener los datasets completos localmente.
3.  **Explorar Notebooks**: Los análisis se encuentran en la carpeta `notebooks/`.

## 🛠️ Avances Recientes (Marzo 2026)
- ✅ Organización inicial del repositorio.
- ✅ Integración de Git LFS para manejo de datasets pesados (212MB+ de eventos).
- ✅ Extracción de requerimientos directamente desde el PDF del taller.
- ✅ Corrección de rutas de acceso a datos para compatibilidad multiplataforma.

---
*Organizado por Antigravity AI.*
