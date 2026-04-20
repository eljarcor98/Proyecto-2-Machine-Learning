# Premier League Match Lab

Proyecto 2 de Machine Learning enfocado en analisis de partidos de Premier League, modelado de xG por tiro y prediccion del resultado de un partido.

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

Desde `data/raw/events.csv` se construye `data/processed/shots_features.csv` usando [src/features/build_features.py](/c:/Users/Arnold's/Documents/Repositorios%20Machine%20Learning/Proyecto%202%20Machine%20Learning/src/features/build_features.py).

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

- `V1 Streamlit` en [src/dashboard_premier_league.py](/c:/Users/Arnold's/Documents/Repositorios%20Machine%20Learning/Proyecto%202%20Machine%20Learning/src/dashboard_premier_league.py)
- `V2 Dash` en [src/app_dash.py](/c:/Users/Arnold's/Documents/Repositorios%20Machine%20Learning/Proyecto%202%20Machine%20Learning/src/app_dash.py)

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

Importante:

- esta vista anima eventos y trayectorias
- no reconstruye tracking continuo de los 22 jugadores
- para movimiento real frame a frame haria falta `tracking data`

### V2 Dash

La nueva interfaz en Dash esta inspirada en una experiencia tipo sportsbook y apunta a una navegacion mas limpia y visual.

Componentes actuales:

- header del partido con marcador final y contexto
- layout de tres columnas
- panel izquierdo de filtros
- replay central con cancha y animacion
- panel derecho con eventos recientes y resumen por equipo
- tarjetas compactas de acciones, tiros, goles y jugadores
- barras de momentum por equipo
- bloque de prediccion con ganador probable, goles esperados y probabilidades H/D/A
- capa visual y texturas cargadas desde `assets/app_dash.css`

Objetivo de esta version:

- ofrecer una interfaz mas cercana a producto
- mejorar la lectura de partido respecto a la V1
- servir como base para una futura interfaz mas avanzada

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

## Instalacion

### 1. Crear entorno virtual

En PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

```powershell
pip install -r requirements.txt
```

Si estas usando el `.venv` manualmente y quieres asegurarte de instalar en ese entorno exacto:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Como abrir el dashboard

### Opcion 1: Streamlit V1

Desde la raiz del proyecto ejecuta:

```powershell
streamlit run src/dashboard_premier_league.py
```

Normalmente Streamlit abre el navegador automaticamente. Si no lo hace, copia en tu navegador la URL local que aparezca en consola, usualmente:

```text
http://localhost:8501
```

### Opcion 2: Dash V2

Desde la raiz del proyecto ejecuta:

```powershell
python src/app_dash.py
```

Luego abre:

```text
http://127.0.0.1:8050
```

Si la interfaz aparece sin estilos, reinicia la app y recarga el navegador con `Ctrl + F5`. La V2 depende de `assets/app_dash.css`.

## Flujo recomendado para correr todo desde cero

1. Activar el entorno virtual.
2. Instalar dependencias.
3. Regenerar features de tiros si hace falta.
4. Lanzar el dashboard.

```powershell
python src/features/build_features.py
streamlit run src/dashboard_premier_league.py
```

O si quieres abrir la interfaz nueva:

```powershell
python src/features/build_features.py
python src/app_dash.py
```

## Scripts utiles

- `python src/features/build_features.py`
  reconstruye `shots_features.csv`

- `python scripts/download_csvs.py`
  descarga archivos tabulares desde el export del proyecto a `data/raw`

- `python scripts/download_events.py`
  descarga `events.csv` con reintentos y escritura por bloques

- `python src/train_goal_prediction.py`
  prueba la regresion lineal base para goles

- `python src/train_model.py`
  prueba un flujo base de clasificacion sobre partidos

- `python scripts/optimize_threshold.py`
  explora umbrales para el modelo lineal de xG

- `python src/app_dash.py`
  abre la version 2 del dashboard en Dash

## Metodologia y criterio del taller

El proyecto toma como referencia el documento `data/taller2-ml1-premier-league.pdf`.

Puntos clave adoptados:

- separar claramente problemas de tiros y de partidos
- no evaluar solo con accuracy cuando hay clases desbalanceadas
- usar validacion honesta con multiples segmentos
- comparar contra benchmark de apuestas
- reportar metricas interpretables dentro del dashboard

## Notas

- `xG logistic` y `xG linear` resuelven el problema `gol / no gol` por tiro
- el `Match Predictor` resuelve un problema distinto: comportamiento global del partido
- el predictor de partido actual ya usa odds y arbitro, pero aun se puede mejorar con rolling averages y features agregadas de `events`
- el `Match Replay` usa eventos reales del feed, por lo que funciona muy bien para secuencias y patrones, pero no reemplaza tracking data
- `Dash V2` y `Streamlit V1` conviven; la primera busca mejor UX y la segunda concentra el laboratorio completo actual

## Proximos pasos sugeridos

- mejorar el modelo lineal de `total_goals`
- agregar rolling features por equipo
- agregar features agregadas por partido desde `events`
- validar con cross-validation adicional para el predictor de partido
- documentar resultados finales en notebook o reporte
