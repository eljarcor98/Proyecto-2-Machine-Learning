# ⚽ Premier League 2025-26: Avances en Rama `arnold_avances`

Este repositorio contiene los avances del Proyecto 2 de Machine Learning, enfocados en superar los benchmarks de las casas de apuestas (Bet365).

## 🚀 Estado Actual

Hemos implementado las primeras fases de la metodología **CRISP-DM**:

### 1. Comprensión del Negocio y Datos
- **Objetivo**: Superar el 49.8% de precisión de Bet365.
- **Notebook**: `notebooks/crisp_dm_data_mining.ipynb` contiene la documentación detallada de los objetivos y el diccionario de variables.

### 2. Ingeniería de Características (Feature Engineering)
Hemos "minado" la columna `qualifiers` (datos JSON anidados) para extraer variables críticas de gol.
- **Script**: `src/features/build_features.py`
- **Dataset Generado**: `data/processed/shots_features.csv` (7,198 tiros)
- **Variables Extraídas**: Distancia, ángulo, BigChance, Header, Volley, ICT Index, etc.

### 3. Modelado y Benchmarking (Fases 4 y 5)
Hemos validado el desempeño contra el baseline del 88%:
- **Naive Bayes**: Confirmamos que llega al 90.2% de accuracy, pero con un bajo recall (19%), demostrando el sesgo por desbalanceo de clases.
- **Regresión Lineal**: Alcanzó un **89.65% de Accuracy** en el set de Test y **88.75%** en Retest.
- **Optimización de Umbral**: Encontramos que bajar el umbral de decisión a **0.20** permite capturar el **58% de los goles reales** (comparado con solo el 8% usando el estándar de 0.5).

## 🛠️ Notebooks Principales
1. `notebooks/crisp_dm_data_mining.ipynb`: Comprensión técnica.
2. `notebooks/xg_model_linear_regression.ipynb`: Entrenamiento del modelo base.
3. `notebooks/xg_baseline_naive_bayes.ipynb`: Verificación del baseline del 88%.
4. `notebooks/xg_threshold_optimization.ipynb`: Análisis de Precision vs Recall.

## 🚀 Cómo ejecutar
Para regenerar las características y ver la optimización:
```bash
python src/features/build_features.py
python scripts/optimize_threshold.py
```

---
*Documentado por Antigravity AI para el equipo de Arnold.*
