# ⚽ Premier League 2025-26: xG Model & Data Mining
> **Branch:** `arnold_avances` | **Status:** ✅ Research & Initial Modeling Complete

Este repositorio contiene los avances del **Proyecto 2 de Machine Learning**, enfocados en superar los benchmarks de predicción de goles (xG) y análisis de eventos de la Premier League.

---

## 🚀 Estado del Proyecto (CRISP-DM)

Hemos implementado las fases fundamentales de la metodología para asegurar un modelo robusto:

### 1. Comprensión del Negocio y Datos 📋
- **Objetivo**: Superar la precisión del baseline de Bet365 (**49.8%**).
- **Notebook**: [crisp_dm_data_mining.ipynb](file:///c:/Users/Arnold's/Documents/Repositorios%20Machine%20Learning/Proyecto%202%20Machine%20Learning/notebooks/crisp_dm_data_mining.ipynb) (Documentación detallada y diccionario de variables).

### 2. Ingeniería de Características (Feature Engineering) 🛠️
Se ha procesado la columna `qualifiers` (datos JSON complejos) para extraer variables críticas:
- **Variables Extraídas**: Distancia al arco, ángulo de tiro, `BigChance`, `Header`, `Volley`, `ICT Index`, etc.
- **Script**: `src/features/build_features.py`
- **Dataset Generado**: `data/processed/shots_features.csv` (7,198 tiros analizados).

### 3. Modelado y Benchmarking 📊
- **Naive Bayes**: Verificamos que aunque alcanza un **90.2% accuracy**, tiene un recall muy bajo (19%) por el desbalanceo de clases.
- **Linear Regression (xG)**: Logramos un **89.65% de Accuracy** en Test.
- **Optimización de Umbral**: Ajustando el umbral a **0.20**, aumentamos el recall al **58%**, capturando significativamente más goles reales que el estándar de 0.5.

---

## 📂 Estructura del Proyecto

- `data/`: Datos `raw` y `processed` (incluyendo `shots_features.csv`).
- `notebooks/`: Análisis detallado de Naive Bayes, Regresión y Optimización.
- `src/features/`: Lógica de extracción de variables desde JSON.
- `scripts/`: Utilidades para optimización de umbrales y análisis de descriptores.

---

## 🤝 Colaboración: ¿Cómo usar estos avances?

Si eres parte del equipo y quieres integrar estos cambios en tu rama local para trabajar sobre ellos o hacer push a Git, sigue estos pasos:

### 1. Actualizar tu repositorio
```bash
git fetch origin
```

### 2. Integrar los cambios de Arnold
Estando en **tu propia rama**, ejecuta:
```bash
git merge origin/arnold_avances
```

### 3. Resolver conflictos (si los hay) y subir cambios
```bash
git push origin <tu-rama>
```

---

## 🛠️ Ejecución Rápida
Para regenerar las variables y ver la optimización de métricas:
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Construir características
python src/features/build_features.py

# 3. Ver optimización de umbral
python scripts/optimize_threshold.py
```

---
*Documentado con apoyo de Antigravity AI para el equipo de Arnold.*
