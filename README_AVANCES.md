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
- **Variables Extraídas**:
  - **Geométricas**: Distancia al arco, ángulo de tiro.
  - **Técnicas**: Cabezazo, pie derecho/izquierdo, volea, primera intención.
  - **Contexto**: BigChance (oportunidad clara), Penal, Contraataque, Desde Corner.
  - **Jugador**: Índice ICT (FPL) y xG acumulado de la temporada.

## 🛠️ Cómo ejecutar
Para regenerar las características desde los datos crudos:
```bash
python src/features/build_features.py
```

---
*Documentado por Antigravity AI para el equipo de Arnold.*
