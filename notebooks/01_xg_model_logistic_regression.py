# %% [markdown]
# # ⚽ Modelo 1: Expected Goals (xG) - Regresión Logística
# Este script interactivo (formato de celdas '# %%' de Python/Jupyter) aborda el 
# entrenamiento de un modelo para predecir si un tiro resulta en gol.
# La partición elegida es: Train (70%), Validación (15%), Test (15%).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Configuración visual
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Carga y Filtrado de Datos
# Aca cargaremos los eventos brutos y extraeremos únicamente los tiros.

# %%
# Definir la ruta de los datos 
file_path = "../data/raw/events.csv"

# Cargando eventos.
print("Cargando events.csv... Esto puede tomar unos segundos.")
df_events = pd.read_csv(file_path, low_memory=False)

# Extraer solo los tiros
print(f"Total eventos cargados: {len(df_events)}")

# Filtrar según la columna 'is_shot' (o identificando los tiros en caso de no existir de manera clara)
if 'is_shot' in df_events.columns:
    df_shots = df_events[df_events['is_shot'] == True].copy()
else:
    # Alternativa si es un API dump sin boolean
    df_events['is_shot'] = df_events['type'].astype(str).apply(
        lambda x: any(kw in x for kw in ['Goal', 'MissedShots', 'SavedShot', 'ShotOnPost'])
    )
    df_shots = df_events[df_events['is_shot'] == True].copy()
    
print(f"Total de tiros (Shots) convertidos en observaciones: {len(df_shots)}")

# %% [markdown]
# ## 2. Feature Engineering (Ingeniería de Características)
# Convertiremos campos JSON a features matemáticas útiles referidas a la probabilidad de gol.

# %%
print("Extrayendo Features de la columna qualifiers...")

# Usamos operaciones vectorizadas (str.contains) que son MUY rápidas vs un Json parser 1 a 1
q = df_shots['qualifiers'].astype(str)

df_shots['is_big_chance'] = q.str.contains('BigChance', na=False, case=False).astype(int)
df_shots['is_penalty'] = q.str.contains('Penalty', na=False, case=False).astype(int)
df_shots['is_head'] = q.str.contains('Head', na=False, case=False).astype(int)
df_shots['is_right_foot'] = q.str.contains('RightFoot', na=False, case=False).astype(int)
df_shots['is_left_foot'] = q.str.contains('LeftFoot', na=False, case=False).astype(int)
df_shots['is_first_touch'] = q.str.contains('FirstTouch', na=False, case=False).astype(int)
df_shots['is_volley'] = q.str.contains('Volley', na=False, case=False).astype(int)

# Target Feature (Gol o no gol):
# Muchas veces está explícito como 'is_goal' en el evento o como GoalMouth
df_shots['is_goal'] = q.str.contains('Goal', na=False, case=False).astype(int)
# Sumamos los tipos de evento que definen un Goal directo, por si aca.
if 'is_goal' not in df_shots.columns or df_shots['is_goal'].max() == 0:
    df_shots.loc[df_shots['type'].astype(str).str.contains('Goal'), 'is_goal'] = 1

print(f"Tiros que son Goles: {df_shots['is_goal'].sum()} vs No Goles: {len(df_shots) - df_shots['is_goal'].sum()}")

# Features Geométricas (Distancia y ángulo) 
# X: de 0 a 100, Y: de 0 a 100. El arco rival en normalización de Opta/Premier API está típicamente en (100, 50).
df_shots['x'] = pd.to_numeric(df_shots['x'], errors='coerce')
df_shots['y'] = pd.to_numeric(df_shots['y'], errors='coerce')

# Distancia Euclídea base al centro del arco
df_shots['distancia_arco'] = np.sqrt((100 - df_shots['x'])**2 + (50 - df_shots['y'])**2)

# Ángulo (usando trigonometría para representar el ángulo de visión) convertido a grados
df_shots['angulo_arco'] = np.arctan2(np.abs(50 - df_shots['y']), 100 - df_shots['x']) * (180/np.pi)

# Quitamos registros donde las coordenadas eran nulas 
# (Ej: Substitutions o Cards que por error tuvieran is_shot==True)
df_shots = df_shots.dropna(subset=['distancia_arco', 'angulo_arco', 'is_goal'])

# %% [markdown]
# ## 3. Train / Validation / Test Split (70% - 15% - 15%)
# Para predecir xG mantemos la proporción original de goles ("stratify").

# %%
features = [
    'distancia_arco', 'angulo_arco', 'is_big_chance', 'is_penalty', 
    'is_head', 'is_right_foot', 'is_left_foot', 'is_first_touch', 'is_volley'
]
target = 'is_goal'

X = df_shots[features]
y = df_shots[target]

# 1. Separar 70% Train, 30% Temporal (Para Val + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=123
)

# 2. Separar el 30% Temporal en Mitades iguales (15% Val y 15% Test)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=123
)

print("\n--- TAMAÑOS DE PARTICIÓN ---")
print(f"Train:      {X_train.shape[0]} registros ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation: {X_val.shape[0]} registros ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test:       {X_test.shape[0]} registros ({len(X_test)/len(X)*100:.1f}%)")

# Escalar Variables Continuas Geométricas para Regresión Logística
scaler = StandardScaler()
cols_to_scale = ['distancia_arco', 'angulo_arco']

# FIT SOLO EN TRAIN (Evita Data Leakage!!!)
X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_val.loc[:, cols_to_scale]  = scaler.transform(X_val[cols_to_scale])
X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# %% [markdown]
# ## 4. Entrenamiento y Evaluación sobre Validación

# %%
# Modelo Base Regresión Logística
# Usamos class_weight='balanced' para indicarle a la Regresión que atienda la clase minoritaria (Gol).
model = LogisticRegression(class_weight='balanced', penalty='l2', max_iter=1000, random_state=123)
model.fit(X_train, y_train)

# Evaluamos en conjunto de Validación
y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]

print("\n--- MÉTRICAS EN CONJUNTO DE VALIDACIÓN (15%) ---")
print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall:    {recall_score(y_val, y_val_pred):.4f}")
print(f"F1 Score:  {f1_score(y_val, y_val_pred):.4f}")
print(f"AUC ROC:   {roc_auc_score(y_val, y_val_prob):.4f}")

# %% [markdown]
# ## 5. Evaluación Definitiva en Test

# %%
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

print("\n--- MÉTRICAS DEFINITIVAS EN CONJUNTO TEST (15%) ---")
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_test_pred):.4f}")
print(f"AUC ROC:   {roc_auc_score(y_test, y_test_prob):.4f}")

# Comparando con el Baseline Naive de la industria.
# Un modelo Naive predice "0" siempre debido a que muy poquitos tiros son goles.
baseline_preds = np.zeros(len(y_test))
baseline_acc = accuracy_score(y_test, baseline_preds)

print(f"\nAccuracy predictivo de SIEMPRE decir 'No Gol': {baseline_acc:.4f}")
if test_acc > baseline_acc:
    print("✅ ¡El modelo supera el Accuracy del Baseline!")
else:
    print("⚠️ El modelo tiene Accuracy menor al Baseline.")
    print("   El desbalance exige evaluar la ganancia combinada del Recall (detectar goles de verdad).")

# Matriz de Confusión visual para Test
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicción: No Gol', 'Predicción: Gol'], 
            yticklabels=['Realidad: No Gol', 'Realidad: Gol'])
plt.title('Matriz de Confusión - Test (15%)')
plt.show()

# Curva ROC visual
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'Logistic Reg (AUC = {roc_auc_score(y_test, y_test_prob):.4f})', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Predictor de xG')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ## 6. Importancia de las Features Extras
# Veremos qué feature impacta más la decisión de ser gol o no (coeficientes lógicos).

# %%
coeficientes = pd.DataFrame({
    'Feature': features,
    'Coeficiente': model.coef_[0]
}).sort_values(by='Coeficiente', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=coeficientes, x='Coeficiente', y='Feature', palette='viridis')
plt.title('Impacto en la Regresión Logística de predecir Gol')
plt.show()
