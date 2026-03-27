import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# Load data
df = pd.read_csv('data/processed/shots_features.csv')
features = ['distance', 'angle', 'is_header', 'is_big_chance', 'is_penalty', 'is_counter']
df = df.dropna(subset=features + ['is_goal'])
X = df[features]
y = df['is_goal'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_probs = model.predict(X_test)

results = []
for t in np.arange(0.05, 0.55, 0.05):
    y_pred = [1 if p > t else 0 for p in y_probs]
    results.append({
        'Threshold': round(t, 2),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred)
    })

df_output = pd.DataFrame(results)
print(df_output.to_string(index=False))
df_output.to_csv('data/processed/threshold_optimization.csv', index=False)
