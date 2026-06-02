import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv('data/heart.csv', names=column_names, na_values='?')
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
df = df.fillna(df.median())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', 
                    solver='adam', max_iter=500, random_state=42, early_stopping=True)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
print(classification_report(y_test, y_pred))

model_path = 'models/mlp_heart_model.pkl'
joblib.dump(mlp, model_path)

if os.path.exists(model_path):
    print(f"Modelo guardado. Tamaño: {os.path.getsize(model_path)/1024:.2f} KB")

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues')
plt.title("Matriz de Confusion")
plt.savefig('matriz_confusion.png')
plt.show()