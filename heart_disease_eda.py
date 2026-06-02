import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = pd.read_csv('data/heart.csv', names=column_names, na_values='?')

print("Primeras 5 filas")
print(df.head()) 

df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

df = df.fillna(df.median())

plt.figure(figsize=(10,6))

df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].hist(figsize=(10,8))
plt.tight_layout()
plt.savefig('histogramas.png')

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Mapa de Correlación")
plt.savefig('heatmap.png')

plt.figure(figsize=(8,5))
sns.boxplot(x='target', y='thalach', data=df)
plt.title("Ritmo Cardíaco Máximo vs Target")
plt.savefig('boxplot.png')

plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df)
plt.title("Distribución de Clases (Balanceado)")
plt.savefig('countplot.png')

plt.show()