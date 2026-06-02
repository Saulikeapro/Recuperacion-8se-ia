import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mlp = joblib.load('models/mlp_heart_model.pkl')
scaler = joblib.load('models/scaler.pkl')

print("Parámetros del modelo:", mlp.get_params())

nuevos_pacientes = np.array([
    [55, 1, 0, 140, 250, 0, 1, 150, 0, 1.2, 1, 0, 2],
    [40, 0, 1, 120, 200, 0, 0, 172, 0, 0.0, 2, 0, 1],
    [65, 1, 3, 150, 280, 1, 2, 110, 1, 2.5, 1, 3, 3]
])

pacientes_scaled = scaler.transform(nuevos_pacientes)
predicciones = mlp.predict(pacientes_scaled)
probabilidades = mlp.predict_proba(pacientes_scaled)

for i, pred in enumerate(predicciones):
    estado = "Enfermo" if pred == 1 else "Sano"
    confianza = probabilidades[i][pred] * 100
    print(f"Paciente {i+1}: {estado} ({confianza:.2f}% de confianza)")

plt.bar(['P1', 'P2', 'P3'], probabilidades[:, 1], color='orange')
plt.title("Probabilidad de Enfermedad por Paciente")
plt.ylabel("Probabilidad (0 a 1)")
plt.savefig('inference_chart.png')
plt.show()