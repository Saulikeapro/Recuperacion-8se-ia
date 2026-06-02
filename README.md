EXAMEN RECUPERACION IA Clasificacion con redes neuronales

Martinez Monge Saul Guillermo 22760566

Este proyecto ayuda predecir la presencia o ausencia de una enfermedad cardiaca en pacientes basandose en resultados de pruebas clinicas

Primero ejecutar el comando:
pip install -r requirements.txt
para verificar si las librerias se instalaon correctamente ejecutar:
python heart_disease_eda.py
para correr el primer scrip

el orden para ejecutar este proyecto es el siguiente

1. heart_disease_eda.py con el comando python heart_disease_eda.py
2. mlp_training.py con el comando python mlp_training.py
3. mlp_inference.py con el comando python mlp_inference.py

El heatmap revela una correlación positiva moderada entre el target y variables como cp y thal, indicando su relevancia para el modelo inversamente la variable thalach muestra una correlación negativa con la presencia de enfermedad cardíaca
<img width="548" height="395" alt="image" src="https://github.com/user-attachments/assets/1300b577-7092-47a8-b60b-be274b0ecaab" />

Se observa que la variable de edad (age) presenta una distribución normal, concentrando a la mayoría de los pacientes entre los 50 y 60 años
<img width="557" height="441" alt="image" src="https://github.com/user-attachments/assets/31587bf1-a1bb-45c4-ad76-03c5d379261d" />

La grafica de frecuencia muestra que el sexo masculino predomina en esta muestra del dataset de Cleveland asimismo el tipo de dolor de pecho (cp) más frecuente es el nivel 0 lo que sugiere una alta presencia de dolor asintomatico en los sujetos analizados
<img width="649" height="437" alt="image" src="https://github.com/user-attachments/assets/61400ec2-0abf-4110-9a40-06d05fcdf336" />

Al comparar las distribuciones, se nota que los pacientes con enfermedad clase 1 tienden a mostrar una frecuencia cardíaca máxima (thalach) más baja en comparación con los pacientes sanos esto sugiere que thalach es un indicador fisiologico importante para diferenciar entre ambas clases
<img width="682" height="419" alt="image" src="https://github.com/user-attachments/assets/19dfab49-e524-436b-9b65-c07ccd015526" />

Conclucion
¿Que tan bien generaliza el modelo?
Presenta una buena camacidad para generalizar
¿Hay indicios de overfitting o underfitting?
no hay indicadores de overfitting
¿Cuál clase se predice mejor (0 o 1) y por qué crees que ocurre esto?
En este dataset la clase 0 (Sano) suele tener numeros ligeramente superiores
¿Qué variables del EDA considera más relevantes para la predicción del modelo?
cp (Tipo de dolor de pecho): Es la variable con mayor correlación positiva
thalach (Frecuencia cardiaca maxima): Mostro una correlación negativa clara en el EDA
ca (Vasos coloreados): El numero de vasos principales obstruidos es un predictor fisico crítico que el modelo MLP prioriza para separar las clases
