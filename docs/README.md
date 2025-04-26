# Proyecto Fraud_Detection

## Descripcion de Proyecto
Este proyecto tiene como objetivo construir modelos de machine learning capaces de predecir transacciones fraudulentas con tarjetas de crédito. Debido a la naturaleza desbalanceada de los datos, se aplicaron estrategias de preprocesamiento, reducción de dimensionalidad mediante PCA y técnicas de balanceo de clases para optimizar el rendimiento de los modelos predictivos.

---
## Dataset 

Fuente: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulbcreditcardfraud)

Cantidad de registros: 284,807 transacciones

Cantidad de variables: 30 (28 variables PCA: V1-V28, más Amount y Time)

Variable objetivo: Class(Binaria, 1 - Frude , 0 - No Fraude)

Particularidades:

  * Solo el 0.172% de las transacciones son fraudulentas.

  * Variables anonimizadas a través de PCA para preservar confidencialidad.

---

## Preprocesamiento

   * División estratificada en train, validation y test (70%-15%-15%).

   *  Escalado robusto (RobustScaler) para mitigar el efecto de outliers.

   *  Opcional: Reducción de dimensionalidad adicional mediante PCA.

   *  Técnicas de balanceo: Random UnderSampling y SMOTE dependiendo del modelo.

--- 
## Modelos Utilizados
  * LightGBM: Modelo de referencia rápido y eficiente.

  * Support Vector Classifier (SVC): Evaluado con y sin Random Undersampling.

  * XGBoost: Usado para buscar mejoras de rendimiento.

Cada modelo fue evaluado principalmente con métricas de Recall, F1-Score y AUC-ROC debido al alto desbalance.
---
## Cómo Reproducir el Proyecto

  *  Clona este repositorio.

  *  Instala las dependencias necesarias:

````
pip install -r requirements.txt
````

Ejecuta los scripts en el siguiente orden:

    src/EDA.ipynb(Para eliminar outliers)

    src/train_test_split.py (para generar splits escalados)

    src/lightgbm_classifier.py, src/support_vector_classifier.py, etc.

    src/final_models.ipynb 

Los gráficos se guardarán automáticamente en docs/plots/ por modelo y conjunto (Train, Test, Validation).

## Estructura de directorios
```
FRAUDDETECTION_ML/
├── data/
│   └── in/
│       ├── Fraud_Detection_Dataset.zip [ Dataset ] 
│       └── fraud_detection.zip [ Copia del Dataset ]
├── data/
│   └── out/
│       ├── test.csv [ Outputs de train_test_split]
│       ├── train.csv [ Outputs de train_test_spli ]
│       └── val.csv [ Outputs de train_test_spli ]
├── docs/
│   ├── classifier_logs/
│   │   ├── log_lgbm.txt [metricas del modelo optimo de lightgbm_classifier.py ]
│   │   ├── log_svc_rus.txt [metricas de modelo optimo de Support_Vector_Classifier_RUS.py ]
│   │   └── log_svc.txt [metricas de modelo optimo de support_vector_classifier.py ]
|   └── plots/ [graficos separados mpor disntintos mdoelso ]
|   └── README.md
├── src/
│   ├── EDA_fraud_detection.ipynb [Analisis Exploratorio y Limpieza de Outliers ]
│   ├── final_models.ipynb [Modelo Xgbost2 ]
│   ├── lightgbm_classifier.py [Script de entrenamiento de lightGBM ]
│   ├── Support_Vector_Classifier_RUS.py [Script de Entrenamiento de SVC con undersampling ]
│   ├── support_vector_classifier.py [Script de entrenamiento de de SVC sin sampling]
|   ├── graficos.py [Funciones para pintar graficos output en graficos/]  
│   └── train_test_split.py [split de dat en train,test y val en 70,15,15]
├── .gitignore 
└── requirements.txt 

```
