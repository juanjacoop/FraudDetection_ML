# Proyecto Fraud_Detection

## Estructura de directorios

```
/Fraud_Detection  
├── /Data              — *Almacena los datos del proyecto*  
│   ├── in             — *Datos sin procesar*  
│   └── out            — *Datos procesados y listos para usar si se requiere*  
│  
├── /src               — *Código fuente del proyecto*  
│   ├── Eda.ipynb      — *Análisis exploratorio de datos (EDA)*  
│   ├── utils.py       — *Funciones auxiliares*  
│   ├── app.py         — *Aplicación principal (si aplica)*  
│   ├── classifier.py  — *Script de entrenamiento del modelo de clasificación*  
│   ├── preprocess.py  — *Preprocesamiento de datos*  
│   └── train.py       — *Pipeline de entrenamiento del modelo*  
│  
├── /doc               — *Documentación del proyecto*  
│   ├── README.md      — *Descripción general del proyecto*  
│   └── report.pdf     — *Informe final del proyecto*  
│  
└── requirements.txt   — *Lista de dependencias del entorno*  
```
Descripcion proyecto Fraud_Detection:
  Este proyecto tiene como objetivo contruir modelos de machine learning que sean capaces de predecir las transacciones fraudulentas con tarjetas de credito. Nos hemos enfrentado ante problemas con los datos proporcionados por lo que hemos tenido que realizar estrategias de preprocesamiento, transformacion PCA, y balancear las clases para obtener un mejor rendimiento de los modelos. 
