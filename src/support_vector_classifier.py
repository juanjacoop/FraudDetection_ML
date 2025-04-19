from sklearn.svm import SVC 
import pandas as pd 
import numpy as np 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
import os

log_file = r"docs\classifier_logs\log_svc.txt"

# Crear el archivo si no existe
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("=== Registro de Entrenamiento SVC ===\n\n")

with open(log_file, "w") as log:


    # Directorio base
    base_dir = os.path.join('data', 'out')

    # Directorio base
    base_dir = os.path.join('data', 'out')

    # Cargar los DataFrames
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
    val_df = pd.read_csv(os.path.join(base_dir, 'val.csv'))

    # Declaracion de la variable objetivo
    target = 'Class'

    """
#==========================================================================================================================
    # Cargo un 20% de mi dataset de prueba ya que no tengo la capacidad para entrenar un modelo en local
    # Esto es solo de prueba para verificar que el codigo esta bien y se puede comentar/borrar al final.
    train_df, _ = train_test_split(train_df, 
                               train_size=0.2, 
                               stratify=train_df['Class'], 
                               random_state=42)
#========================================================================================================================== 
    """   
    #Separacion de caracteristicas y variable objetivo
    x_train, y_train = train_df.drop(columns=target), train_df[target]
    x_test, y_test = test_df.drop(columns=target), test_df[target]
    x_val, y_val = val_df.drop(columns=target), val_df[target]


    #Instanciación y utlizacion de la tecnica SMOTE 

    rus = RandomUnderSampler(random_state=42)
    x_train_smt, y_train_smt = rus.fit_resample(x_train, y_train)
    print(y_train_smt.value_counts())

    # Instanciacion del modelo y sus hiperparametros a probar
    clf = SVC()



    param_grid = [
    #{'C': [1], 'kernel': ['linear']}
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['sigmoid']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    ]

    grid = GridSearchCV(clf, param_grid, refit = True, verbose = 3) 

    # Entrenamiento del Modelo  
    grid.fit(x_train_smt, y_train_smt) 


    #Extraccion de mejores hiperparametros y pruebas de validez del modelo, su seguida exportacion a un registro.
    grid_predictions = grid.predict(x_test) 
    
    best_params = grid.best_params_
    best_estimator = grid.best_estimator_


    log.write(f"Target Value Counts: {y_train_smt.value_counts()}\n\n")
    log.write(f"Best params: {best_params}\n\n")
    log.write(f"Best estimator: {best_estimator}\n\n")

    train_preds =  grid.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_preds)
    train_report = classification_report(y_train, train_preds)
    train_cm = confusion_matrix(y_train, train_preds)

    log.write("===Dataset de Training ===\n")
    log.write(f"Accuracy Score: {train_accuracy:.4f}\n")
    log.write(f"Classification Report:\n{train_report}\n")
    log.write(f"Confusion Matrix:\n{train_cm}\n\n")

    test_preds = grid.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_report = classification_report(y_test, test_preds)
    test_cm = confusion_matrix(y_test, test_preds)

    log.write("===Dataset de Testing ===\n")
    log.write(f"Accuracy Score: {test_accuracy:.4f}\n")
    log.write(f"Classification Report:\n{test_report}\n")
    log.write(f"Confusion Matrix:\n{test_cm}\n\n")


    val_preds = grid.predict(x_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    val_report = classification_report(y_val, val_preds)
    val_cm = confusion_matrix(y_val, val_preds)


    log.write("=== Dataset de Validacion ===\n")
    log.write(f"Accuracy Score: {val_accuracy:.4f}\n")
    log.write(f"Classification Report:\n{val_report}\n")
    log.write(f"Confusion Matrix:\n{val_cm}\n\n")

    log.write("=== Bondad del Modelo ===\n")
    log.write(f"Train Accuracy: {train_accuracy:.4f}\n")
    log.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    log.write(f"Validation Accuracy: {val_accuracy:.4f}\n")

    if (train_accuracy - test_accuracy > 0.05) or (train_accuracy - val_accuracy > 0.05) > 0.05:
        log.write("Posible overfitting detectado (diferencia > 5%)\n")
    else:
        log.write("No se detecta overfitting significativo\n")