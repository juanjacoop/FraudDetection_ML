from lightgbm import LGBMClassifier,early_stopping
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score,make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from graficos import *
import os

log_file = r"docs\classifier_logs\log_lgbm.txt"

# Crear el archivo si no existe
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("=== Registro de Entrenamiento LightGBM ===\n\n")

with open(log_file, "w") as log:

    # Cargar los DataFrames
    base_dir = os.path.join('data', 'out')
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
    val_df = pd.read_csv(os.path.join(base_dir, 'val.csv'))

    target = 'Class'

#==========================================================================================================================
    # Cargo un 20% de mi dataset de prueba ya que no tengo la capacidad para entrenar un modelo en local
    # Esto es solo de prueba para verificar que el codigo esta bien y se puede comentar/borrar al final.
    train_df, _ = train_test_split(train_df, 
                               train_size=0.2, 
                               stratify=train_df['Class'], 
                               random_state=42)
#========================================================================================================================== 

    
    x_train, y_train = train_df.drop(columns=target), train_df[target]
    x_test, y_test = test_df.drop(columns=target), test_df[target]
    x_val, y_val = val_df.drop(columns=target), val_df[target]

    # Con random under sampling
    # rus = RandomUnderSampler(random_state=42)
    #x_train_smt, y_train_smt = rus.fit_resample(x_train, y_train)


    # Escalado robusto -> Lo aplico en training tambien por SMOTE
    scaler = RobustScaler()
    scaler.fit(x_train)
    column_names = x_train.columns.tolist()
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)



    x_train_smt, y_train_smt  = x_train, y_train
    print(y_train_smt.value_counts())


    # LightGBM Classifier

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    clf = LGBMClassifier(objective='binary',
                     scale_pos_weight=scale_pos_weight, #scale_pos_weight da mejor resultado porque aplica una penalización más suave.
                     random_state=42)

    # clf = LGBMClassifier(objective='binary', class_weight='balanced', random_state=42)

    # Param grid para GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 7,-1],
        'num_leaves': [10,31, 50],
        'min_child_samples':[20]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, pos_label=1)
    grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scorer, verbose=5, n_jobs=-1)


    fit_params = {
    "eval_set": [(x_val, y_val)],
    "eval_metric": "auc",
    "callbacks": [early_stopping(stopping_rounds=20, verbose=False)]
    }   

    # Entrenamiento
    grid.fit(x_train_smt, y_train_smt, **fit_params)

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
    train_auc = roc_auc_score(y_train, train_preds)

    log.write("===Dataset de Training ===\n")
    log.write(f'AUC Score: {train_auc:.4f}\n')
    log.write(f"Accuracy Score: {train_accuracy:.4f}\n")
    log.write(f"Classification Report:\n{train_report}\n")
    log.write(f"Confusion Matrix:\n{train_cm}\n\n")

    test_preds = grid.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_report = classification_report(y_test, test_preds)
    test_cm = confusion_matrix(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_preds)

    log.write("===Dataset de Testing ===\n")
    log.write(f'AUC Score: {test_auc:.4f}\n')
    log.write(f"Accuracy Score: {test_accuracy:.4f}\n")
    log.write(f"Classification Report:\n{test_report}\n")
    log.write(f"Confusion Matrix:\n{test_cm}\n\n")


    val_preds = grid.predict(x_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    val_report = classification_report(y_val, val_preds)
    val_cm = confusion_matrix(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_preds)

    log.write("=== Dataset de Validacion ===\n")
    log.write(f'AUC Score: {val_auc:.4f}\n')
    log.write(f"Accuracy Score: {val_accuracy:.4f}\n")
    log.write(f"Classification Report:\n{val_report}\n")
    log.write(f"Confusion Matrix:\n{val_cm}\n\n")

    train_f1 = f1_score(y_train, train_preds, pos_label=1, zero_division=0)
    val_f1 = f1_score(y_val, val_preds, pos_label=1, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, pos_label=1, zero_division=0)

    log.write("=== F1-Scores for Class 1 ===\n")
    log.write(f"Train F1-score: {train_f1:.4f}\n")
    log.write(f"Validation F1-score: {val_f1:.4f}\n")
    log.write(f"Test F1-score: {test_f1:.4f}\n\n")


     # Guarda los graficos 

    # === Gráficos para Training ===
    plot_all_metrics(
        y_true=y_train,
        y_pred=train_preds,
        model=grid.best_estimator_,
        feature_names=column_names,
        x_features=x_train,
        dataset_name='Train',
        model_name='LIGHTGBM sin sampling'
    )

    # === Gráficos para Testing ===
    plot_all_metrics(
        y_true=y_test,
        y_pred=test_preds,
        model=grid.best_estimator_,
        feature_names=column_names,
        x_features=x_test,
        dataset_name='Test',
        model_name='LIGHTGBM sin sampling'
    )

    # === Gráficos para Validación ===
    plot_all_metrics(
        y_true=y_val,
        y_pred=val_preds,
        model=grid.best_estimator_,
        feature_names=column_names,
        x_features=x_val,
        dataset_name='Validacion',
        model_name='LIGHTGBM sin sampling'
)


