import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score

# Carpeta base para guardar los gráficos
EXPORT_DIR = r'docs\plots'

def create_model_folder(model_folder):
    """
    Crea y retorna una subcarpeta dentro del directorio de exportación según el nombre del modelo.
    """
    path = os.path.join(EXPORT_DIR, model_folder)
    os.makedirs(path, exist_ok=True)
    return path

def save_plot(name, model_folder):
    """
    Guarda el gráfico actual en el directorio del modelo.
    """
    create_model_folder(model_folder)
    filepath = os.path.join(EXPORT_DIR,model_folder, f'{name}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, dataset_name='Dataset', model_folder=None):
    """
    Grafica y guarda la matriz de confusión normalizada.
    """
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(5,5))
    sns.heatmap(cmn, annot=True, fmt='.2%', cmap='Blues')
    plt.title(f'{dataset_name} - Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    save_plot(f'{dataset_name}_confusion_matrix', model_folder)

def plot_roc_curve(y_true, y_pred_proba, dataset_name='Dataset', model_folder=None):
    """
    Grafica y guarda la curva ROC y muestra el AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
    plt.title(f'{dataset_name} - Curva ROC')
    plt.legend(loc='lower right')
    plt.grid()
    save_plot(f'{dataset_name}_roc_curve', model_folder)

def plot_precision_recall_curve_custom(y_true, y_pred_proba, dataset_name='Dataset', model_folder=None):
    """
    Grafica y guarda la curva de Precisión-Recall.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.figure(figsize=(6,6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precisión')
    plt.title(f'{dataset_name} - Curva Precisión-Recall')
    plt.grid()
    save_plot(f'{dataset_name}_precision_recall_curve', model_folder)

def plot_feature_importance(model, feature_names=None, max_num_features=10, dataset_name='Dataset', model_folder=None):
    """
    Grafica y guarda la importancia de las características.
    Funciona para modelos que tienen `feature_importances_` (RandomForest, XGBoost, LightGBM, CatBoost).
    Opcionalmente, también puede usar `coef_` para modelos lineales (SVC, LogisticRegression, etc.).
    """
    importances = None
    model_type = type(model).__name__.lower()

    # Si no se pasa `feature_names`, obtenemos los nombres de las columnas del dataset
    if feature_names is None:
        if hasattr(model, 'feature_importances_'):
            # Si el modelo tiene `feature_importances_` (como RandomForest, XGBoost, LightGBM, etc.)
            # Lo asumimos como un modelo que puede usar los índices del dataset
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None

        if feature_names is None:
            print(f'No se proporcionaron nombres de características y el modelo no tiene "feature_importances_" o "feature_names_in_".')
            return

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    
    elif hasattr(model, 'coef_'):  # Para SVC lineal, regresiones lineales, etc
        importances = np.abs(model.coef_).flatten()
    
    else:
        print(f'El modelo {model_type} no tiene feature_importances_ ni coef_. No se puede graficar la importancia.')
        return

    # Top N features
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:max_num_features]
    features, scores = zip(*feat_imp)

    plt.figure(figsize=(8,6))
    sns.barplot(x=scores, y=features, orient='h', palette='viridis')
    plt.title(f'{dataset_name} - Importancia de Características')
    plt.xlabel('Importancia')
    plt.ylabel('Características')
    plt.tight_layout()
    save_plot(f'{dataset_name}_feature_importance', model_folder)

    
def plot_all_metrics(y_true, y_pred, model=None, x_features=None, dataset_name='Dataset', model_name=None, feature_names=None):
    """
    Generate and save all the metrics:
    - Confusion Matrix
    - ROC Curve
    - Precision-Recall Curve
    - Feature Importance (optional)
    """
    if model is not None and x_features is not None:
        # Calculate predicted probabilities if the model is provided
        y_pred_proba = model.predict_proba(x_features)[:,1]
    else:
        raise ValueError("To plot ROC and Precision-Recall, you need to pass both model and x_features.")

    plot_confusion_matrix(y_true, y_pred, dataset_name, model_name)
    plot_roc_curve(y_true, y_pred_proba, dataset_name, model_name)
    plot_precision_recall_curve_custom(y_true, y_pred_proba, dataset_name, model_name)

    if model is not None:
        # Pass feature names to plot_feature_importance
        plot_feature_importance(model, feature_names=feature_names, dataset_name=dataset_name, model_folder=model_name)
