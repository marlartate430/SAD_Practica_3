# -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo kNN con Grid Search
Recoge los datos preprocesados de Train y Test por separado y calcula métricas avanzadas por clase.
"""

import sys
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score


def load_data(file):
    data = pd.read_csv(file)
    return data


def load_config(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config


def format_confusion_matrix(y_test, y_pred):
    """
    Función para calcular la matriz de confusión y devolverla con etiquetas bonitas usando Pandas
    """
    cm = confusion_matrix(y_test, y_pred)

    # Obtenemos las etiquetas únicas presentes en test o pred para nombrar filas/columnas
    labels = sorted(list(set(y_test) | set(y_pred)))

    # Creamos un DataFrame de Pandas para que se imprima con nombres
    df_cm = pd.DataFrame(
        cm,
        index=[f"Real {lbl}" for lbl in labels],
        columns=[f"Pred {lbl}" for lbl in labels]
    )
    return df_cm


def print_advanced_metrics(y_test, y_pred):
    """
    Función para calcular e imprimir Accuracy Global, y Precision, Recall, Specificity y F-score por cada clase.
    Calcula la media Macro solo si hay más de 2 clases.
    """
    labels = sorted(list(set(y_test) | set(y_pred)))
    num_classes = len(labels)

    # El Accuracy es global (cuántos acertó de todos los que había)
    accuracy = accuracy_score(y_test, y_pred)
    print(f" -> Accuracy Global (Exactitud): {accuracy:.4f}\n")

    # Al poner average=None, sklearn nos devuelve un array con la nota de CADA clase
    precision_per_class = precision_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=labels, zero_division=0)

    # Cálculo manual de la Specificity (Especificidad) por cada clase
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    FP = cm.sum(axis=0) - np.diag(cm)  # Falsos Positivos
    FN = cm.sum(axis=1) - np.diag(cm)  # Falsos Negativos
    TP = np.diag(cm)  # Verdaderos Positivos
    TN = cm.sum() - (FP + FN + TP)  # Verdaderos Negativos

    # División segura para evitar errores de división por cero
    specificity_per_class = np.divide(TN, TN + FP, out=np.zeros_like(TN, dtype=float), where=(TN + FP) != 0)

    # Imprimir la tabla detallada
    print(f"{'Clase':<12} | {'Precision':<10} | {'Recall':<10} | {'Specificity':<12} | {'F1-Score':<10}")
    print("-" * 65)

    for i, label in enumerate(labels):
        print(
            f"{label:<12} | {precision_per_class[i]:<10.4f} | {recall_per_class[i]:<10.4f} | {specificity_per_class[i]:<12.4f} | {f1_per_class[i]:<10.4f}")

    # Si hay más de 2 clases (Multiclase), calculamos y mostramos las medias Macro
    if num_classes > 2:
        print("-" * 65)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        spec_macro = np.mean(specificity_per_class)

        print(
            f"{'MEDIA (Macro)':<12} | {prec_macro:<10.4f} | {rec_macro:<10.4f} | {spec_macro:<12.4f} | {f1_macro:<10.4f}")


def kNN_sweep(train_data, test_data, target_col, knn_config):
    # 1. Separar características (X) y objetivo (y) para TRAIN
    if target_col in train_data.columns:
        X_train = train_data.drop(columns=[target_col]).values
        y_train = train_data[target_col].values
    else:
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

    # 2. Separar características (X) y objetivo (y) para TEST
    if target_col in test_data.columns:
        X_test = test_data.drop(columns=[target_col]).values
        y_test = test_data[target_col].values
    else:
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

    print(f" -> Datos cargados: {len(X_train)} filas en Train, {len(X_test)} filas en Test.")

    # 3. Configuramos los parámetros a explorar desde el JSON
    param_grid = {
        'n_neighbors': knn_config.get("n_neighbors", [3, 5, 7]),
        'weights': knn_config.get("weights", ['uniform']),
        'p': knn_config.get("p", [2])
    }

    # 4. Entrenamos el modelo usando GridSearchCV
    print("\n -> Iniciando barrido de hiperparámetros (Grid Search)...")
    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f" -> ¡Barrido completado! Mejores hiperparámetros encontrados: {best_params}")

    # 5. Predecimos sobre el conjunto Test usando el mejor modelo
    y_pred = best_model.predict(X_test)

    return y_test, y_pred, best_params, best_model


if __name__ == "__main__":
    '''
    if len(sys.argv) < 2:
        print("Error en los parámetros de entrada")
        print("Uso: python knn.py <config.json>")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    config_file = sys.argv[3]
    '''
    train_file = "train_listo.csv"
    test_file = "test_listo.csv"
    config_file = "config.json"

    config = load_config(config_file)
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    target_col = config["preproceso"]["target"]
    knn_config = config.get("knn", {})

    print("=========================================")
    print("      ENTRENAMIENTO MODELO KNN           ")
    print("=========================================")

    y_test, y_pred, best_params, best_model = kNN_sweep(train_data, test_data, target_col, knn_config)

    print("\n================ RESULTADOS EN CONJUNTO TEST ================")
    print("\nMatriz de Confusión:")
    print(format_confusion_matrix(y_test, y_pred))

    print("\n================ MÉTRICAS DE EVALUACIÓN ================")
    print_advanced_metrics(y_test, y_pred)

    nombre_archivo = 'mejor_modelo_knn.pkl'
    joblib.dump(best_model, nombre_archivo)
    print(f"\n[+] El mejor modelo ha sido exportado exitosamente a: {nombre_archivo}")