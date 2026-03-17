# -*- coding: utf-8 -*-
"""
Script para la evaluación de modelos de Machine Learning.
Carga los datos de test procesados y los modelos exportados (.pkl)
para calcular métricas avanzadas y la matriz de confusión.
"""

import json
import sys

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score


def format_confusion_matrix(y_test, y_pred):
    """
    Función para calcular la matriz de confusión
    """
    cm = confusion_matrix(y_test, y_pred)
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
            f"{'MEDIA ':<12} | {prec_macro:<10.4f} | {rec_macro:<10.4f} | {spec_macro:<12.4f} | {f1_macro:<10.4f}")


def evaluar_modelo(nombre_modelo, archivo_pkl, df_test, target_col):
    """
    Función genérica para cargar un modelo y evaluarlo con el set de test.
    """
    print(f"\n==========================================================")
    print(f" EVALUANDO MODELO: {nombre_modelo.upper()}")
    print(f"==========================================================")

    if not os.path.exists(archivo_pkl):
        print(f" [!] Archivo '{archivo_pkl}' no encontrado. Asegúrate de entrenar este modelo primero.")
        return

    # 1. Separar características (X) y objetivo (y) para TEST
    if target_col in df_test.columns:
        X_test = df_test.drop(columns=[target_col]).values
        y_test = df_test[target_col].values
    else:
        X_test = df_test.iloc[:, :-1].values
        y_test = df_test.iloc[:, -1].values

    # 2. Cargar el modelo
    modelo = joblib.load(archivo_pkl)
    print(f" -> Modelo cargado exitosamente desde: {archivo_pkl}")

    # 3. Predecir
    y_pred = modelo.predict(X_test)

    # 4. Mostrar Resultados
    print("\n[ Matriz de Confusión ]")
    print(format_confusion_matrix(y_test, y_pred))

    print("\n[ Métricas de Evaluación ]")
    print_advanced_metrics(y_test, y_pred)


if __name__ == "__main__":
    test_file = "test_listo.csv"
    config_file = "config.json"

    # Verificamos que existan los archivos necesarios
    if not os.path.exists(config_file):
        print(f"Error: No se encuentra {config_file}.")
        sys.exit(1)

    if not os.path.exists(test_file):
        print(f"Error: No se encuentra {test_file}. Ejecuta el preprocesamiento primero.")
        sys.exit(1)

    # Cargar la configuración para saber cuál es el target
    with open(config_file, 'r') as f:
        config = json.load(f)
    target_col = config["preproceso"]["target"]

    # Cargar los datos de Test
    df_test = pd.read_csv(test_file)
    print(f" -> Datos de Test cargados: {len(df_test)} filas.")

    # ==============================================
    # 1. EVALUAR KNN
    # ==============================================
    evaluar_modelo("K-Nearest Neighbors (KNN)", "mejor_modelo.pkl", df_test, target_col)

    # ==============================================
    # 2. EVALUAR DECISION TREES
    # ==============================================
    # evaluar_modelo("Decision Trees", "mejor_modelo_dt.pkl", df_test, target_col)

    # ==============================================
    # 3. EVALUAR RANDOM FOREST
    # ==============================================
    # evaluar_modelo("Random Forest", "mejor_modelo_rf.pkl", df_test, target_col)

    # ==============================================
    # 4. EVALUAR LOGISTIC REGRESSION
    # ==============================================
    # evaluar_modelo("Logistic Regression", "mejor_modelo_lr.pkl", df_test, target_col)

    # ==============================================
    # 5. EVALUAR NAÏVE BAYES
    # ==============================================
    # evaluar_modelo("Naïve Bayes", "mejor_modelo_nb.pkl", df_test, target_col)