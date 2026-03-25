# -*- coding: utf-8 -*-
"""
Script para la evaluación de modelos de Machine Learning.
Carga los datos de test procesados y los modelos exportados (.pkl)
para calcular métricas avanzadas, la matriz de confusión y exportar las predicciones.
"""

import json
import sys
import pandas as pd
import numpy as np
import joblib
import os
import glob
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
    Calcula las medias Macro y Micro.
    """
    labels = sorted(list(set(y_test) | set(y_pred)))

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

    print("-" * 65)

    # Calcular y mostrar las medias Macro y Micro
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    spec_macro = np.mean(specificity_per_class)

    # Micro F1, Precision y Recall son matemáticamente iguales al Accuracy en clasificación multiclase
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

    print(f"{'MEDIA MACRO':<12} | {prec_macro:<10.4f} | {rec_macro:<10.4f} | {spec_macro:<12.4f} | {f1_macro:<10.4f}")
    print(f"{'MEDIA MICRO':<12} | {'-':<10} | {'-':<10} | {'-':<12} | {f1_micro:<10.4f}")


def evaluar_modelo(archivo_pkl, df_test, target_col):
    """
    Función genérica para cargar un modelo, evaluarlo con el set de test
    y devolver las predicciones generadas.
    """
    nombre_modelo = os.path.basename(archivo_pkl).replace(".pkl", "")

    print(f"\n==========================================================")
    print(f" EVALUANDO MODELO: {nombre_modelo.upper()}")
    print(f"==========================================================")

    # 1. Separar características (X) y objetivo (y) para TEST
    if target_col in df_test.columns:
        X_test = df_test.drop(columns=[target_col]).values
        y_test = df_test[target_col].values
    else:
        X_test = df_test.iloc[:, :-1].values
        y_test = df_test.iloc[:, -1].values

    # 2. Cargar el modelo
    try:
        modelo = joblib.load(archivo_pkl)
        print(f" -> Modelo cargado exitosamente desde: {archivo_pkl}")
    except Exception as e:
        print(f" [!] Error al cargar el modelo '{archivo_pkl}': {e}")
        return None

    # 3. Predecir
    y_pred = modelo.predict(X_test)

    # 4. Mostrar Resultados
    print("\n[ Matriz de Confusión ]")
    print(format_confusion_matrix(y_test, y_pred))

    print("\n[ Métricas de Evaluación ]")
    print_advanced_metrics(y_test, y_pred)

    return y_pred


if __name__ == "__main__":
    test_file = "test_listo.csv"
    config_file = "config.json"
    archivo_predicciones = "predicciones_modelos.csv"

    # Cargar la configuración para saber cuál es el target
    with open(config_file, 'r') as f:
        config = json.load(f)
    target_col = config["preproceso"]["target"]

    # Cargar los datos de Test
    df_test = pd.read_csv(test_file)
    print(f" -> Datos de Test cargados: {len(df_test)} filas.")

    # Preparamos un DataFrame para guardar las predicciones
    # Incluimos la columna Real (target) como primera columna
    df_predicciones = pd.DataFrame()

    if target_col in df_test.columns:
        df_predicciones[f"Real_{target_col}"] = df_test[target_col]
    else:
        df_predicciones[f"Real_{target_col}"] = df_test.iloc[:, -1]

    # Buscar todos los archivos .pkl en el directorio actual
    modelos_pkl = ["mejor_Decision Trees.pkl","mejor_KNN.pkl", "mejor_modelo.pkl", "mejor_Naïve Bayes.pkl", "mejor_Random Forest.pkl"]

    if not modelos_pkl:
        print("\n [!] No se han encontrado archivos de modelos (.pkl) en el directorio.")
        sys.exit(0)
    print(f"\n -> Se han encontrado {len(modelos_pkl)} modelos para evaluar.")

    # Iterar sobre cada modelo encontrado y evaluarlo
    for archivo_modelo in modelos_pkl:
        y_pred = evaluar_modelo(archivo_modelo, df_test, target_col)

        # Si la evaluación fue exitosa, guardamos las predicciones
        if y_pred is not None:
            nombre_columna = os.path.basename(archivo_modelo).replace(".pkl", "")
            df_predicciones[f"Pred_{nombre_columna}"] = y_pred

    # Guardar todas las predicciones en un archivo CSV
    df_predicciones.to_csv(archivo_predicciones, index=False)
    print(f"\n==========================================================")
    print(f" [✓] Todas las predicciones se han guardado en: {archivo_predicciones}")
    print(f"==========================================================")