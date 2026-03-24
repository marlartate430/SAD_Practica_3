# -*- coding: utf-8 -*-
import signal
import sys
import pandas as pd
import numpy as np
import json
import re
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
import tqdm
from sklearn.tree import DecisionTreeClassifier
import time
import random


# POR CAMBIAR
CV_POR_DEFECTO = 5
CPU_POR_DEFECTO = -1
STIMATOR_POR_DEFECTO = None


nltk.download('stopwords', quiet=True)


def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)


# ==========================================
# 1. FUNCIÓN PARA ELIMINAR DUPLICADOS
# ==========================================
def eliminar_duplicados(df_train, df_test, config):
    if config.get("drop_duplicates"):
        df_train = df_train.drop_duplicates()
        df_test = df_test.drop_duplicates()
        print(" -> Filas duplicadas eliminadas en Train y Test.")
    return df_train, df_test


# ==========================================
# 2. FUNCIÓN ELIMINAR LAS COLUMNAS ESPECIFICADAS
# ==========================================
def eliminar_columnas(df_train, df_test, config):
    columnas_a_borrar = config.get("drop_columns")
    if columnas_a_borrar:
        cols_train = [col for col in columnas_a_borrar if
                      col in df_train.columns]  # se asegura que las columnas estén en df
        df_train = df_train.drop(columns=cols_train)

        cols_test = [col for col in columnas_a_borrar if col in df_test.columns]
        df_test = df_test.drop(columns=cols_test)
        print(f" -> Columnas eliminadas: {columnas_a_borrar}")
    else:
        print(" -> No se especificaron columnas para eliminar.")
    return df_train, df_test


# ==========================================
# 3. FUNCIÓN PARA ASIGNAR TIPOS
# ==========================================
def asignar_tipos(df_train, df_test, tipos_json):
    for col, tipo in zip(df_train.columns, tipos_json):
        if tipo == "int":
            df_train[col], df_test[col] = df_train[col].astype('Int64'), df_test[col].astype('Int64')
        elif tipo == "double":
            df_train[col], df_test[col] = df_train[col].astype('float64'), df_test[col].astype('float64')
        elif tipo == "string":
            df_train[col], df_test[col] = df_train[col].astype('category'), df_test[col].astype('category')
        elif tipo == "text":
            df_train[col], df_test[col] = df_train[col].astype('string'), df_test[col].astype('string')
        elif tipo == "id":
            df_train[col], df_test[col] = df_train[col].astype('object'), df_test[col].astype('object')
    print(" -> Tipos de datos asignados según config.json.")
    return df_train, df_test


# ==========================================
# 4. FUNCIÓN PARA VALORES ERRÓNEOS
# ==========================================
def tratar_valores_erroneos(df_train, df_test, config):
    config_err = config.get("erroneous_values")
    if not config_err or config_err.get("action") == "none":
        return df_train, df_test

    accion_global = config_err.get("action")
    reglas = config_err.get("rules", {})  # diccionario con que hacer en cada columna

    def aplicar_errores(df, es_train=False, df_referencia=None):
        df_resultado = df.copy()
        for col, regla in reglas.items():  # itetrar cada columna
            if col not in df_resultado.columns:
                continue

            condiciones = regla.get("conditions", [])
            estrategia = regla.get("strategy", "none")

            mascara_errores = pd.Series(False,
                                        index=df_resultado.index)  # array de booleanos del mismo tamaño que los datos
            # marca a true si hay un valor erroneo

            for cond in condiciones:  # iterar cada condicion de la columna
                tipo = cond.get("type")
                valor = cond.get("value")

                # poner a true en la mascara los que cumplan la condicion
                if tipo == "less_than":
                    mascara_errores = mascara_errores | (df_resultado[col] < valor)
                elif tipo == "greater_than":
                    mascara_errores = mascara_errores | (df_resultado[col] > valor)
                elif tipo == "equals":
                    mascara_errores = mascara_errores | (df_resultado[col] == valor)
                elif tipo == "in_list":
                    mascara_errores = mascara_errores | (df_resultado[col].isin(valor))
                elif tipo == "regex":
                    mascara_errores = mascara_errores | df_resultado[col].astype(str).str.contains(valor, regex=True,
                                                                                                   na=False)
                elif tipo == "has_decimals":
                    if pd.api.types.is_numeric_dtype(df_resultado[col]):
                        if valor is True:
                            mascara_errores = mascara_errores | (df_resultado[col].notna() & (df_resultado[
                                                                                                  col] % 1 != 0))  # Csi el resto de dividir entre 1 no es 0, tiene decimales

            # arreglar las filas que estan a true en la mascara
            if mascara_errores.any():
                if accion_global == "delete":
                    df_resultado = df_resultado[~mascara_errores]
                elif accion_global == "impute" and estrategia != "none":
                    # APRENDEMOS SIEMPRE DEL TRAIN
                    fuente_datos = df_resultado if es_train else df_referencia

                    if estrategia == "mean" and pd.api.types.is_numeric_dtype(fuente_datos[col]):
                        valor_imputar = fuente_datos[col].mean()
                        if pd.api.types.is_integer_dtype(fuente_datos[col]):
                            valor_imputar = int(round(valor_imputar))  # Redondear si es entero
                        df_resultado.loc[mascara_errores, col] = valor_imputar

                    elif estrategia == "median" and pd.api.types.is_numeric_dtype(fuente_datos[col]):
                        valor_imputar = fuente_datos[col].median()
                        if pd.api.types.is_integer_dtype(fuente_datos[col]):
                            valor_imputar = int(round(valor_imputar))  # Redondear si es entero
                        df_resultado.loc[mascara_errores, col] = valor_imputar

                    elif estrategia == "mode":
                        if not fuente_datos[col].mode().empty:
                            df_resultado.loc[mascara_errores, col] = fuente_datos[col].mode()[0]
        return df_resultado

    # Aplicamos primero a Train (aprendiendo de sí mismo)
    df_train = aplicar_errores(df_train, es_train=True)
    # Aplicamos a Test (aprendiendo del Train)
    df_test = aplicar_errores(df_test, es_train=False, df_referencia=df_train)

    print(" -> Valores erróneos tratados.")
    return df_train, df_test


# ==========================================
# 5. FUNCIÓN PARA NULOS
# ==========================================
def tratar_nulos(df_train, df_test, config):
    accion = config.get("missing_values")
    if accion == "delete":
        print(" -> Filas con nulos eliminadas.")
        return df_train.dropna(), df_test.dropna()
    elif accion == "impute":
        lista_estrategia = config.get("impute_strategy")
        lista_tipos = config.get("categoria")

        for col, tipo, estrategia in zip(df_train.columns, lista_tipos, lista_estrategia):
            if tipo != "text" and tipo != "object" and estrategia != "none":

                # Aprender el valor de imputación del TRAIN
                valor_imputar = None
                if estrategia == "mode" and not df_train[col].mode().empty:
                    valor_imputar = df_train[col].mode()[0]
                elif estrategia == "mean" and pd.api.types.is_numeric_dtype(df_train[col]):
                    valor_imputar = df_train[col].mean()
                    if pd.api.types.is_integer_dtype(df_train[col]):
                        valor_imputar = int(round(valor_imputar))  # Redondear si es entero
                elif estrategia == "median" and pd.api.types.is_numeric_dtype(df_train[col]):
                    valor_imputar = df_train[col].median()
                    if pd.api.types.is_integer_dtype(df_train[col]):
                        valor_imputar = int(round(valor_imputar))  # Redondear si es entero
                elif estrategia == "max":
                    valor_imputar = df_train[col].max()
                elif estrategia == "min":
                    valor_imputar = df_train[col].min()

                # Aplicar a AMBOS si hay un valor que imputar
                if valor_imputar is not None:
                    df_train[col] = df_train[col].fillna(valor_imputar)
                    df_test[col] = df_test[col].fillna(valor_imputar)

        print(" -> Valores nulos imputados según estrategia (Valores extraídos del Train).")
    return df_train, df_test


# ==========================================
# 6. FUNCIÓN PARA OUTLIERS
# ==========================================
def tratar_outliers(df_train, df_test, config):
    accion = config.get("outliers")
    target = config.get("target")

    lista_estrategia = config.get("outlier_strategy")
    cols_numericas = df_train.select_dtypes(include=[np.number]).columns  # obtener las columnas de numeros
    if target in cols_numericas:
        cols_numericas = cols_numericas.drop(target)

    for col, estrategia in zip(cols_numericas, lista_estrategia):
        if estrategia != "none":
            # Métod del Rango Intercuartílico (IQR): va a meter los outliers en el rango de 1.5*el rango del 50% de la mediana
            Q1 = df_train[col].quantile(0.25)
            Q3 = df_train[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inf = Q1 - 1.5 * IQR
            limite_sup = Q3 + 1.5 * IQR

            # Si la columna es entera, redondeamos
            if pd.api.types.is_integer_dtype(df_train[col]):
                limite_inf = int(round(limite_inf))
                limite_sup = int(round(limite_sup))

            es_outlier_train = (df_train[col] < limite_inf) | (
                    df_train[col] > limite_sup)  # definir que es un valor Outlier (usando IQR)
            es_outlier_test = (df_test[col] < limite_inf) | (df_test[col] > limite_sup)

            if accion == "delete":
                df_train = df_train[~es_outlier_train]  # Solo dejas no Outliers (la ~ es como un NOT)
                df_test = df_test[~es_outlier_test]
                print(f" -> Outliers tratados de columna '{col}' usando la acción: {accion}.")
            elif accion == "impute":
                if estrategia == "max-min":
                    df_train[col] = df_train[col].clip(lower=limite_inf, upper=limite_sup)
                    df_test[col] = df_test[col].clip(lower=limite_inf, upper=limite_sup)
                else:
                    valor_imputar = df_train[col].median() if estrategia == "median" else df_train[col].mean()
                    if pd.api.types.is_integer_dtype(df_train[col]):
                        valor_imputar = int(round(valor_imputar))

                    df_train.loc[es_outlier_train, col] = valor_imputar
                    df_test.loc[es_outlier_test, col] = valor_imputar
                print(f" -> Outliers tratados de columna '{col}' usando la acción: {accion} con '{estrategia}'.")
    return df_train, df_test


# ==========================================
# 7. FUNCIÓN PARA ESCALADO
# ==========================================
def escalar_datos(df_train, df_test, config):
    lista_estrategia = config.get("scaling")
    target = config.get("target")

    cols_numericas = df_train.select_dtypes(include=[np.number]).columns  # obtener las columnas de numeros
    if target in cols_numericas:
        cols_numericas = cols_numericas.drop(target)

    for col, estrategia in zip(cols_numericas, lista_estrategia):
        if estrategia == "z-score":
            train_mean = df_train[col].mean()
            train_std = df_train[col].std()
            df_train[col] = (df_train[col] - train_mean) / train_std
            df_test[col] = (df_test[col] - train_mean) / train_std
        elif estrategia == "min-max":
            train_min = df_train[col].min()
            train_max = df_train[col].max()
            rango = train_max - train_min if (train_max - train_min) != 0 else 1
            df_train[col] = (df_train[col] - train_min) / rango
            df_test[col] = (df_test[col] - train_min) / rango
        print(f" -> Columna '{col}' escalada con {estrategia}.")
    return df_train, df_test


# ==========================================
# 8. FUNCIONES DE LIMPIEZA DE TEXTO
# ==========================================
def limpiar_y_normalizar_texto(df_train, df_test, config):
    stop_words = set(
        stopwords.words('english'))  # Una lista con todas las palabras basura (the, a, is, in, at, ...)
    stemmer = SnowballStemmer('english')  # Algoritmo que convierte las palabras en su raiz
    cols_texto = df_train.select_dtypes(
        include=['string']).columns  # Obtener las columnas que hemos definido como string (texto)

    def procesar_celda(texto):
        if pd.isna(texto): return texto  # Vacia
        if config.get("text_process") == "basic_clean":
            texto = str(texto).lower().strip()
            # .lower(): convierte a minusculas
            # .strip(): quita espacios de más que puede haber
        if config.get("normalize_strategy") == "basic":
            texto = re.sub(r'[^\w\s]', '', str(texto))  # Quita todos los signos de puntuación
            # \w: cualquier letra
            # \s: espacio en blanco
            # ^: inversa
            # re.sub(): sustituye lo que le digas(los signos de puntuación por vacios) en un texto.
            palabras = texto.split()  # Separa por palabras
            palabras_limpias = [stemmer.stem(p) for p in palabras if
                                p not in stop_words]  # Quita las basura y pone la raiz de las otras
            texto = " ".join(palabras_limpias)  # devolvemos la frase con espacio entre las palabras
        return texto

    for col in cols_texto:
        df_train[col] = df_train[col].apply(procesar_celda)
        df_test[col] = df_test[col].apply(procesar_celda)
    print(" -> Texto limpiado y normalizado (Stopwords eliminadas y aplicado Stemming) en ambos sets.")
    return df_train, df_test


# ==========================================
# 9. FUNCIÓN PARA VECTORIZAR TEXTO
# ==========================================
def vectorizar_texto(df_train, df_test, config):
    estrategia = config.get("text_encoding")
    target = config.get("target")
    if estrategia == "none": return df_train, df_test

    cols_texto = df_train.select_dtypes(include=['string']).columns
    for col in cols_texto:
        if col == target: continue  # Si la columna es el target, la saltamos

        # METODOS DE sklearn
        if estrategia == "one-hot":
            vec = CountVectorizer(binary=True)
        elif estrategia == "frequency":
            vec = CountVectorizer()
        elif estrategia == "tf-idf":
            vec = TfidfVectorizer()

        textos_train = df_train[col].fillna('')  # Rellena huecos vacios que pueden haber quedado
        textos_test = df_test[col].fillna('')

        # ENTRENAMOS (Fit) SOLO CON TRAIN, APLICAMOS (Transform) A AMBOS
        matriz_train = vec.fit_transform(textos_train)  # ejecuta la estrategia de vectorización
        # Crea un diccionario con todas las palabras de los datos y da los valores a cada fila
        matriz_test = vec.transform(textos_test)

        nombres_cols = [f"{col}_{palabra}" for palabra in
                        vec.get_feature_names_out()]  # Darle nombres a las columnas (comentario_yo, comentario_comer, comentario_hacer, ...)

        df_vec_train = pd.DataFrame(matriz_train.toarray(), columns=nombres_cols,
                                    index=df_train.index)  # juntar los nombres con los datos vectorizados
        # matriz.toarray(): los datos vectorizados
        # columns=nombres_cols: los nombres que acabamos de crear
        # index=df.index: el index (los id) de las filas. Pandas le da un valor numerico a cada fila (un id), como igual hemos borrado filas le decimos que siga ese indice
        df_vec_test = pd.DataFrame(matriz_test.toarray(), columns=nombres_cols, index=df_test.index)

        df_train = pd.concat([df_train, df_vec_train], axis=1).drop(col,
                                                                    axis=1)  # Unir todas las columnas y Borrar la columna de texto
        df_test = pd.concat([df_test, df_vec_test], axis=1).drop(col, axis=1)
        print(f" -> Columna '{col}' vectorizada con {estrategia}.")

    return df_train, df_test


# ==========================================
# 10. FUNCIÓN PARA CODIFICAR EL TARGET
# ==========================================
def codificar_objetivo(df_train, df_test, config):
    target = config.get("target")

    if target and target in df_train.columns:
        if not pd.api.types.is_numeric_dtype(
                df_train[target]):  # Si es un problema de clasificacion, lo pasamos a números
            le = LabelEncoder()
            # Ajustamos (Fit) con Train y aplicamos a ambos
            df_train[target] = le.fit_transform(df_train[target])

            # Para Test, manejamos posibles etiquetas nuevas no vistas en Train
            df_test[target] = df_test[target].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            print(f" -> Variable objetivo '{target}' codificada con LabelEncoder.")

    return df_train, df_test


# ==========================================
# 11. FUNCIÓN PARA CODIFICAR CATEGORÍAS
# ==========================================
def codificar_categoricas(df_train, df_test, config):
    estrategia = config.get("categorical_encoding")
    target = config.get("target")  # Escudo
    if estrategia == "none": return df_train, df_test

    cols_categoricas = df_train.select_dtypes(
        include=['category']).columns  # Conseguir las columnas que anets hemos definido como "Category" (los string)
    cols_categoricas = [c for c in cols_categoricas if c != target]  # Quitamos el target para no romperlo
    if not cols_categoricas: return df_train, df_test

    if estrategia == "one-hot":
        df_train = pd.get_dummies(df_train, columns=cols_categoricas, drop_first=True, dtype=int)
        df_test = pd.get_dummies(df_test, columns=cols_categoricas, drop_first=True, dtype=int)
        # ALINEACIÓN MÁGICA DE PANDAS: Asegura que el Test tenga exactamente las mismas columnas que el Train
        df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

    elif estrategia == "label":
        for col in cols_categoricas:
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col])
            df_test[col] = df_test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    print(f" -> Variables categóricas codificadas usando: {estrategia}.")
    return df_train, df_test


# ==========================================
# 12. FUNCIÓN PARA BALANCEAR LOS DATOS
# ==========================================
def balancear_datos(df_train, df_test, config):
    estrategia = config.get("sampling_strategy")
    target = config.get("target")
    ratio = config.get("sampling_ratio")
    seed = config.get("sampling_seed", 42)

    if estrategia == "none" or not target or target not in df_train.columns:
        return df_train, df_test

    # ¡REGLA DE ORO! Solo tocamos el Train. El Test representa el mundo real.
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]

    try:
        if estrategia == "oversample":
            # Le pasamos el ratio al inicializar la herramienta
            ros = RandomOverSampler(sampling_strategy=ratio, random_state=seed)
            X_res, y_res = ros.fit_resample(X_train, y_train)
        elif estrategia == "undersample":
            rus = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
            X_res, y_res = rus.fit_resample(X_train, y_train)
        else:
            return df_train, df_test

        df_train_res = pd.concat([X_res, y_res], axis=1)
        print(f" -> Filas antes: {len(df_train)} | Filas después: {len(df_train_res)}")
        return df_train_res, df_test

    except Exception as e:
        print(f" ⚠️ Error al balancear: {e}")
        return df_train, df_test


# ==========================================
# PIPELINE PRINCIPAL DE PREPROCESADO
# ==========================================
def pipeline_preprocesamiento(json_path):
    with open(json_path, 'r') as f:
        config_completo = json.load(f)
        config = config_completo.get("preproceso")
    df_train = pd.read_csv(config_completo.get("train"))
    df_test = pd.read_csv(config_completo.get("test"))

    print("\n[1/12] Comprobando duplicados...")
    df_train, df_test = eliminar_duplicados(df_train, df_test, config)

    print("\n[2/12] Limpiando columnas innecesarias...")
    df_train, df_test = eliminar_columnas(df_train, df_test, config)

    print("\n[3/12] Asignando tipos de datos...")
    df_train, df_test = asignar_tipos(df_train, df_test, config.get("categoria", []))

    print("\n[4/12] Tratando valores erróneos...")
    df_train, df_test = tratar_valores_erroneos(df_train, df_test, config)

    print("\n[5/12] Tratando valores nulos...")
    df_train, df_test = tratar_nulos(df_train, df_test, config)

    print("\n[6/12] Tratando outliers...")
    df_train, df_test = tratar_outliers(df_train, df_test, config)

    print("\n[7/12] Limpiando y normalizando texto...")
    df_train, df_test = limpiar_y_normalizar_texto(df_train, df_test, config)

    print("\n[8/12] Vectorizando texto (NLP)...")
    df_train, df_test = vectorizar_texto(df_train, df_test, config)

    print("\n[9/12] Codificando la variable objetivo (Target)...")
    df_train, df_test = codificar_objetivo(df_train, df_test, config)

    print("\n[10/12] Codificando variables categóricas...")
    df_train, df_test = codificar_categoricas(df_train, df_test, config)

    print("\n[11/12] Balanceando clases...")
    df_train, df_test = balancear_datos(df_train, df_test, config)

    df_train_unscaled = df_train.copy()
    df_test_unscaled = df_test.copy()

    print("\n[12/12] Escalando datos numéricos...")
    df_train, df_test = escalar_datos(df_train, df_test, config)

    print(
        f"\n--- Finalizado. DataFrame resultante: Train {df_train.shape[0]} filas, Test {df_test.shape[0]} filas, {df_train.shape[1]} columnas ---")

    return df_train_unscaled, df_test_unscaled, df_train, df_test, config_completo


# ==========================================
# ENTRENAMIENTO KNN
# ==========================================
def kNN_sweep(train_data, target_col, knn_config):
    # 1. Separar características (X) y objetivo (y) SOLAMENTE para TRAIN
    if target_col in train_data.columns:
        X_train = train_data.drop(columns=[target_col]).values
        y_train = train_data[target_col].values
    else:
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

    print(f" -> Datos cargados para KNN: {len(X_train)} filas en Train.")

    # 3. Configuramos los parámetros a explorar desde el JSON
    param_grid = {
        'n_neighbors': knn_config.get("n_neighbors", [3, 5, 7, 9, 11]),
        'weights': knn_config.get("weights", ["uniform", "distance"]),
        'p': knn_config.get("p", [1, 2])
    }
    scoring = knn_config.get("scoring", "f1_macro")

    use_k_fold = knn_config.get("use_k_fold", True)  # Por defecto hacemos K-Folds

    if use_k_fold:
        cv_folds = knn_config.get("cv_folds", 5)  # 5 particiones por defecto
        cv_strategy = cv_folds
        print(f"\n -> Iniciando barrido de hiperparámetros (K-Folds con {cv_folds} particiones)...")
    else:
        test_size = knn_config.get("test_size", 0.2)
        cv_strategy = ShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        print(f"\n -> Iniciando barrido de hiperparámetros (1 solo split con {test_size * 100}% para Dev)...")

    # 4. Entrenamos el modelo usando GridSearchCV
    knn = KNeighborsClassifier()

    # Le pasamos nuestra variable cv_strategy, que se ha adaptado según el JSON
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=cv_strategy, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_  # Extraemos la puntuación para comparar después

    print(f" -> ¡Barrido completado! Mejores hiperparámetros encontrados: {best_params} (Score: {best_score:.4f})")

    return best_params, best_model, best_score


# ==========================================
# ENTRENAMIENTO DECISION TREE
# ==========================================
def dt_sweep(df_pro, target_col, config):
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """

    # No habria que pasar este bloque de codigo antes de las llamadas a los modelos?
    # 1. Separar características (X) y objetivo (y) SOLAMENTE para TRAIN

    x_train = None
    y_train = None

    if target_col in df_pro.columns:
        x_train = df_pro.drop(columns=[target_col]).values
        y_train = df_pro[target_col].values
    else:
        x_train = df_pro.iloc[:, :-1].values
        y_train = df_pro.iloc[:, -1].values

    parametros_decision_tree = []
    claves = config.keys()

    for indice, profundidad in enumerate(config.get("max_depth", [])):
        diccionario_actual = {}
        for clave in claves:
            if clave == "min_samples_split" or clave == "min_samples_leaf":
                diccionario_actual[clave] = range(1, config[clave] + 1)
            else:
                diccionario_actual[clave] = [config[clave][indice]]

        parametros_decision_tree.append(diccionario_actual)

    print(parametros_decision_tree)

    # El enunciado dice que hay que usar 1 y 2, pero sklearn da advertencias

    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:
        # TODO Llamar al decision trees
        gs = GridSearchCV(
            DecisionTreeClassifier(),
            parametros_decision_tree,
            cv=CV_POR_DEFECTO,
            n_jobs=CPU_POR_DEFECTO,
            scoring=STIMATOR_POR_DEFECTO
        )

        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random() * 2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)

    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + "segundos")

    best_params = gs.best_params_
    best_model = gs.best_estimator_
    best_score = gs.best_score_


    # Mostramos los resultados
    # mostrar_resultados(gs, x_dev, y_dev)

    # Guardamos el modelo utilizando pickle
    # save_model(gs)

    return best_params, best_model, best_score



# ==========================================
# ENTRENAMIENTO NAÏVE BAYES
# ==========================================
def nb_sweep(train_data, target_col, nb_config):
    # 1. Separar características (X) y objetivo (y) SOLAMENTE para TRAIN
    if target_col in train_data.columns:
        X_train = train_data.drop(columns=[target_col]).values
        y_train = train_data[target_col].values
    else:
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

    print(f" -> Datos cargados para Naïve Bayes: {len(X_train)} filas en Train.")

    tipo_modelo = nb_config.get("model_type", "gaussian")

    if tipo_modelo == "multinomial":
        print(" -> Instanciando MultinomialNB (Ideal para conteos discretos / NLP).")
        nb = MultinomialNB()
        param_grid = {
            'alpha': nb_config.get("alpha", [0.01, 0.1, 0.5, 1.0, 2.0])
        }
    elif tipo_modelo == "categorical":
        print(" -> Instanciando CategoricalNB (Ideal para características categóricas discretas).")
        nb = CategoricalNB()
        param_grid = {
            'alpha': nb_config.get("alpha", [0.01, 0.1, 0.5, 1.0, 2.0])
        }
    else:
        print(" -> Instanciando GaussianNB (Ideal para variables continuas o mixtas).")
        nb = GaussianNB()
        param_grid = {
            'var_smoothing': nb_config.get("var_smoothing", [1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
        }

    scoring = nb_config.get("scoring", "f1_macro")
    use_k_fold = nb_config.get("use_k_fold", True)

    if use_k_fold:
        cv_folds = nb_config.get("cv_folds", 5)
        cv_strategy = cv_folds
        print(f"\n -> Iniciando barrido de hiperparámetros (K-Folds con {cv_folds} particiones)...")
    else:
        test_size = nb_config.get("test_size", 0.2)
        cv_strategy = ShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        print(f"\n -> Iniciando barrido de hiperparámetros (1 solo split con {test_size * 100}% para Dev)...")

    # 3. Entrenamos el modelo usando GridSearchCV
    grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=cv_strategy, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_

    print(f" -> ¡Barrido completado! Mejores hiperparámetros encontrados: {best_params} (Score: {best_score:.4f})")

    return best_params, best_model, best_score


# ==========================================
# FUNCIÓN PARA SELECCIONAR Y EXPORTAR EL MEJOR MODELO
# ==========================================
def evaluar_y_seleccionar_mejor_modelo(modelos_entrenados):
    """
    Función que recibe todos los modelos generados y sus métricas de validación,
    los compara, anuncia el ganador imprimiendo sus parámetros y lo exporta.
    """
    print("\n=========================================")
    print("      RESULTADOS FINALES Y EXPORTACIÓN   ")
    print("=========================================")

    if not modelos_entrenados:
        print("[!] No se ha entrenado ningún modelo para comparar.")
        return

    # Buscamos el nombre del algoritmo que tenga el mayor "score" en el diccionario
    mejor_algoritmo = max(modelos_entrenados, key=lambda k: modelos_entrenados[k]["score"])

    mejor_modelo_global = modelos_entrenados[mejor_algoritmo]["modelo"]
    mejor_score_global = modelos_entrenados[mejor_algoritmo]["score"]
    mejores_params_globales = modelos_entrenados[mejor_algoritmo]["params"]

    print(" -> Evaluando contendientes (basado en Cross-Validation de entrenamiento)...")
    for algo, datos in modelos_entrenados.items():
        print(f"    - {algo}: Score {datos['score']:.4f}")

    print(f"\n[🏆] EL GANADOR ABSOLUTO ES: {mejor_algoritmo}")
    print(f"     -> Puntuación (F1 Macro CV): {mejor_score_global:.4f}")
    print(f"     -> Parámetros Óptimos: {mejores_params_globales}")

    # Exportamos solo el mejor modelo
    nombre_archivo = 'mejor_modelo.pkl'
    joblib.dump(mejor_modelo_global, nombre_archivo)
    print(f"\n[+] El modelo campeón ha sido exportado exitosamente a: {nombre_archivo}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # interrupcion CTR + C

    # 1. EJECUCIÓN DEL PIPELINE DE PREPROCESADO
    # Ahora devuelve también los DataFrames sin escalar
    df_train_unscaled, df_test_unscaled, df_train_proc, df_test_proc, config_completo = pipeline_preprocesamiento(
        'config.json')

    target_col = config_completo["preproceso"]["target"]

    # Extraemos la lista de algoritmos a usar (por defecto si no la encuentra hace KNN y Naive Bayes)
    algoritmos_elegidos = config_completo.get("algoritmos_a_usar", ["knn", "naive_bayes"])

    # Guardamos los escalados por defecto en el CSV
    df_train_proc.to_csv('train_listo.csv', index=False)
    df_test_proc.to_csv('test_listo.csv', index=False)
    print("\n¡Archivos preprocesados guardados con éxito!")

    print("\n=========================================")
    print("      ENTRENAMIENTO DE MODELOS           ")
    print("=========================================")

    modelos_entrenados = {}

    # ==============================================
    # 1. K-NEAREST NEIGHBORS (KNN)
    # ==============================================
    if "knn" in algoritmos_elegidos:
        print("\n--- 1. ENTRENANDO KNN ---")
        knn_config = config_completo.get("knn", {})
        # KNN usa los datos ESCALADOS (df_train_proc)
        best_params_knn, best_model_knn, best_score_knn = kNN_sweep(df_train_proc, target_col, knn_config)

        modelos_entrenados["KNN"] = {
            "modelo": best_model_knn,
            "score": best_score_knn,
            "params": best_params_knn
        }

    # ==============================================
    # 2. DECISION TREES (ÁRBOLES DE DECISIÓN)
    # ==============================================
    if "random_forest" in algoritmos_elegidos:
        print("\n--- 2. ENTRENANDO DECISION TREES ---")
        # TODO: Añadir lógica y GridSearch para Decision Trees.
        # Recuerda: Este modelo no necesita datos escalados.
        dt_config = config_completo.get("arbol_decision", {})
        best_params_dt, best_model_dt, best_score_dt = dt_sweep(df_train_proc, target_col, dt_config)
        modelos_entrenados["Decision Trees"] = {
            "modelo": best_model_dt,
            "score": best_score_dt,
            "params": best_params_dt
        }

    # ==============================================
    # 3. RANDOM FOREST
    # ==============================================
    if "random_forest" in algoritmos_elegidos:
        print("\n--- 3. ENTRENANDO RANDOM FOREST ---")
        # TODO: Añadir lógica y GridSearch para Random Forest.
        # Recuerda: Al igual que los árboles de decisión, es insensible al escalado.
        # best_params_rf, best_model_rf, best_score_rf = rf_sweep(...)
        # modelos_entrenados["Random Forest"] = {
        #     "modelo": best_model_rf, "score": best_score_rf, "params": best_params_rf
        # }

    # ==============================================
    # 4. LOGISTIC REGRESSION (REGRESIÓN LOGÍSTICA)
    # ==============================================
    if "logistic_regression" in algoritmos_elegidos:
        print("\n--- 4. ENTRENANDO LOGISTIC REGRESSION ---")
        # TODO: Añadir lógica y GridSearch para Logistic Regression.
        # Recuerda: Modelo sensible a la escala (como el KNN), el uso de df_train_proc escalado es ideal.
        # best_params_lr, best_model_lr, best_score_lr = lr_sweep(...)
        # modelos_entrenados["Logistic Regression"] = {
        #     "modelo": best_model_lr, "score": best_score_lr, "params": best_params_lr
        # }

    # ==============================================
    # 5. NAÏVE BAYES
    # ==============================================
    if "naive_bayes" in algoritmos_elegidos:
        print("\n--- 5. ENTRENANDO NAÏVE BAYES ---")
        # TODO: Añadir lógica para Naïve Bayes.
        # Recuerda: Usualmente se prueba con GaussianNB, MultinomialNB (si los datos no son negativos), etc.
        nb_config = config_completo.get("naive_bayes", {})
        # Usamos df_train_unscaled porque Naïve Bayes NO requiere escalado
        best_params_nb, best_model_nb, best_score_nb = nb_sweep(df_train_unscaled, target_col, nb_config)

        modelos_entrenados["Naïve Bayes"] = {
            "modelo": best_model_nb,
            "score": best_score_nb,
            "params": best_params_nb
        }

    # ==============================================
    # SELECCIÓN Y EXPORTACIÓN DEL MEJOR MODELO GLOBAL
    # ==============================================
    # Pasamos TODOS los modelos entrenados a la función final para que los evalúe y elija
    evaluar_y_seleccionar_mejor_modelo(modelos_entrenados)

    print("\n[!] Fin del pipeline.")