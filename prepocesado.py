import signal
import sys
import pandas as pd
import numpy as np
import json
import re

# Librerías de Machine Learning y NLP
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import LabelEncoder

# Descargar diccionarios de NLTK en inglés de fondo
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
def eliminar_duplicados(df, config):
    if config.get("drop_duplicates"):
        antes = len(df)
        df = df.drop_duplicates()
        despues = len(df)
        if antes != despues:
            print(f" -> ¡Se han eliminado {antes - despues} filas duplicadas!")
        else:
            print(" -> No se encontraron filas duplicadas.")
    return df


# ==========================================
# 2. FUNCIÓN ELIMINAR LAS COLUMNAS ESPECIFICADAS
# ==========================================
def eliminar_columnas(df, config):
    columnas_a_borrar = config.get("drop_columns")
    if columnas_a_borrar:
        cols = [col for col in columnas_a_borrar if col in df.columns]  # se asegura que las columnas estén en df
        df = df.drop(columns=cols)
        print(f" -> Columnas eliminadas: {cols}")
    else:
        print(" -> No se especificaron columnas para eliminar.")
    return df


# ==========================================
# 3. FUNCIÓN PARA ASIGNAR TIPOS
# ==========================================
def asignar_tipos(df, tipos_json):
    for col, tipo in zip(df.columns, tipos_json):
        if tipo == "int":
            df[col] = df[col].astype('Int64')
        elif tipo == "double":
            df[col] = df[col].astype('float64')
        elif tipo == "string":
            df[col] = df[col].astype('category')
        elif tipo == "text":
            df[col] = df[col].astype('string')
        elif tipo == "id":
            df[col] = df[col].astype('object')
    print(" -> Tipos de datos asignados según config.json.")
    return df


# ==========================================
# 4. FUNCIÓN PARA NULOS
# ==========================================
def tratar_nulos(df, config):
    accion = config.get("missing_values")
    if accion == "delete":
        print(" -> Filas con nulos eliminadas.")
        return df.dropna()
    elif accion == "impute":
        lista_estrategia = config.get("impute_strategy")
        lista_tipos = config.get("categoria")

        for col, tipo, estrategia in zip(df.columns, lista_tipos, lista_estrategia):
            if tipo != "text":
                if df[col].isnull().any():
                    if estrategia == "mode" and not df[col].mode().empty:
                        df[col] = df[col].fillna(df[col].mode()[0])
                    elif estrategia == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                    elif estrategia == "median" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    elif estrategia == "max" and df[col].mode().empty:
                        df[col] = df[col].fillna(df[col].max())
                    elif estrategia == "min" and df[col].mode().empty:
                        df[col] = df[col].fillna(df[col].min())
        print(" -> Valores nulos imputados según estrategia.")
    return df


# ==========================================
# 5. FUNCIÓN PARA OUTLIERS
# ==========================================
def tratar_outliers(df, config):
    accion = config.get("outliers")
    target = config.get("target")

    lista_estrategia = config.get("outlier_strategy")
    cols_numericas = df.select_dtypes(include=[np.number]).columns  # obtener las columnas de numeros
    if target in cols_numericas:
        cols_numericas = cols_numericas.drop(target)

    for col, estrategia in zip(cols_numericas, lista_estrategia):
        if estrategia != "none":
            # Métod del Rango Intercuartílico (IQR): va a meter los outliers en el rango de 1.5*el rango del 50% de la mediana
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inf = Q1 - 1.5 * IQR
            limite_sup = Q3 + 1.5 * IQR
            es_outlier = (df[col] < limite_inf) | (df[col] > limite_sup)  # definir que es un valor Outlier (usando IQR)

            if accion == "delete":
                df = df[~es_outlier]  # Solo dejas no Outliers (la ~ es como un NOT)
                print(f" -> Outliers tratados de columna '{col}' usando la acción: {accion}.")
            elif accion == "impute" and es_outlier.any():
                if estrategia == "max-min":
                    df[col] = df[col].clip(lower=limite_inf, upper=limite_sup)
                elif estrategia == "median":
                    df.loc[es_outlier, col] = df[col].median()
                elif estrategia == "mean":
                    df.loc[es_outlier, col] = round(df[col].mean())
                print(f" -> Outliers tratados de columna '{col}' usando la acción: {accion} con '{estrategia}'.")
    return df


# ==========================================
# 6. FUNCIÓN PARA ESCALADO
# ==========================================
def escalar_datos(df, config):
    lista_estrategia = config.get("scaling")
    target = config.get("target")

    cols_numericas = df.select_dtypes(include=[np.number]).columns  # obtener las columnas de numeros
    if target in cols_numericas:
        cols_numericas = cols_numericas.drop(target)

    for col, estrategia in zip(cols_numericas, lista_estrategia):
        if estrategia == "z-score":
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        elif estrategia == "min-max":
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        print(f" -> Columna '{col}' escalada con {estrategia}.")
    return df


# ==========================================
# 7. FUNCIONES DE LIMPIEZA DE TEXTO
# ==========================================
def limpiar_texto(df, config):
    if config.get("text_process") == "basic_clean":
        cols_texto = df.select_dtypes(
            include=['string']).columns  # Obtener las columnas que hemos definido como string (texto)
        for col in cols_texto:
            df[col] = df[col].str.lower().str.strip()
            # .lower(): convierte a minusculas
            # .strip(): quita espacios de más que puede haber
        print(" -> Texto pasado a minúsculas y limpiado de espacios.")
    return df


def normalizar_texto(df, config):
    estrategia = config.get("normalize_strategy")
    if estrategia == "none":
        return df

    if estrategia == "basic":
        stop_words = set(
            stopwords.words('english'))  # Una lista con todas las palabras basura (the, a, is, in, at, ...)
        stemmer = SnowballStemmer('english')  # Algoritmo que convierte las palabras en su raiz
        cols_texto = df.select_dtypes(
            include=['string']).columns  # Obtener las columnas que hemos definido como string (texto)

        for col in cols_texto:
            def procesar_celda(texto):
                if pd.isna(texto): return texto  # Vacia

                texto = re.sub(r'[^\w\s]', '', str(texto))  # Quita todos los signos de puntuación
                # \w: cualquier letra
                # \s: espacio en blanco
                # ^: inversa
                # re.sub(): sustituye lo que le digas(los signos de puntuación por vacios) en un texto.

                palabras = texto.split()  # Separa por palabras

                palabras_limpias = [stemmer.stem(p) for p in palabras if
                                    p not in stop_words]  # Quita las basura y pone la raiz de las otras

                return " ".join(palabras_limpias)  # devolvemos la frase con espacio entre las palabras

            df[col] = df[col].apply(procesar_celda)
        print(" -> Texto normalizado (Stopwords eliminadas y aplicado Stemming).")
    return df


# ==========================================
# 8. FUNCIÓN PARA VECTORIZAR TEXTO
# ==========================================
def vectorizar_texto(df, config):
    estrategia = config.get("text_encoding")
    target = config.get("target")
    if estrategia == "none": return df

    cols_texto = df.select_dtypes(include=['string']).columns
    for col in cols_texto:
        if col == target: continue  # Si la columna es el target, la saltamos

        # METODOS DE sklearn
        if estrategia == "one-hot":
            vec = CountVectorizer(binary=True)
        elif estrategia == "frequency":
            vec = CountVectorizer()
        elif estrategia == "tf-idf":
            vec = TfidfVectorizer()

        textos = df[col].fillna('')  # Rellena huecos vacios que pueden haber quedado
        matriz = vec.fit_transform(textos)  # ejecuta la estrategia de vectorización
            # Crea un diccionario con todas las palabras de los datos y da los valores a cada fila

        nombres_cols = [f"{col}_{palabra}" for palabra in
                        vec.get_feature_names_out()]  # Darle nombres a las columnas (comentario_yo, comentario_comer, comentario_hacer, ...)
        df_vec = pd.DataFrame(matriz.toarray(), columns=nombres_cols, index=df.index)  # juntar los nombres con los datos vectorizados
            # matriz.toarray(): los datos vectorizados
            # columns=nombres_cols: los nombres que acabamos de crear
            # index=df.index: el index (los id) de las filas. Pandas le da un valor numerico a cada fila (un id), como igual hemos borrado filas le decimos que siga ese indice

        df = pd.concat([df, df_vec], axis=1)  # Unir todas las columnas
        df = df.drop(col, axis=1)  # Borrar la columna de texto
        print(f" -> Columna '{col}' vectorizada con {estrategia}.")

    return df


# ==========================================
# 9. FUNCIÓN PARA CODIFICAR EL TARGET
# ==========================================
def codificar_objetivo(df, config):
    target = config.get("target")

    if target and target in df.columns:

        if not pd.api.types.is_numeric_dtype(df[target]):  # Si es un problema de clasificacion, lo pasamos a números
            le = LabelEncoder()
            df[target] = le.fit_transform(df[target])
            print(f" -> Variable objetivo '{target}' codificada con LabelEncoder.")

    return df


# ==========================================
# 10. FUNCIÓN PARA CODIFICAR CATEGORÍAS
# ==========================================
def codificar_categoricas(df, config):
    estrategia = config.get("categorical_encoding")
    target = config.get("target_column")  # Escudo
    if estrategia == "none": return df

    cols_categoricas = df.select_dtypes(
        include=['category']).columns  # Conseguir las columnas que anets hemos definido como "Category" (los string)
    cols_categoricas = [c for c in cols_categoricas if c != target]  # Quitamos el target para no romperlo
    if not cols_categoricas: return df

    if estrategia == "one-hot":
        df = pd.get_dummies(df, columns=cols_categoricas, drop_first=True, dtype=int)
    elif estrategia == "label":
        for col in cols_categoricas:
            df[col] = df[col].cat.codes

    print(f" -> Variables categóricas codificadas usando: {estrategia}.")
    return df


# ==========================================
# 11. FUNCIÓN PARA BALANCEAR LOS DATOS
# ==========================================
def balancear_datos(df, config):
    estrategia = config.get("sampling")
    target = config.get("target_column")
    ratio = config.get("sampling_ratio")
    seed = config.get("sampling_seed", 42)

    if estrategia == "none" or not target or target not in df.columns:
        return df

    X = df.drop(columns=[target])
    y = df[target]

    try:
        if estrategia == "oversample":
            # Le pasamos el ratio al inicializar la herramienta
            ros = RandomOverSampler(sampling_strategy=ratio, random_state=seed)
            X_res, y_res = ros.fit_resample(X, y)

        elif estrategia == "undersample":
            rus = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
            X_res, y_res = rus.fit_resample(X, y)
        else:
            return df

        df_res = pd.concat([X_res, y_res], axis=1)
        print(f" -> Filas antes: {len(df)} | Filas después: {len(df_res)}")
        return df_res

    except Exception as e:
        print(f" ⚠️ Error al balancear: {e}")
        return df


# ==========================================
# PIPELINE PRINCIPAL DE PREPROCESADO
# ==========================================
def pipeline_preprocesamiento(df_path, json_path):
    df = pd.read_csv(df_path)
    with open(json_path, 'r') as f:
        config = json.load(f)["preproceso"]

    print(f"--- Iniciando preprocesado ({len(df)} filas originales) ---")

    print("\n[1/11] Comprobando duplicados...")
    df = eliminar_duplicados(df, config)
    # df.to_csv('1_duplicados_eliminados.csv', index=False)

    print("\n[2/11] Limpiando columnas innecesarias...")
    df = eliminar_columnas(df, config)
    # df.to_csv('2_columnas_eliminadas.csv', index=False)

    print("\n[3/11] Asignando tipos de datos...")
    df = asignar_tipos(df, config.get("categoria", []))
    # df.to_csv('3_tipos_asignados.csv', index=False)

    print("\n[4/11] Tratando valores nulos...")
    df = tratar_nulos(df, config)
    # df.to_csv('4_nulos_tratados.csv', index=False)

    print("\n[5/11] Tratando outliers...")
    df = tratar_outliers(df, config)
    # df.to_csv('5_outliers_tratados.csv', index=False)

    print("\n[6/11] Escalando datos numéricos...")
    df = escalar_datos(df, config)
    # df.to_csv('6_datos_escalados.csv', index=False)

    print("\n[7/11] Limpiando y normalizando texto...")
    df = limpiar_texto(df, config)
    df = normalizar_texto(df, config)
    # df.to_csv('7_texto_normalizado.csv', index=False)

    print("\n[8/11] Vectorizando texto (NLP)...")
    df = vectorizar_texto(df, config)
    # df.to_csv('8_texto_vectorizado.csv', index=False)

    print("\n[9/11] Codificando la variable objetivo (Target)...")
    df = codificar_objetivo(df, config)
    # df.to_csv('9_objetivo_codificado.csv', index=False)

    print("\n[10/11] Codificando variables categóricas...")
    df = codificar_categoricas(df, config)
    # df.to_csv('10_categoricas_codificadas.csv', index=False)

    print("\n[11/11] Balanceando clases (Oversampling / Undersampling)...")
    df = balancear_datos(df, config)
    # df.to_csv('11_datos_balanceados.csv', index=False)

    print(f"\n--- Finalizado. DataFrame resultante: {df.shape[0]} filas, {df.shape[1]} columnas ---")
    return df


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # interrupcion CTR + C

    df_procesado = pipeline_preprocesamiento('datos.csv', 'config.json')

    df_procesado.to_csv('datos_listos_para_modelo.csv', index=False)
    print("\n¡Archivo 'datos_listos_para_modelo.csv' guardado con éxito!")