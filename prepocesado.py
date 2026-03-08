import pandas as pd
import numpy as np
import json
import re

# Librerías de Machine Learning y NLP
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Descargar diccionarios de NLTK en inglés (se ejecuta silenciosamente)
nltk.download('stopwords', quiet=True)


# ==========================================
# 1. FUNCIÓN PARA ASIGNAR TIPOS
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
    return df


# ==========================================
# 2. FUNCIÓN PARA NULOS
# ==========================================
def tratar_nulos(df, config):
    accion = config.get("missing_values")
    if accion == "delete":
        return df.dropna()
    elif accion == "impute":
        estrategia = config.get("impute_strategy")
        lista_tipos = config.get("categoria")

        for col, tipo in zip(df.columns, lista_tipos):
            if tipo != "text":
                if df[col].isnull().any():
                    if estrategia == "mode" and not df[col].mode().empty:
                        df[col] = df[col].fillna(df[col].mode()[0])
                    elif estrategia == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                    elif estrategia == "median" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
    return df


# ==========================================
# 3. FUNCIÓN PARA OUTLIERS
# ==========================================
def tratar_outliers(df, config):
    accion = config.get("outliers")
    if accion == "none": return df

    estrategia = config.get("outlier_strategy")
    cols_numericas = df.select_dtypes(include=[np.number]).columns #obtener las columnas de numeros

    for col in cols_numericas:
        #Métod del Rango Intercuartílico (IQR): va a meter los outliers en el rango de 1.5*el rango del 50% de la mediana
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR
        es_outlier = (df[col] < limite_inf) | (df[col] > limite_sup) # definir que es un valor Outlier (usando IQR)

        if accion == "delete":
            df = df[~es_outlier] #Solo dejas no Outliers (la ~ es como un NOT)
        elif accion == "impute" and es_outlier.any():
            if estrategia == "max-min":
                df[col] = df[col].clip(lower=limite_inf, upper=limite_sup)
            elif estrategia == "median":
                df.loc[es_outlier, col] = df[col].median()
            elif estrategia == "mean":
                df.loc[es_outlier, col] = round(df[col].mean())
    return df


# ==========================================
# 4. FUNCIÓN PARA ESCALADO
# ==========================================
def escalar_datos(df, config):
    accion = config.get("scaling")
    if accion == "none": return df

    cols_numericas = df.select_dtypes(include=[np.number]).columns #obtener las columnas de numeros
    for col in cols_numericas:
        if accion == "z-score":
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        elif accion == "min-max":
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


# ==========================================
# 5. FUNCIÓN PARA CODIFICAR CATEGORÍAS
# ==========================================
def codificar_categoricas(df, config):
    estrategia = config.get("categorical_encoding")
    if estrategia == "none": return df

    cols_categoricas = df.select_dtypes(include=['category']).columns #Conseguir las columnas que anets hemos definido como "Category" (los string)
    if cols_categoricas.empty: return df

    if estrategia == "label":
        for col in cols_categoricas:
            df[col] = df[col].cat.codes
    return df


# ==========================================
# 6. FUNCIONES DE LIMPIEZA DE TEXTO
# ==========================================
def limpiar_texto(df, config):
    if config.get("text_process") == "basic_clean":
        cols_texto = df.select_dtypes(include=['string']).columns # Obtener las columnas que hemos definido como string (texto)
        for col in cols_texto:
            df[col] = df[col].str.lower().str.strip()
                # .lower(): convierte a minusculas
                # .strip(): quita espacios de más que puede haber
    return df


def normalizar_texto(df, config):
    estrategia = config.get("normalize_strategy")
    if estrategia == "none":
        return df

    if estrategia == "basic":
        stop_words = set(stopwords.words('english')) # Una lista con todas las palabras basura (the, a, is, in, at, ...)
        stemmer = SnowballStemmer('english') # Algoritmo que convierte las palabras en su raiz
        cols_texto = df.select_dtypes(include=['string']).columns # Obtener las columnas que hemos definido como string (texto)

        for col in cols_texto:
            def procesar_celda(texto):
                if pd.isna(texto): return texto # Vacia

                texto = re.sub(r'[^\w\s]', '', str(texto)) # Quita todos los signos de puntuación
                    # \w: cualquier letra
                    # \s: espacio en blanco
                    # ^: inversa
                    # re.sub(): sustituye lo que le digas(los signos de puntuación por vacios) en un texto.

                palabras = texto.split() #Separa por palabras

                palabras_limpias = [stemmer.stem(p) for p in palabras if p not in stop_words] # Quita las basura y pone la raiz de las otras

                return " ".join(palabras_limpias) #devolvemos la frase con espacio entre las palabras

            df[col] = df[col].apply(procesar_celda)

    return df


# ==========================================
# 7. FUNCIÓN PARA VECTORIZAR TEXTO
# ==========================================
def vectorizar_texto(df, config):
    estrategia = config.get("text_encoding")
    if estrategia == "none": return df

    cols_texto = df.select_dtypes(include=['string']).columns
    for col in cols_texto:

        #METODOS DE sklearn
        if estrategia == "one-hot":
            vec = CountVectorizer(binary=True)
        elif estrategia == "frequency":
            vec = CountVectorizer()
        elif estrategia == "tf-idf":
            vec = TfidfVectorizer()

        textos = df[col].fillna('')  # Rellena huecos vacios que pueden haber quedado
        matriz = vec.fit_transform(textos)  # ejecuta la estrategia de vectorización
            # Crea un diccionario con todas las palabras de los datos y da los valores a cada fila


        nombres_cols = [f"{col}_{palabra}" for palabra in vec.get_feature_names_out()] # Darle nombres a las columnas (comentario_yo, comentario_comer, comentario_hacer, ...)
        df_vec = pd.DataFrame(matriz.toarray(), columns=nombres_cols, index=df.index) # juntar los nombres con los datos vectorizados
            # matriz.toarray(): los datos vectorizados
            # columns=nombres_cols: los nombres que acabamos de crear
            # index=df.index: el index (los id) de las filas. Pandas le da un valor numerico a cada fila (un id), como igual hemos borrado filas le decimos que siga ese indice

        df = pd.concat([df, df_vec], axis=1) # Unir todas las columnas
        df = df.drop(col, axis=1) # Borrar la columna de texto

    return df


# ==========================================
# 8. ORQUESTADOR PRINCIPAL
# ==========================================
def pipeline_preprocesamiento(df_path, json_path):
    df = pd.read_csv(df_path)
    with open(json_path, 'r') as f:
        config = json.load(f)["preproceso"]

    print(f"--- Iniciando preprocesado ({len(df)} filas originales) ---")

    df = asignar_tipos(df, config.get("categoria", []))

    # TRATAMIENTO DE DATOS
    df = tratar_nulos(df, config)
    #df.to_csv('1_nulos_tratados.csv', index=False)

    df = tratar_outliers(df, config)
    #df.to_csv('2_outliers_tratados.csv', index=False)

    df = escalar_datos(df, config)
    #df.to_csv('3_datos_escalados.csv', index=False)

    # PROCESAMIENTO DE CATEGORÍAS
    df = codificar_categoricas(df, config)
    #df.to_csv('4_categoricas_codificadas.csv', index=False)

    # PROCESAMIENTO DE TEXTO
    df = limpiar_texto(df, config)
    #df.to_csv('5_texto_limpio.csv', index=False)

    df = normalizar_texto(df, config)
    #df.to_csv('6_texto_normalizado.csv', index=False)

    df = vectorizar_texto(df, config)
    #df.to_csv('7_texto_vectorizado_final.csv', index=False)

    print(f"\n--- Finalizado. DataFrame resultante: {df.shape[0]} filas, {df.shape[1]} columnas ---")
    return df


if __name__ == "__main__":
    df_procesado = pipeline_preprocesamiento('datos.csv', 'config.json')

    df_procesado.to_csv('datos_listos_para_modelo.csv', index=False)
    print("\n¡Archivo 'datos_listos_para_modelo.csv' guardado con éxito!")