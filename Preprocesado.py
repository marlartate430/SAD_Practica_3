#######################################################
#  Receta Python para imputar  missing con la media   #
#######################################################

import pandas as pd
### Choice of aggregate functions
## On non-NA values in the group
## - numeric choice : mean, median, sum, std, var, min, max, prod
## - group choice : first, last, count
import pandas as pd

import pandas as pd


def filling_function():
    df = pd.read_csv('datos.csv')
    for column in df.columns:
        if not df[column].mode().empty:
            moda = df[column].mode()[0]  # Conseguir la moda
            #print(f"La moda de {column} es: {moda}")
            for i in range(len(df[column])):
                valor_actual = df[column][i]
                if pd.isna(valor_actual): #valor vacio
                    df.loc[i, column] = moda


    df.to_csv('datos_procesados.csv', index=False)






'''
# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

critAgrup = 'A_PEDAL_FORMAT'
#���CUIDADO!! Debe ser el nombre del dataset que has puesto como output de tu receta y le he a�adido "_df"

NombreDatasetOutput_df['A_PEDAL_F'] = NombreDatasetOutput_df.groupby(critAgrup)['A_PEDAL_F'].transform(filling_function)#aplicar la funcion de imputacion

NombreDatasetOutput_df['A_PEDAL_F'] = NombreDatasetOutput_df['A_PEDAL_F'].apply('int64')#convertirlo en integer, prueba a no ponerlo y ver que pasa

# Write recipe outputs
NombreDatasetOutput = dataiku.Dataset("NombreDatasetOutput")#generar el dataset nuevo tras la imputacion
NombreDatasetOutput.write_with_schema(NombreDatasetOutput_df)#incorporarlo al esquema

##############################################################
#   Receta Python para escalar los atributos descompensados  #
##############################################################

# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# apply the z-score method in Pandas using the .mean() and .std() methods
def z_score(v):
    #input: it expects a column from the data frame
    #output: the column will be scaled using the z-score technique
    
    # copy the column
    v_norm = v
    # apply the z-score method
    v_norm = (v - v.mean()) / v.std()
    return v_norm


# apply the maximum absolute scaling in Pandas using the .abs() and .max() methods
def maximum_absolute_scaling(df):
    #input: a whole dataFrame
    #output: a whole dataFrame where integer type columns with a mean value > 60 will be scaled
    
    print(df.head())
    # copy the dataframe
    df_scaled = df.copy()
    # apply maximum absolute scaling
    for column in df_scaled.columns:
        if df_scaled.dtypes[column] == np.int64 and df_scaled[column].mean()>60:
            df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()
    return df_scaled


# apply the min-max scaling in Pandas using the .min() and .max() methods
def min_max_scaling(v):
    #input: it expects a column from the data frame
    #output: the column will be scaled using the min-max technique
    
    # copy the column
    v_norm = v
    # apply min-max scaling
    #for column in df_norm.columns:
    v_norm = (v - v.min()) / (v.max() - v.min())

    return v_norm 



# Read recipe inputs
NombreDataSetOutput = dataiku.Dataset("NombreDataSetInput)
NombreDataSetOutput_df = NombreDataSetOutput.get_dataframe()


# Compute recipe outputs from inputs

#calling the min_max_scaling funtion with a column
#ACC_imputed_discrtized_df['COUNTY_F']=min_max_scaling(ACC_imputed_discrtized_df['COUNTY_FORMAT'])

#Calling the maximum_absolute_scaling function with the whole dataFrame as parameter, so all integer columns will be scaled under certain conditions
NombreDataSetOutput_df=maximum_absolute_scaling(NombreDataSetOutput)

# Write recipe outputs
NombreDataSetOutput = dataiku.Dataset("NombreDataSetOutput")
NombreDataSetOutput.write_with_schema(NombreDataSetOutput_df)


########################################################################################
#  Receta para transformar atributos categoriales en numericos sin generar nuevos atr. #
########################################################################################

# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
NombreDatasetOutput = dataiku.Dataset("NombreDatasetInput")
NombreDatasetOutput_df = NombreDatasetOutput.get_dataframe()


# Compute recipe outputs from inputs

cat_columns = NombreDatasetOutput_df.columns



NombreDatasetOutput_df[cat_columns] = NombreDatasetOutput_df[cat_columns].apply(lambda x: pd.factorize(x)[0])


# Write recipe outputs
NombreDatasetOutput = dataiku.Dataset("NombreDatasetOutput")
NombreDatasetOutput.write_with_schema(NombreDatasetOutput_df)

##########################################################################################
#  Receta para transformar atributos categoriales en numericos sin generar nuevos atr.   #
##########################################################################################
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
NombreDatasetOutput = dataiku.Dataset("NombreDatasetInput")
NombreDatasetOutput_df = NombreDatasetOutput.get_dataframe()


# Compute recipe outputs from inputs

cat_columns = kk_df.columns
catFormat=list()

def filterFORMAT(cols):
    catFormatLoc=list()
    for cat in cols:
        if 'FORMAT' in cat:
            catFormatLoc.append(cat)
    return catFormatLoc


def cat2num(df):
    catFormat=list()
    cat_columns = kk_df.keys()
    catFormat=filterFORMAT(cat_columns)
    for cat in catFormat:
        newCat=str(cat)+"_Num"
        df[newCat]=pd.factorize(df[cat])[0]
    return(df)


NombreDatasetOutput_df= cat2num(NombreDatasetOutput_df)

# Write recipe outputs
NombreDatasetOutput = dataiku.Dataset("NombreDatasetOutput")
NombreDatasetOutput.write_with_schema(NombreDatasetOutput_df)
'''
if __name__ == "__main__":
    filling_function()