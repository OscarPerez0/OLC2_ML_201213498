from numpy.core.fromnumeric import size
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.naive_bayes import GaussianNB

def ClGaussiano(_info):
    st.title('Clasificador Gaussiano')
    st.subheader('Datos obtenidos del archivo')
    st.write(_info)
    st.subheader('Parametros para clasificacion')
    param = st.text_input('Ingrese Columna a clasificar','*age')
    
    data_top = _info.columns.values
    listaa = data_top.tolist()

    listaaux = ["Columna a borrar"]+listaa
    eliminarcolumna =st.selectbox('Elimine columnas String o Float',listaaux)

   
    listaa.remove(param)
    if eliminarcolumna != 'Columna a borrar':
        listaa.remove(eliminarcolumna)
   
    result = _info[param]


    listadedf = []
    for i in listaa:
        aux = _info[i]
        aux = np.asarray(aux)
        
        listadedf.append(aux)
    listadedf = np.array(listadedf)


    
    le = preprocessing.LabelEncoder()

   
    listafittransform = []
    for x in listadedf:
        listafittransform.append(le.fit_transform(x))

    
    label = le.fit_transform(result)
    

    
    with st.expander("Matriz, modelo"):
        featuresencoders = list(zip((listafittransform)))
        featuresencoders = np.array(featuresencoders)
        tamcolumnas = len(listaa)
        tamfilas = featuresencoders.size
        featuresencoders = featuresencoders.reshape(int(tamfilas/tamcolumnas),tamcolumnas)
       
        st.dataframe(featuresencoders)

    
    with st.expander('Resultados sin strings'):
        features = list(zip(np.asarray(listadedf)))
        features = np.asarray(features)
        tamcolumnas = len(features)
        tamfilas=features.size
        features = features.reshape(int(tamfilas/tamcolumnas),tamcolumnas)
        #st.dataframe(features)
    
    
   
    model = GaussianNB()
    model2 = GaussianNB()
    
    model.fit(np.asarray(features),np.asarray(result))
    model2.fit(featuresencoders,label)

    columna = len(listaa)
    texto = "debe ingresar:  "+str(columna)+" parametros del vector para predecir columna (no ingresar dato de columna), separados por coma(,)"
    predecirresult = st.text_input(texto,'')

    if predecirresult != '':
        entrada = predecirresult.split(",")
        map_obj = list(map(int,entrada))
        map_obj = np.array(map_obj)
        predicted = model.predict(np.asarray([map_obj]))
        predicted2 = model2.predict(np.asarray([map_obj]))
        print(np.asarray([map_obj]))
        
        co1,co2,co3 = st.columns(3)
        with co1:
            st.subheader('Se clasifica en este rango de *Age')
            st.write(predicted)

        coo1,coo2,coo3 = st.columns(3)
       # with coo2:
            #st.subheader('Prediccion sin etiquetas')
           # st.write(predicted2)
    
    