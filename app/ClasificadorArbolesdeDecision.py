from numpy.core.fromnumeric import size
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing

def ArbolD(_info):
    st.title('Clasificador de Arbol de desicion')
    st.subheader('Informacion')
    st.write(_info)
   

    st.subheader('Parametros')
    param = st.text_input('Ingrese parametro de aproximacion','I')
    
    
    
    data_top = _info.columns.values
    listaa = data_top.tolist()

    listaaux = ["Seleccionar"]+listaa
    eliminarcolumna =st.selectbox('Eliminar una columna?',listaaux)

   
    listaa.remove(param)
    if eliminarcolumna != 'Seleccionar':
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

 
    with st.expander("Matriz de valores"):
        featuresencoders = list(zip((listafittransform)))
        featuresencoders = np.array(featuresencoders)
        tamcolumnas = len(listaa)
        tamfilas = featuresencoders.size
        featuresencoders = featuresencoders.reshape(int(tamfilas/tamcolumnas),tamcolumnas)
      
      

    with st.expander('matriz de clasificacion con valores reales'):
        features = list(zip(np.asarray(listadedf)))
        features = np.asarray(features)
        tamcolumnas = len(features)
        tamfilas=features.size
        features = features.reshape(int(tamfilas/tamcolumnas),tamcolumnas)
        st.dataframe(features)
  

  
    clf = DecisionTreeClassifier(max_depth=4).fit(features,result)
    fig,ax = plt.subplots()
    plot_tree(clf,filled = True, fontsize=10)
    st.subheader('Graficas')
    with st.expander("Arbol sin clasificar"):
        plt.figure(figsize=(50,50))
        st.pyplot(fig)

    clf2 = DecisionTreeClassifier(max_depth=5).fit(featuresencoders,label)
    fig2,ax2 = plt.subplots()
    plot_tree(clf2,filled = True)
    with st.expander("Arbol clasificando"):
        plt.figure(figsize=(50,50))
        st.pyplot(fig2)
