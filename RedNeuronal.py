import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
def RedNeuronal(data):
    st.title('Red Neuronal')
    
    st.subheader('Datos obtenidos del archivo')
    st.write(data)
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox('Columna Independiente x', data.columns)
    with col2:
        y_var = st.selectbox('Columna Dependiente y', data.columns)

    predict = st.text_input('Ingrese el valor a predecir, Columna X')
    predict = [[int(predict)]]

    # x = np.asarray(data[x_var]).reshape(-1, 1)
    x = data[x_var]
    y = data[y_var]

    lab = preprocessing.LabelEncoder()
    if data[x_var].dtype == 'object':
        x = lab.fit_transform(x)


    st.title('Resultados')
    X = x[:, np.newaxis]
    i = 0
    fig, ax = plt.subplots()
    plt.rc('font', size = 10)
    colors = [ 'pink',  'aqua', 'green',  'yellow', 'purple', 'olive',  'chocolate', 'wheat']
    while True:
        i = i + 1
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        mlr = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)
        mlr.fit(X_train, y_train)

        plt.scatter(X_train, y_train, color='red', label="Training" if i == 1 else "test")
        plt.scatter(X_test, y_test, color='orange', label="Testing" if i == 1 else "")
        plt.scatter(X, mlr.predict(X), c=np.random.rand(3,), label="#Iteracion " + str(i))
        plt.legend(["Training" if i == 1 else "", "Testing" if i == 1 else "", "#Iteracion: " + str(i)])
        if mlr.score(X_train, y_train) > 0.8 or i == 20:
            break

    st.subheader('Resultado de la prediccion columna dependiente')
    st.write(mlr.predict(predict))
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.figure(figsize=(20, 10))
    st.pyplot(fig)