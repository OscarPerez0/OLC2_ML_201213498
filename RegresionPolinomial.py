import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

def RPol(_info):
    st.title("Regresion Polinomial")
    st.subheader('Datos Obtenidos del archivo')
    st.write(_info)
    st.subheader('Parametros para Analisis')
    col1,col2,col3 = st.columns(3)
    with col1:
        paramx = st.text_input('Columna independiente X','X')
    
    with col2:
        paramy = st.text_input('Columna dependiente Y','Y')
    
    with col3:
        grado = st.text_input('Grado de Ecuacion Polinomial','2')

    co1,co2,co3 = st.columns(3)
    with co1:
        prediccion = st.text_input('Ingrese valor de la columna X a predecir','000')

    x = _info[paramx]
    y = _info[paramy]

    x = np.asarray(x)
    y = np.asarray(y)

    x = x[:,np.newaxis]
    y = y[:,np.newaxis]

    #plot  de Ecucacion
    #fig1, ax = plt.subplots()
    #ax.scatter(x,y)
 
    
    #Preparacion de la informacion
    nb_degree = int(grado)
    polinomial_feature = PolynomialFeatures(degree = nb_degree)
    X_TRANSF = polinomial_feature.fit_transform(x)

 
    #Definir y entrenar el modelo
    model = LinearRegression()
    model.fit(X_TRANSF,y)

    
    #Calcular bayesiana
    Y_NEW = model.predict(X_TRANSF)
    
    rmse = np.sqrt(mean_squared_error(y,Y_NEW))
    r2 = r2_score(y, Y_NEW)

  
    #Prediccion
    x_new_min = 0.0
    x_new_max = float(prediccion)  # el calculo de la prediccion

    X_NEW = np.linspace(x_new_min,x_new_max,50)
    X_NEW = X_NEW[:,np.newaxis]

    X_NEW_TRANSF = polinomial_feature.fit_transform(X_NEW)
    Y_NEW = model.predict(X_NEW_TRANSF)

    fig, ax = plt.subplots()
    plt.plot(X_NEW, Y_NEW, color='red',linewidth=3)
    ax.scatter(X_NEW,Y_NEW, color='black')
    plt.grid()
    plt.xlim(x_new_min, x_new_max)
    title = 'Degree = {}; RMSE = {}; R2={};'.format(nb_degree,rmse,r2)
    plt.title("Regresion polinomial\n"+title ,fontsize = 10)
    plt.xlabel(paramx)
    plt.ylabel(paramy)


    #Impresion de resultados
    
    st.subheader('Resultados')
    ecuacion = ''
    coeficientes = model.coef_
    coeficientes = np.asarray(coeficientes)
    
    auxcoeficientes = coeficientes.reshape(-1,1)
    intercept = model.intercept_
    intercept = str(model.intercept_).replace('[','')
    intercept = str(intercept).replace(']','')
    print('\n\n\nCoeficientes= ',auxcoeficientes)
    print('\nIntercept= ',intercept)
    for xxx in range(int(grado),0,-1):
        aux = str(auxcoeficientes[xxx]).replace('[','')
        aux = aux.replace(']','')
        print('coeficiente = ',aux)
        ecuacion += str(aux)+'X^'+str(xxx)+' + '
    ecuacion+= str(intercept)

    print('Ecuacion = ',str(ecuacion))
    d = {'RMSE' : rmse,'R2':  r2,'Prediccion':Y_NEW[Y_NEW.size-1],'Ecuacion de '+grado+' grado':str(ecuacion)}
    dresult = pd.DataFrame(data = d)
    st.dataframe(dresult)

    with st.expander("Grafica de Puntos"):
        fig2,ax2 = plt.subplots()
        plt.grid()
        ax2.scatter(x,y, color='black')
        st.pyplot(fig2)

    with st.expander("Grafica regresion polinomial"):
        st.pyplot(fig)
    