import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def RL(_info):
    print("Ingreso a regresion Lineal") #print server
    st.title('Regresion lineal') #titulo pagina
    st.subheader('Informacion obtenida del archivo')
    st.write(_info)
    st.subheader('Columnas a analizar')
    col1,col2 = st.columns(2)
    with col1:
        paramx = st.text_input('Ingrese columna independiente X','X') #el NO varia segun lo ingresado en el input
    
    with col2:
        paramy = st.text_input('Ingrese columna dependiente Y','Y') #el A varia segun lo ingresado en el input
   

    #inicio del Algoritmo de  regresion lineal
    x = np.asarray(_info[paramx]).reshape(-1,1)
    y = _info[paramy]
    
    regr = linear_model.LinearRegression()
    regr.fit(x,y)

    y_pred = regr.predict(x)
    regresion = regr.coef_

    #prueba de resultado
  

    st.subheader('Datos de la Regresion')
    
    fig, ax = plt.subplots()
    ax.scatter(x,y, color='black')
    ax.plot(x,y_pred,color='red')
    plt.title('Coeficiente de regresion: '+str(regresion))#,'  con un error cuadratico: ',mean_squared_error(y,y_pred))
    plt.xlabel(paramx)
    plt.ylabel(paramy)
    plt.grid()
    #st.pyplot(fig)
    texto = str(round(regr.coef_[0],2))+"X+"+str(round(regr.intercept_,2))
    print("Ecuacion: ",texto)
    d = {'Coeficiente de regresion': [regresion], 'Error cuadratico':[mean_squared_error(y,y_pred)], 'Coeficinte de determinacion':[r2_score(y,y_pred)],'Ecuacion lineal Ax + B':texto}
    dresult = pd.DataFrame(data=d)
    st.dataframe(dresult)

    with st.expander("Regresion lineal y tendencia"):
        
        st.pyplot(fig)
    
    with st.expander("Grafica de Puntos"):
        
        fig2,ax2 = plt.subplots()
        ax2.scatter(x,y, color='red')
        st.pyplot(fig2)

#ingresando valor para aproximar
    c1,c2,c3 = st.columns(3)
    with c1:
       
        calcular = st.text_input('Ingrese valor para aproximar con este modelo: ','0')
        variable = regr.predict([[int(calcular)]])
        st.text(variable)