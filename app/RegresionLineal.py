import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def RL(_info):
    print("Ingreso a regresion Lineal") 
    st.title('Regresion lineal') 
    st.subheader('Informacion obtenida')
    st.write(_info)
    st.subheader('Parametros para analisis')
    col1,col2 = st.columns(2)
    with col1:
        px = st.text_input('Ingrese columna X','NO') 
    
    with col2:
        py = st.text_input('Ingrese columna Y','A') 

    
    x = np.asarray(_info[px]).reshape(-1,1)
    y = _info[py]
    
    regresion = linear_model.LinearRegression()
    regresion.fit(x,y)

    y_pred = regresion.predict(x)
    regresion = regresion.coef_



    st.subheader('Resultados')
    
    fig, ax = plt.subplots()
    ax.scatter(x,y, color='black')
    ax.plot(x,y_pred,color='red')
    plt.title('Regresio lineal\nCoeficiente de regresion: '+str(regresion))
    plt.xlabel(px)
    plt.ylabel(py)
    plt.grid()
 
    texto = str(round(regresion.coef_[0],2))+"X+"+str(round(regresion.intercept_,2))
    print("Ecuacion: ",texto)
    d = {'Coeficiente de regresion': [regresion], 'Error cuadratico':[mean_squared_error(y,y_pred)], 'Coeficinte de determinacion':[r2_score(y,y_pred)],'Ecuacion lineal Ax + B':texto}
    dresult = pd.DataFrame(data=d)
    st.dataframe(dresult)

    with st.expander("Grafica regresion lineal"):
        st.pyplot(fig)
    
    with st.expander("Grafica de Puntos"):
        fig2,ax2 = plt.subplots()
        ax2.scatter(x,y, color='red')
        st.pyplot(fig2)


    c1,c2,c3 = st.columns(3)
    with c2:
       
        calcular = st.text_input('Ingrese valor para aproximar con este modelo: ','0')
        variable = regresion.predict([[int(calcular)]])
        st.text(variable)