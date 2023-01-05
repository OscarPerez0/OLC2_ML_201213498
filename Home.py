
import streamlit as st
import pandas as pd
from RegresionLineal import * 
from RegresionPolinomial import *
from Gaussiano import *
from ClasificadorArbolesdeDecision import *
from RedNeuronal import *


#File Uploader y Header
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Machine Learning - Oscar Perez 201213498')

st.subheader('OLC2 - Diciembre 2022')
uploaded_file = st.file_uploader(label = "Seleccione Archivo para Analizar",type=['csv','xls','xlsx','json'])

df = "" #Variable donde se carga el dataset
if uploaded_file is not None:
  #Cuando lo cargue
  print(uploaded_file)
  print(uploaded_file.type)
  print('Si cargo archivo')
  try:
    if uploaded_file.type == 'text/csv': #CSV 
      print('csv')
      st.text("csv")
      df = pd.read_csv(uploaded_file)
     
    if uploaded_file.type == 'application/vnd.ms-excel': #XLS   pip install xlrd (para instalar libreria)
      print('xls')
      st.text("xls")
      
      df = pd.read_excel(uploaded_file) #XLSX
    if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
      print("xlsx")
      st.text("xlsx")
      df = pd.read_excel(uploaded_file)
    if uploaded_file.type == 'application/json':  #Json
      print("json")
      st.text("json")
      df = pd.read_json(uploaded_file)
  
  except Exception as e: 
    print(e)
    st.subheader('Error al cargar archivo: ',e)

#dropdown
option = st.selectbox(
     'Algoritmos para analizar',
     ('Seleccione una opcion','Regresión lineal', 'Regresión polinomial', 'Clasificador Gaussiano','Clasificador de árboles de decisión','Redes neuronales'))

st.write('Algoritmo Seleccionado: ', option)

#Redirige contenido a mostrar, llama el metodo y pasa el data set

if option == 'Regresión lineal':
  RL(df)
elif option == 'Regresión polinomial':
  RPol(df)
elif option == 'Clasificador Gaussiano':
  ClGaussiano(df)
  
elif option == 'Clasificador de árboles de decisión':
 
  ArbolD(df)
elif option == 'Redes neuronales':
  RedNeuronal(df)
  








