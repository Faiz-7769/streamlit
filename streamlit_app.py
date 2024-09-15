import streamlit as st
import pandas as pd

st.title('Penguin - Classifier')

st.info('This is a Machine Learning Model which Classifies the Penguin')

with st.expander('Data'):
  st.write('Raw Data')
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
  df

  st.write("X")
  X = df.drop('species',axis = 1)
  X

  st.write("Y")
  y = df.species
  y
  
with st.expander("Data Visualization"):
  st.scatter_chart(data = df, x ='bill_length_mm', y = "body_mass_g", color='species')

  #Data prepration
  with st.sidebar:
    st.header("Input Features")
    
    island = st.selectbox('Island',('Biscoe','Dream','Torgesen'))
    gender = st.selectbox('Sex',('Male','Female'))
    bill_length_mm = st.slider("Bill length (mm)", 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider("Bill Depth (mm)", 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider("Flipper Length (mm)", 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Weight (g)', 2700.0, 6300.0, 4200.0)
