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

#create a Data Frame for the einput features
  data = {'island':island,'bill_length_mm':bill_length_mm,'bill_depth_mm':bill_depth_mm,
         'flipper_length_mm':flipper_length_mm,'body_mass_g':body_mass_g, 'Gender':gender}
  input_df = pd.Dataframe(data, index[0])
  input penguins = pd.concat([input_df, X], axis = 0)
input_df
