import streamlit as st
import pandas as pd

st.title('Penguin - Classifier')

st.info('This is a Machine Learning Model which Classifies the Penguin')

with st.expander('Data'):
  st.write('Raw Data')
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
  df
