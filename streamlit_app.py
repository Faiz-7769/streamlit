import streamlit as st

st.title('Penguin - Classifier')

st.info('This is a Machine Learning Model which Classifies the Penguin')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
df
