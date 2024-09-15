import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title('Penguin - Classifier')

st.info('This is a Machine Learning Model which Classifies the Penguin')

with st.expander('Data'):
  st.write('Raw Data')
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
  df

  st.write("X")
  X_raw = df.drop('species',axis = 1)
  X_raw

  st.write("Y")
  y_raw = df.species
  y_raw
  
with st.expander("Data Visualization"):
  st.scatter_chart(data = df, x ='bill_length_mm', y = "body_mass_g", color='species')

#Input Features
with st.sidebar:
  st.header("Input Features")
    
  island = st.selectbox('Island',('Biscoe','Dream','Torgesen'))
  gender = st.selectbox('Sex',('male','female'))
  bill_length_mm = st.slider("Bill length (mm)", 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider("Bill Depth (mm)", 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider("Flipper Length (mm)", 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Weight (g)', 2700.0, 6300.0, 4200.0)

#create a Data Frame for the einput features
  data = {'island':island,
          'bill_length_mm':bill_length_mm,
          'bill_depth_mm':bill_depth_mm,
         'flipper_length_mm':flipper_length_mm,
          'body_mass_g':body_mass_g,
          'sex':gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis = 0)

with st.expander('Input Features'):
  st.write('**Input Penguin**')
  input_df
  st.write("**Combined Data**")
  input_penguins

#Data preparation
#Encode X
encode = ["island", "sex"]
df_penguins = pd.get_dummies(input_penguins, prefix = encode)
X = df_penguins[1:]
input_row = df_penguins[:1]

#Encode y
target_mapper = {"Adelie":0,
                 "Chinstrap":1,
                 "Gentoo":2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)


with st.expander("Data Preparation"):
  st.write('**Encoded X (Input Penguin) **')
  input_row
  st.write("Encoded y")
  y

#Model Training and inference
## train ML model
model = RandomForestClassifier()
model.fit(X,y)

##Apply model for prediction
prediction = model.predict(input_row)
prediction_probs = model.predict_proba(input_row)

df_prediction_probs = pd.DataFrame(prediction_probs)
df_prediction_probs.columns = ["Adelie", "Chinstarp", "Gentoo"]
df_prediction_probs.rename(columns = {0:'Adelie',
                                   1:'Chinstrap',
                                   2:'Gentoo'})



#display species
st.subheader('Predicted Species')
df_prediction_probs
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction[0]]))







