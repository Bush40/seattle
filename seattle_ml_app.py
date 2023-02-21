import streamlit as st
import pandas as pd

import sklearn

from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# SeattleHouse Price Prediction App
This app predicts the **Seattle House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
seattle=pd.read_csv(r"C:\Users\User\Cleaned_data_5.csv")
Y = seattle['price']
X = seattle[['grade','sqft_living','bedrooms','bathrooms']]
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    # how old is the house? (age)
    grade = st.number_input('How old is the house (in grade)?', min_value=0, step=1)
# area of the house
    sqft_living = st.slider('Area of the house', 1000, 5000, 1500)
 
# no. of bedrooms in the house
    bedrooms = st.number_input('No. of bedrooms', min_value=0, step=1)
 
# no. of balconies in the house
    bathrooms = st.radio('No. of bathrooms', (0, 1, 2 , 3))
 

 
 
    data = {'grade': grade,
            'sqft_living': sqft_living,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            
            }
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(X)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of Price')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
