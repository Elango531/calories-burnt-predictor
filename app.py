import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Train model directly from CSV
@st.cache_resource
def train_model():
    df = pd.read_csv('calories.csv')
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df = df.drop(columns=['User_ID'])
    X = df.drop(columns=['Calories'])
    y = df['Calories']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# App title
st.title('🔥 Calories Burnt Predictor')
st.write('Enter your exercise details to predict calories burned!')

# Sidebar inputs
st.sidebar.header('Enter Your Details')

gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
age = st.sidebar.slider('Age', 18, 80, 25)
height = st.sidebar.slider('Height (cm)', 140, 200, 170)
weight = st.sidebar.slider('Weight (kg)', 40, 150, 70)
duration = st.sidebar.slider('Exercise Duration (mins)', 1, 60, 30)
heart_rate = st.sidebar.slider('Heart Rate (bpm)', 60, 130, 90)
body_temp = st.sidebar.slider('Body Temperature (°C)', 37.0, 42.0, 39.0)

# Convert gender to number
gender_num = 1 if gender == 'Male' else 0

# Predict button
if st.sidebar.button('Predict Calories! 🔥'):
    input_data = np.array([[gender_num, age, height,
                            weight, duration,
                            heart_rate, body_temp]])
    prediction = model.predict(input_data)[0]

    st.success(f'🔥 You burned approximately **{prediction:.1f} calories!**')

    st.subheader('Your Exercise Summary')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Duration', f'{duration} mins')
        st.metric('Heart Rate', f'{heart_rate} bpm')
        st.metric('Weight', f'{weight} kg')
    with col2:
        st.metric('Age', f'{age} years')
        st.metric('Height', f'{height} cm')
        st.metric('Body Temp', f'{body_temp} °C')

    st.subheader('What does this mean? 🍽️')
    if prediction < 100:
        st.info('🚶 Light workout — equivalent to a small snack')
    elif prediction < 200:
        st.info('🏃 Moderate workout — equivalent to a full meal')
    else:
        st.info('💪 Intense workout — equivalent to a large meal!')

else:
    st.info('👈 Adjust the sliders and click Predict!')
