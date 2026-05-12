import streamlit as st
import joblib
import numpy as np

model = joblib.load("calories_model.pkl")

st.title("calories burnt")
st.write('Enter your exercise details to predict calories burned!')

gender = st.sidebar.selectbox('Gender', ['male', 'female'])
age = st.sidebar.slider('Age', 18, 80, 25)
height = st.sidebar.slider("Height (cm)", 140, 200, 170)
weight = st.sidebar.slider('Weight (kg)', 40, 150, 70)
duration = st.sidebar.slider('Exercise Duration (mins)', 1, 60, 30)
heart_rate = st.sidebar.slider('Heart Rate (bpm)', 60, 130, 90)
body_temp = st.sidebar.slider('Body Temperature (°C)', 37.0, 42.0, 39.0)

gender_num = 1 if gender == 'male' else 0

if st.sidebar.button("Predict calories.."):
    input_data = np.array([[gender_num, age, height, weight,
                            duration, heart_rate, body_temp]])
    prediction = model.predict(input_data)[0]
    
    st.success(f'you burnt approximatly **{prediction:.1f} calories**')
    st.subheader('Your Exercise Summary')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Duration', f'{duration} mins')
        st.metric('Heart rate', f'{heart_rate} bpm')
        st.metric('Weight', f'{weight} kgs' )
    with col2:
        st.metric('Age', f'{age} years')
        st.metric('Height', f'{height }cm')
        st.metric('Body Temp', f'{body_temp} °C')
    
    st.subheader("What does this mean?")
    if prediction < 100:
        st.info('Light workout — equivalent to a small snack')
    elif prediction < 200:
        st.info('Moderate workout — equivalent to a Full meal')
    else:
        st.info('Intense workout — equivalent to a Large meal')
else:
    st.info("Adjust the sliders and click predict")