import streamlit as st
import pickle
import pandas as pd

#page config

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

#load trained model

with open("models/student_model.pkl", "rb") as file:
    model = pickle.load(file)

#title

st.title("🎓 Student Performance Predictor")
st.write(
    "Predict a student's final grade (G3) using academic and personal factors."
)

st.divider()

#userinputs

st.subheader("Student Information")

col1, col2 = st.columns(2)

with col1:
    Medu = st.selectbox(
        "Mother's Education Level",
        [0, 1, 2, 3, 4]
    )

    Fedu = st.selectbox(
        "Father's Education Level",
        [0, 1, 2, 3, 4]
    )

    traveltime = st.selectbox(
        "Travel Time",
        [1, 2, 3, 4]
    )

    studytime = st.selectbox(
        "Study Time",
        [1, 2, 3, 4]
    )

    failures = st.number_input(
        "Previous Failures",
        min_value=0,
        max_value=10,
        value=0
    )

    higher = st.selectbox(
        "Wants Higher Education?",
        ["Yes", "No"]
    )

with col2:
    internet = st.selectbox(
        "Internet Access at Home?",
        ["Yes", "No"]
    )

    freetime = st.slider(
        "Free Time",
        1, 5, 3
    )

    goout = st.slider(
        "Social Activity (Going Out)",
        1, 5, 3
    )

    health = st.slider(
        "Health Status",
        1, 5, 3
    )

    absences = st.number_input(
        "Absences",
        min_value=0,
        max_value=100,
        value=0
    )

st.divider()

st.subheader("Academic Performance")

col3, col4 = st.columns(2)

with col3:
    G1 = st.number_input(
        "First Period Grade (G1)",
        min_value=0,
        max_value=20,
        value=10
    )

with col4:
    G2 = st.number_input(
        "Second Period Grade (G2)",
        min_value=0,
        max_value=20,
        value=10
    )

#encoding

higher = 1 if higher == "Yes" else 0
internet = 1 if internet == "Yes" else 0

# ----------------------------------
# Prediction
# ----------------------------------

if st.button("Predict Final Grade"):

    input_data = pd.DataFrame([[
        Medu,
        Fedu,
        traveltime,
        studytime,
        failures,
        higher,
        internet,
        freetime,
        goout,
        health,
        absences,
        G1,
        G2
    ]], columns=[
        "Medu",
        "Fedu",
        "traveltime",
        "studytime",
        "failures",
        "higher",
        "internet",
        "freetime",
        "goout",
        "health",
        "absences",
        "G1",
        "G2"
    ])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Final Grade (G3): {prediction:.2f}"
    )

    if prediction >= 16:
        st.balloons()
        st.write("Excellent expected performance!")
    elif prediction >= 10:
        st.write("Average expected performance.")
    else:
        st.write("Student may require additional academic support.")