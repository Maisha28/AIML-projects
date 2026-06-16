import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Salary Prediction System",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------

with open("models/salary_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/model_columns.pkl", "rb") as file:
    model_columns = pickle.load(file)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("📊 Project Information")

st.sidebar.info(
    """
    Model: Random Forest Regressor

    Best R² Score: 0.547

    Project Type:
    Salary Prediction System
    """
)

# -----------------------------
# Title
# -----------------------------

st.title("💰 Salary Prediction System")

st.write(
    "Predict an employee's monthly salary using qualifications, "
    "experience, skills, and workplace information."
)

st.divider()

# -----------------------------
# Education Section
# -----------------------------

st.header("🎓 Education")

col1, col2 = st.columns(2)

with col1:

    yrs_qual = st.number_input(
        "Years Qualified",
        min_value=0,
        value=1
    )

    highest_qual = st.selectbox(
        "Highest Qualification",
        [1, 2, 3, 7, 9, 11, 12, 13, 14, 15]
    )

with col2:

    area_of_study = st.selectbox(
        "Area Of Study",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    sex = st.selectbox(
        "Sex",
        [1, 2]
    )

st.divider()

# -----------------------------
# Job Information
# -----------------------------

st.header("💼 Job Information")

col1, col2 = st.columns(2)

with col1:

    industry = st.selectbox(
        "Industry",
        [
            "A","B","C","D","E","F","G",
            "H","I","J","K","L","M","N",
            "O","P","Q","R","S","T","U"
        ]
    )

    occupation = st.selectbox(
        "Occupation",
        [0,1,2,3,4,5,6,7,8,9,15]
    )

    sector = st.number_input(
        "Sector",
        min_value=0,
        value=1
    )

with col2:

    job_quals = st.number_input(
        "Job Qualifications",
        min_value=0,
        value=1
    )

    qual_needed = st.number_input(
        "Qualification Needed",
        min_value=0,
        value=1
    )

    experience_needed = st.number_input(
        "Experience Needed",
        min_value=0,
        value=1
    )

st.divider()

# -----------------------------
# Skills
# -----------------------------

st.header("🧠 Skills")

col1, col2 = st.columns(2)

with col1:

    computer = st.number_input(
        "Computer Usage",
        min_value=0,
        value=1
    )

    computer_level = st.number_input(
        "Computer Level",
        min_value=0,
        value=1
    )

    manual_skill = st.number_input(
        "Manual Skill",
        min_value=0,
        value=1
    )

with col2:

    problem_solving_quick = st.number_input(
        "Problem Solving Quick",
        min_value=0,
        value=1
    )

    problem_solving_long = st.number_input(
        "Problem Solving Long",
        min_value=0,
        value=1
    )

    labour = st.number_input(
        "Labour",
        min_value=0,
        value=1
    )

st.divider()

# -----------------------------
# Workplace
# -----------------------------

st.header("👥 Workplace")

col1, col2 = st.columns(2)

with col1:

    no_subordinates = st.number_input(
        "Number Of Subordinates",
        min_value=0,
        value=0
    )

    influencing = st.number_input(
        "Influencing",
        min_value=0,
        value=1
    )

    negotiating = st.number_input(
        "Negotiating",
        min_value=0,
        value=1
    )

with col2:

    advising = st.number_input(
        "Advising",
        min_value=0,
        value=1
    )

    instructing = st.number_input(
        "Instructing",
        min_value=0,
        value=1
    )

    choose_hours = st.number_input(
        "Choose Hours",
        min_value=0,
        value=1
    )

# -----------------------------
# Prediction Button
# -----------------------------

st.divider()

if st.button("💰 Predict Salary"):

    input_data = {
        "yrs_qual": yrs_qual,
        "sex": sex,
        "influencing": influencing,
        "negotiating": negotiating,
        "sector": sector,
        "no_subordinates": no_subordinates,
        "choose_hours": choose_hours,
        "job_quals": job_quals,
        "qual_needed": qual_needed,
        "experience_needed": experience_needed,
        "advising": advising,
        "instructing": instructing,
        "problem_solving_quick": problem_solving_quick,
        "problem_solving_long": problem_solving_long,
        "labour": labour,
        "manual_skill": manual_skill,
        "computer": computer,
        "computer_level": computer_level,
        "industry": industry,
        "occupation": occupation,
        "highest_qual": highest_qual,
        "area_of_study": area_of_study
    }

    input_df = pd.DataFrame([input_data])

    input_df = pd.get_dummies(
        input_df,
        columns=[
            "industry",
            "occupation",
            "highest_qual",
            "area_of_study"
        ],
        drop_first=True
    )

    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(input_df)

    st.success("Prediction Complete")

    st.metric(
        label="💰 Predicted Monthly Salary",
        value=f"₹ {prediction[0]:,.0f}"
    )