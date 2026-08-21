import streamlit as st
import pandas as pd
import joblib


# LOAD MODEL
model = joblib.load("medical_result_model.pkl")

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Medical Result Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical Result Predictor")

st.write(
    "Enter the patient information below to predict "
    "the medical test result."
)

st.info(
    "This application is for educational purposes only "
    "and is not a medical diagnostic tool."
)


# INPUT FEATURES
age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50,
    step=1
)

gender = st.selectbox(
    "Gender",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

heart_rate = st.number_input(
    "Heart Rate",
    min_value=20.0,
    max_value=200.0,
    value=80.0
)

systolic_bp = st.number_input(
    "Systolic Blood Pressure",
    min_value=40.0,
    max_value=250.0,
    value=120.0
)

diastolic_bp = st.number_input(
    "Diastolic Blood Pressure",
    min_value=30.0,
    max_value=160.0,
    value=80.0
)

blood_sugar = st.number_input(
    "Blood Sugar",
    min_value=0.0,
    max_value=600.0,
    value=120.0
)

ck_mb = st.number_input(
    "CK-MB",
    min_value=0.0,
    max_value=350.0,
    value=5.0
)

troponin = st.number_input(
    "Troponin",
    min_value=0.0,
    max_value=15.0,
    value=0.1
)

# PREDICTION

if st.button("Predict Result", type="primary"):

    # CREATE INPUT DATAFRAME
   

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Heart rate": [heart_rate],
        "Systolic blood pressure": [systolic_bp],
        "Diastolic blood pressure": [diastolic_bp],
        "Blood sugar": [blood_sugar],
        "CK-MB": [ck_mb],
        "Troponin": [troponin]
    })

    
    # FORCE EXACT TRAINING FEATURE ORDER
   

    feature_order = [
        "Age",
        "Gender",
        "Heart rate",
        "Systolic blood pressure",
        "Diastolic blood pressure",
        "Blood sugar",
        "CK-MB",
        "Troponin"
    ]

    input_data = input_data[feature_order]

  
    # PREDICTION


    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    negative_probability = probabilities[0]
    positive_probability = probabilities[1]

    # DISPLAY RESULT
 

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("🔴 Prediction: POSITIVE")

    else:

        st.success("🟢 Prediction: NEGATIVE")

  
    # DISPLAY PROBABILITIES
   

    st.write(
        f"Negative Probability: "
        f"{negative_probability:.2%}"
    )

    st.write(
        f"Positive Probability: "
        f"{positive_probability:.2%}"
    )

    # DISPLAY INPUTS
   

    with st.expander("View Input Data"):

        st.dataframe(input_data)

    # DEBUG INFORMATION
   

    with st.expander("Model Information"):

        st.write(
            "Model Classes:",
            model.classes_
        )

        st.write(
            "Prediction Value:",
            prediction
        )

        st.write(
            "Raw Probabilities:",
            probabilities
        )
