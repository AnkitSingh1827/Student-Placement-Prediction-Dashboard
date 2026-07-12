import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "placement_prediction_dataset_1000_students.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    df = load_dataset().copy()
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler().fit(X)
    model = None
    model_loaded = False

    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as file:
                model = pickle.load(file)
            if hasattr(model, "coef_") and getattr(model, "n_features_in_", None) == X.shape[1]:
                model_loaded = True
        except Exception:
            model = None

    if model is None or not model_loaded:
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(scaler.transform(X), y)

    return scaler, model, model_loaded


def predict_placement(input_features, scaler, model):
    values = np.array(input_features, dtype=float).reshape(1, -1)
    values_scaled = scaler.transform(values)
    prediction = model.predict(values_scaled)[0]
    proba = model.predict_proba(values_scaled)[0]
    return prediction, proba[1]


def main():
    st.set_page_config(
        page_title="Student Placement Prediction",
        page_icon=":mortar_board:",
        layout="centered",
    )

    st.title("Student Placement Prediction Dashboard")
    st.write(
        "This Streamlit app uses the same feature pipeline from the notebook and predicts whether a student is likely to be placed. "
        "Gender is encoded as Male=1 and Female=0, then the inputs are scaled before prediction."
    )

    scaler, model, model_loaded = load_model_and_scaler()
    if not model_loaded:
        st.warning(
            "The included `model.pkl` was not a fitted logistic regression model. "
            "This dashboard trains a fallback logistic regression model from the original dataset at startup."
        )
    else:
        st.success("Loaded the original saved model from `model.pkl`.")

    with st.sidebar:
        st.header("Input feature ranges")
        df = load_dataset()
        st.write(df.describe().loc[["min", "max", "mean"]].T)
        st.markdown("---")
        st.caption("Original dataset columns are used exactly as in the notebook.")

    with st.form(key="placement_form"):
        st.subheader("Student information")

        cols = st.columns(2)

        with cols[0]:
            age = st.slider("Age", min_value=20, max_value=25, value=22)
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            tenth_pct = st.slider("10th Percentage", min_value=55.0, max_value=98.0, value=75.0, step=0.1)
            twelfth_pct = st.slider("12th Percentage", min_value=55.1, max_value=97.9, value=76.0, step=0.1)
            iq = st.slider("IQ Score", min_value=80, max_value=143, value=108)
            cgpa = st.slider("CGPA", min_value=5.0, max_value=9.9, value=7.4, step=0.1)
            college_tier = st.selectbox("College Tier", [1, 2, 3], index=1)

        with cols[1]:
            communication = st.slider("Communication Skills", min_value=40, max_value=100, value=70)
            technical = st.slider("Technical Skills", min_value=35, max_value=100, value=67)
            aptitude = st.slider("Aptitude Score", min_value=40, max_value=100, value=71)
            coding = st.slider("Coding Score", min_value=30, max_value=100, value=64)
            internships = st.slider("Internships", min_value=0, max_value=3, value=0)
            projects = st.slider("Projects", min_value=1, max_value=8, value=5)
            backlogs = st.slider("Backlogs", min_value=0, max_value=3, value=0)

        submit_button = st.form_submit_button(label="Predict Placement")

    if submit_button:
        gender_encoded = 1 if gender == "Male" else 0
        feature_order = [
            age,
            gender_encoded,
            tenth_pct,
            twelfth_pct,
            iq,
            cgpa,
            college_tier,
            communication,
            technical,
            aptitude,
            coding,
            internships,
            projects,
            backlogs,
        ]

        prediction, placement_proba = predict_placement(feature_order, scaler, model)
        placement_label = "Placed" if prediction == 1 else "Not Placed"

        st.markdown("## Prediction result")
        st.metric("Placement prediction", placement_label)
        st.write(f"Placement probability: **{placement_proba:.2%}**")

        st.markdown("### Feature values used for prediction")
        st.write(
            {
                "Age": age,
                "Gender": gender,
                "10th Percentage": tenth_pct,
                "12th Percentage": twelfth_pct,
                "IQ": iq,
                "CGPA": cgpa,
                "College Tier": college_tier,
                "Communication Skills": communication,
                "Technical Skills": technical,
                "Aptitude Score": aptitude,
                "Coding Score": coding,
                "Internships": internships,
                "Projects": projects,
                "Backlogs": backlogs,
            }
        )

    st.markdown("---")
    st.caption("This dashboard preserves the notebook's original feature inputs and prediction pipeline.")


if __name__ == "__main__":
    main()
