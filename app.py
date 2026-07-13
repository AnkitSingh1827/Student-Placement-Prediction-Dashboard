import joblib
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "student_placement_prediction_dataset_2026.csv"
MODEL_PATH = BASE_DIR / "placement_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
CATEGORICAL_COLUMNS = ["gender", "branch", "college_tier", "volunteer_experience"]
TARGET_COLUMN = "placement_status"
ID_COLUMN = "student_id"


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_feature_columns():
    df = load_dataset()
    return [col for col in df.columns if col not in {ID_COLUMN, TARGET_COLUMN}]


@st.cache_data(show_spinner=False)
def get_encoders():
    df = load_dataset()
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoder.fit(df[col].astype(str))
        encoders[col] = encoder
    return encoders


@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    df = load_dataset().copy()
    feature_columns = get_feature_columns()
    encoders = get_encoders()

    X = df[feature_columns].copy()
    for col in CATEGORICAL_COLUMNS:
        X[col] = encoders[col].transform(X[col].astype(str))

    y = df[TARGET_COLUMN]
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y.astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(float))

    model = None
    model_loaded = False

    if MODEL_PATH.exists() and SCALER_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            if hasattr(model, "coef_") and getattr(model, "n_features_in_", None) == len(feature_columns):
                model_loaded = True
        except Exception:
            model = None
            scaler = StandardScaler().fit(X_scaled)

    if model is None or not model_loaded:
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, y_encoded)

    return scaler, model, model_loaded, encoders, target_encoder, feature_columns


def prepare_input_frame(input_values, feature_columns, encoders):
    input_df = pd.DataFrame([input_values], columns=feature_columns)
    for col in CATEGORICAL_COLUMNS:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
            input_df[col] = encoders[col].transform(input_df[col])
    for col in feature_columns:
        if col not in CATEGORICAL_COLUMNS:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").astype(float)
    return input_df[feature_columns]


def predict_placement(input_values, scaler, model, encoders, feature_columns):
    values = prepare_input_frame(input_values, feature_columns, encoders)
    values_scaled = scaler.transform(values.astype(float))
    prediction = int(model.predict(values_scaled)[0])
    proba = float(model.predict_proba(values_scaled)[0][1])
    return prediction, proba


def main():
    st.set_page_config(
        page_title="Student Placement Prediction",
        page_icon=":mortar_board:",
        layout="centered",
    )

    st.title("Student Placement Prediction Dashboard")
    st.write(
        "This dashboard uses the same preprocessing and feature order as the notebook-trained model. "
        "Categorical values are encoded with the same LabelEncoder logic before scaling and prediction."
    )

    scaler, model, model_loaded, encoders, target_encoder, feature_columns = load_model_and_scaler()
    df = load_dataset()

    if not model_loaded:
        st.warning(
            "The saved notebook artifacts were not available, so the app trained a fallback logistic regression model from the current dataset."
        )
    else:
        st.success("Loaded the notebook-trained model and scaler artifacts.")

    with st.sidebar:
        st.header("Dataset summary")
        st.write(df[feature_columns].describe().loc[["min", "max", "mean"]].T)
        st.markdown("---")
        st.caption("The model expects the same feature columns used in the notebook training pipeline.")

    with st.form(key="placement_form"):
        st.subheader("Student details")

        input_values = {}
        cols = st.columns(2)

        with cols[0]:
            input_values["age"] = st.number_input("Age", min_value=18, max_value=30, value=int(df["age"].mean()), step=1)
            input_values["gender"] = st.selectbox("Gender", options=sorted(encoders["gender"].classes_.tolist()))
            input_values["cgpa"] = st.number_input("CGPA", min_value=4.0, max_value=10.0, value=float(df["cgpa"].mean()), step=0.01)
            input_values["branch"] = st.selectbox("Branch", options=sorted(encoders["branch"].classes_.tolist()))
            input_values["college_tier"] = st.selectbox("College Tier", options=sorted(encoders["college_tier"].classes_.tolist()))
            input_values["internships_count"] = st.number_input("Internships", min_value=0, max_value=10, value=int(df["internships_count"].mean()), step=1)
            input_values["projects_count"] = st.number_input("Projects", min_value=0, max_value=20, value=int(df["projects_count"].mean()), step=1)
            input_values["certifications_count"] = st.number_input("Certifications", min_value=0, max_value=10, value=int(df["certifications_count"].mean()), step=1)
            input_values["coding_skill_score"] = st.number_input("Coding Skill Score", min_value=0.0, max_value=100.0, value=float(df["coding_skill_score"].mean()), step=0.1)
            input_values["aptitude_score"] = st.number_input("Aptitude Score", min_value=0.0, max_value=100.0, value=float(df["aptitude_score"].mean()), step=0.1)

        with cols[1]:
            input_values["communication_skill_score"] = st.number_input("Communication Skill Score", min_value=0.0, max_value=100.0, value=float(df["communication_skill_score"].mean()), step=0.1)
            input_values["logical_reasoning_score"] = st.number_input("Logical Reasoning Score", min_value=0.0, max_value=100.0, value=float(df["logical_reasoning_score"].mean()), step=0.1)
            input_values["hackathons_participated"] = st.number_input("Hackathons Participated", min_value=0, max_value=20, value=int(df["hackathons_participated"].mean()), step=1)
            input_values["github_repos"] = st.number_input("GitHub Repos", min_value=0, max_value=50, value=int(df["github_repos"].mean()), step=1)
            input_values["linkedin_connections"] = st.number_input("LinkedIn Connections", min_value=0, max_value=2000, value=int(df["linkedin_connections"].mean()), step=1)
            input_values["mock_interview_score"] = st.number_input("Mock Interview Score", min_value=0.0, max_value=100.0, value=float(df["mock_interview_score"].mean()), step=0.1)
            input_values["attendance_percentage"] = st.number_input("Attendance %", min_value=0.0, max_value=100.0, value=float(df["attendance_percentage"].mean()), step=0.1)
            input_values["backlogs"] = st.number_input("Backlogs", min_value=0, max_value=10, value=int(df["backlogs"].mean()), step=1)
            input_values["extracurricular_score"] = st.number_input("Extracurricular Score", min_value=0.0, max_value=100.0, value=float(df["extracurricular_score"].mean()), step=0.1)
            input_values["leadership_score"] = st.number_input("Leadership Score", min_value=0.0, max_value=100.0, value=float(df["leadership_score"].mean()), step=0.1)
            input_values["volunteer_experience"] = st.selectbox("Volunteer Experience", options=sorted(encoders["volunteer_experience"].classes_.tolist()))
            input_values["sleep_hours"] = st.number_input("Sleep Hours", min_value=0.0, max_value=12.0, value=float(df["sleep_hours"].mean()), step=0.1)
            input_values["study_hours_per_day"] = st.number_input("Study Hours / Day", min_value=0.0, max_value=12.0, value=float(df["study_hours_per_day"].mean()), step=0.1)
            input_values["salary_package_lpa"] = st.number_input("Salary Package (LPA)", min_value=0.0, max_value=40.0, value=float(df["salary_package_lpa"].mean()), step=0.1)

        submit_button = st.form_submit_button(label="Predict Placement")

    if submit_button:
        prediction, placement_proba = predict_placement(input_values, scaler, model, encoders, feature_columns)
        placement_label = "Placed" if prediction == 1 else "Not Placed"

        st.markdown("## Prediction result")
        st.metric("Placement prediction", placement_label)
        st.write(f"Placement probability: **{placement_proba:.2%}")

        st.markdown("### Input values used")
        st.write(input_values)

    st.markdown("---")
    st.caption("This dashboard follows the notebook's training pipeline and saved model artifacts.")


if __name__ == "__main__":
    main()
