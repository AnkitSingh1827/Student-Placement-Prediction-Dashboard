# Student Placement Prediction Dashboard

This repository contains a Streamlit dashboard for student placement prediction using a trained Logistic Regression model.

## Verified status

The current implementation has been verified in this workspace:

- The app successfully loads the saved model artifact `placement_model.pkl` and scaler artifact `scaler.pkl`.
- The loaded model is a `LogisticRegression` classifier with `n_features_in_ = 24` and output classes `[0, 1]`.
- A smoke test using a real row from the dataset produced a valid prediction and probability output.
- The Streamlit app starts successfully in headless mode on `http://localhost:8501`.


## Live Demo

Link: https://student-placement-prediction-dashboard-jfthzbzmqjshykoomuenvd.streamlit.app

## What is included

- `Student.ipynb` — notebook used for training and data exploration.
- `student_placement_prediction_dataset_2026.csv` — dataset used by the app.
- `placement_model.pkl` — saved trained Logistic Regression model artifact.
- `scaler.pkl` — saved `StandardScaler` artifact used for preprocessing.
- `app.py` — Streamlit dashboard entrypoint.
- `inspect_model.py` — utility to inspect the saved model and scaler metadata.
- `requirements.txt` — Python dependencies.

## How it works

The dashboard loads the dataset, encodes categorical values with `LabelEncoder`, scales the input features with `StandardScaler`, and uses the trained Logistic Regression model to predict placement outcome.

The app currently uses these input features:

- `age`
- `gender`
- `cgpa`
- `branch`
- `college_tier`
- `internships_count`
- `projects_count`
- `certifications_count`
- `coding_skill_score`
- `aptitude_score`
- `communication_skill_score`
- `logical_reasoning_score`
- `hackathons_participated`
- `github_repos`
- `linkedin_connections`
- `mock_interview_score`
- `attendance_percentage`
- `backlogs`
- `extracurricular_score`
- `leadership_score`
- `volunteer_experience`
- `sleep_hours`
- `study_hours_per_day`
- `salary_package_lpa`

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the dashboard

From the repository root:

```bash
streamlit run app.py
```

If you are using the workspace virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Notes

- The app first tries to load `placement_model.pkl` and `scaler.pkl`.
- If those artifacts are missing or not usable, the app falls back to training a new logistic regression model from the current dataset.
- The current saved model artifacts in this workspace are working and the app can run predictions successfully.

## Project structure

- `app.py` — Streamlit dashboard entrypoint.
- `inspect_model.py` — model inspection utility.
- `requirements.txt` — dependencies.
- `README.md` — documentation.
- `Student.ipynb` — notebook.
- `student_placement_prediction_dataset_2026.csv` — dataset.
- `placement_model.pkl` — saved model artifact.
- `scaler.pkl` — saved scaler artifact.

## Troubleshooting

If you encounter errors:

- Make sure you are running Python 3.10+.
- Confirm that `placement_model.pkl`, `scaler.pkl`, and `student_placement_prediction_dataset_2026.csv` are present in the repository root.
- If Streamlit fails to start, reinstall the dependencies and run the app again.
