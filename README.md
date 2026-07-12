# Student Placement Prediction Dashboard

This repository contains a Streamlit dashboard for the student placement prediction model from `Student.ipynb`.

## What is included

- `Student.ipynb` — original notebook containing the dataset exploration and model training steps.
- `placement_prediction_dataset_1000_students.csv` — dataset used by the model.
- `model.pkl` — saved LogisticRegression model from the notebook.
- `app.py` — Streamlit application that loads the saved model and uses the same preprocessing pipeline.
- `requirements.txt` — dependency file for installing the required Python packages.

## How it works

The app uses the same model inputs as the notebook:

- Age
- Gender (Male/Female)
- 10th Percentage
- 12th Percentage
- IQ
- CGPA
- College Tier
- Communication Skills
- Technical Skills
- Aptitude Score
- Coding Score
- Internships
- Projects
- Backlogs

The app encodes `Gender` as `Male=1` and `Female=0`, scales the input features using `StandardScaler`, and then predicts placement with a logistic regression model.

## Setup

1. Create a Python environment and activate it.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the dashboard

From the repository root:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal.

## Notes

- The notebook model is preserved and not modified.
- `app.py` tries to load `model.pkl` first. If the pickle is not a fitted model, the app will train a fallback logistic regression model using the dataset and the same preprocessing pipeline.
- This keeps the app usable even if the saved `model.pkl` cannot be used for prediction directly.

## Project structure

- `app.py` — Streamlit dashboard entrypoint.
- `requirements.txt` — dependencies.
- `README.md` — this documentation.
- `Student.ipynb` — original notebook.
- `placement_prediction_dataset_1000_students.csv` — dataset.
- `model.pkl` — saved model artifact.

## Troubleshooting

If you encounter errors:

- Make sure you are running Python 3.10+.
- Confirm `placement_prediction_dataset_1000_students.csv` and `model.pkl` are present in the same folder as `app.py`.
- If Streamlit fails to load, check the package versions in `requirements.txt`.
