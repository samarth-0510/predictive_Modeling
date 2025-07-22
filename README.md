# Heart Risk Prediction System

This project is a Flask-based web application that predicts the risk of heart failure using a trained Random Forest classifier.

## 🧪 Features Used
- Age, Anaemia, Creatinine Phosphokinase, Diabetes, Ejection Fraction, High Blood Pressure, Platelets, Serum Creatinine, Serum Sodium, Sex, Smoking, Time

## 🚀 Steps to Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Train the model:
```
python model_training.py
```

3. Run the Flask App:
```
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## 📁 Folder Structure
- `app.py`: Flask app
- `model_training.py`: Script to train and save model
- `model.pkl`: Trained model file
- `templates/index.html`: Web UI
- `static/style.css`: UI Styles
- `heart_failure_clinical_records_dataset.csv`: Dataset
