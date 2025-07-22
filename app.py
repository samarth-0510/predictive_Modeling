from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")  # make sure model.pkl is present

@app.route("/")
def home():
    return render_template("index.html", prediction_text="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_features = [float(request.form.get(f)) for f in [
            "age", "anaemia", "creatinine_phosphokinase", "diabetes",
            "ejection_fraction", "high_blood_pressure", "platelets",
            "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
        ]]
        prediction = model.predict([input_features])[0]
        result = "High Risk of Heart Failure" if prediction == 1 else "Low Risk of Heart Failure"
        return render_template("index.html", prediction_text=result)
    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
