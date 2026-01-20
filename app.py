from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("model.pkl")


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "admin123":
            return redirect(url_for("predict"))
        else:
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None

    if request.method == "POST":
        try:
            # Get input data from form
            input_data = np.array([[  
                float(request.form["Pregnancies"]),
                float(request.form["Glucose"]),
                float(request.form["BloodPressure"]),
                float(request.form["SkinThickness"]),
                float(request.form["Insulin"]),
                float(request.form["BMI"]),
                float(request.form["DPF"]),
                float(request.form["Age"])
            ]])

            # Predict class (0 or 1)
            pred_class = model.predict(input_data)[0]

            # Predict probability (if supported)
            if hasattr(model, "predict_proba"):
                probability = round(model.predict_proba(input_data)[0][1] * 100, 2)
            else:
                probability = 0.0

            # Prediction label
            if pred_class == 1:
                prediction = "Disease Detected"
            else:
                prediction = "No Disease"

        except Exception as e:
            prediction = "Invalid Input"
            probability = 0.0
            print("Error:", e)

    return render_template(
        "predict.html",
        prediction=prediction,
        probability=probability
    )


@app.route("/logout")
def logout():
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
