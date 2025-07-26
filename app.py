from flask import Flask, render_template, request
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("logistic_best_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Step 1: Read form inputs
            gender = request.form['gender']
            senior = 1 if request.form.get('SeniorCitizen') == 'on' else 0
            partner = request.form['Partner']
            dependents = request.form['Dependents']
            tenure = float(request.form['tenure'])
            phone = request.form['PhoneService']
            multiline = request.form['MultipleLines']
            online_sec = request.form['OnlineSecurity']
            online_back = request.form['OnlineBackup']
            device_protect = request.form['DeviceProtection']
            tech_support = request.form['TechSupport']
            streaming_tv = request.form['StreamingTV']
            streaming_movies = request.form['StreamingMovies']
            paperless = 1 if request.form.get('PaperlessBilling') == 'on' else 0
            monthly_charges = float(request.form['MonthlyCharges'])
            total_charges = float(request.form['TotalCharges'])

            internet = request.form['InternetService']
            contract = request.form['Contract']
            payment = request.form['PaymentMethod']

            # Step 2: Manual One-Hot Encoding
            input_data = [
                1 if gender == 'Male' else 0,
                senior,
                1 if partner == 'Yes' else 0,
                1 if dependents == 'Yes' else 0,
                tenure,
                1 if phone == 'Yes' else 0,
                1 if multiline == 'Yes' else 0,
                1 if online_sec == 'Yes' else 0,
                1 if online_back == 'Yes' else 0,
                1 if device_protect == 'Yes' else 0,
                1 if tech_support == 'Yes' else 0,
                1 if streaming_tv == 'Yes' else 0,
                1 if streaming_movies == 'Yes' else 0,
                paperless,
                monthly_charges,
                total_charges,
                # One-hot for InternetService
                1 if internet == 'DSL' else 0,
                1 if internet == 'Fiber optic' else 0,
                1 if internet == 'No' else 0,
                # One-hot for Contract
                1 if contract == 'Month-to-month' else 0,
                1 if contract == 'One year' else 0,
                1 if contract == 'Two year' else 0,
                # One-hot for PaymentMethod
                1 if payment == 'Bank transfer (automatic)' else 0,
                1 if payment == 'Credit card (automatic)' else 0,
                1 if payment == 'Electronic check' else 0,
                1 if payment == 'Mailed check' else 0
            ]

            # Step 3: Scale input
            scaled_input = scaler.transform([input_data])

            # Step 4: Predict
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]

            result = "Customer Will Churn" if prediction == 1 else "Customer Will Stay"

            return render_template("index.html", result=result, prob=round(probability * 100, 2))

        except Exception as e:
            return f"<h3>Error: {e}</h3>"

    return render_template("index.html", result=None)
if __name__ == '__main__':
    app.run(debug=True)
