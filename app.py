from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load your trained model and preprocessor
model_path = os.path.join("artifacts", "model.pkl")
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input features from form
        input_data = []
        for i in range(1, 29):
            input_data.append(float(request.form[f"V{i}"]))
        input_data.append(float(request.form["Amount"]))

        # Convert to DataFrame for preprocessing
        columns = [f"V{i}" for i in range(1,29)] + ["Amount"]
        df = pd.DataFrame([input_data], columns=columns)

        # Apply preprocessing
        X_processed = preprocessor.transform(df)

        # Make prediction
        prediction = model.predict(X_processed)[0]  # 0 = legit, 1 = fraud

        # Render result
        return render_template("index.html", prediction=int(prediction))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
