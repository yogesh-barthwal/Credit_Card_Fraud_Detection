from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict_credit_card():
    if request.method == "GET":
        # Show input form
        return render_template("home.html")

    else:
        # Collect form inputs
        try:
            data = CustomData(
                Amount=float(request.form.get("Amount")),
                V1=float(request.form.get("V1")),
                V2=float(request.form.get("V2")),
                V3=float(request.form.get("V3")),
                V4=float(request.form.get("V4")),
                V5=float(request.form.get("V5")),
                V6=float(request.form.get("V6")),
                V7=float(request.form.get("V7")),
                V8=float(request.form.get("V8")),
                V9=float(request.form.get("V9")),
                V10=float(request.form.get("V10")),
                V11=float(request.form.get("V11")),
                V12=float(request.form.get("V12")),
                V13=float(request.form.get("V13")),
                V14=float(request.form.get("V14")),
                V15=float(request.form.get("V15")),
                V16=float(request.form.get("V16")),
                V17=float(request.form.get("V17")),
                V18=float(request.form.get("V18")),
                V19=float(request.form.get("V19")),
                V20=float(request.form.get("V20")),
                V21=float(request.form.get("V21")),
                V22=float(request.form.get("V22")),
                V23=float(request.form.get("V23")),
                V24=float(request.form.get("V24")),
                V25=float(request.form.get("V25")),
                V26=float(request.form.get("V26")),
                V27=float(request.form.get("V27")),
                V28=float(request.form.get("V28")),
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_dataframe()

            # Make prediction
            pipeline = PredictPipeline()
            result = pipeline.predict(pred_df)[0]

            prediction_text = "Fraudulent Transaction" if result == 1 else "Legitimate Transaction"

            return render_template("home.html", prediction_text=prediction_text)

        except Exception as e:
            return render_template("home.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
