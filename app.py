from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = CustomData(
            season=int(request.form.get("season")),
            
            mnth=int(request.form.get("mnth")),
            day=int(request.form.get("day")),
            holiday=int(request.form.get("holiday")),
            weekday=int(request.form.get("weekday")),
            workingday=int(request.form.get("workingday")),
            weathersit=int(request.form.get("weathersit")),
            temp=float(request.form.get("temp")),
            atemp=float(request.form.get("atemp")),
            hum=float(request.form.get("hum")),
            windspeed=float(request.form.get("windspeed"))
        )

        # Get the dataframe for prediction
        pred_df = data.get_data_as_data_frame()

        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        
        # Get the prediction
        prediction = predict_pipeline.predict(pred_df)
        
        return render_template("home.html", prediction=prediction[0])

    return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
