from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

def extract_date_features(df):
    df = pd.DataFrame(df, columns=['dteday'])
    df['year'] = pd.to_datetime(df['dteday']).dt.year
    df['month'] = pd.to_datetime(df['dteday']).dt.month
    df['day'] = pd.to_datetime(df['dteday']).dt.day
    return df.drop(columns=['dteday'])

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Extract data from the form
            data = CustomData(
                dteday=request.form.get('dteday'),
                season=int(request.form.get('season')),
                yr=int(request.form.get('yr')),
                mnth=int(request.form.get('mnth')),
                holiday=int(request.form.get('holiday')),
                weekday=int(request.form.get('weekday')),
                workingday=int(request.form.get('workingday')),
                weathersit=int(request.form.get('weathersit')),
                temp=float(request.form.get('temp')),
                atemp=float(request.form.get('atemp')),
                hum=float(request.form.get('hum')),
                windspeed=float(request.form.get('windspeed'))
            )
            
            # Convert the data into a DataFrame
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            # Predict using the PredictPipeline class
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print("After Prediction")

            # Render the result on the home page
            return render_template('home.html', results=results[0])
        except Exception as e:
            print(f"Error: {e}")
            return render_template('home.html', error="An error occurred during prediction.")
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
