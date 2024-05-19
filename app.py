from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import plotly.express as px
import os
import joblib
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load models
with open('models/logistic_regression_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

with open('models/deployment_plan.pkl', 'rb') as file:
    deployment_plan = pickle.load(file)
centroids = {key: value['cluster_centroid'] for key, value in deployment_plan.items()}

def load_crime_data():
    pickle_path = 'models/predicted_crime.pkl'
    if os.path.exists(pickle_path):
        try:
            data = joblib.load(pickle_path)
        except (ModuleNotFoundError, ImportError):
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
        if 'predicted_df_2025' in data and 'predicted_df_2026' in data:
            return data
        else:
            raise KeyError("Expected keys ('predicted_df_2025', 'predicted_df_2026') not found in data.")
    else:
        raise FileNotFoundError("Pickle file 'predicted_crime.pkl' not found.")
try:
    crime_data = load_crime_data()
except (FileNotFoundError, KeyError, ImportError) as e:
    crime_data = None
    print(f"Error loading data: {e}")

def load_arima_model():
    pickle_path = "models/arima_model.pkl"
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predicted_status = ""
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form['sex']
        present_city = request.form['present_city']
        present_state = request.form['present_state']
        sex_female = 1 if sex == 'FEMALE' else 0
        sex_male = 1 if sex == 'MALE' else 0
        city_bengaluru = 1 if present_city == 'Bengaluru City' else 0
        state_karnataka = 1 if present_state == 'Karnataka' else 0
        input_data = pd.DataFrame({
            'age': [age],
            'Sex_FEMALE': [sex_female],
            'Sex_MALE': [sex_male],
            'PresentCity_Bengaluru City': [city_bengaluru],
            'PresentState_Karnataka': [state_karnataka]
        })
        prediction = logistic_model.predict(input_data)
        predicted_status = prediction[0]
    return render_template('behavior.html', predicted_status=predicted_status)

@app.route('/cluster', methods=['GET'])
def cluster():
    return render_template('cluster.html')

@app.route('/get-closest-cluster', methods=['GET'])
def get_closest_cluster():
    latitude = float(request.args.get('latitude'))
    longitude = float(request.args.get('longitude'))
    closest_cluster = min(
        centroids.keys(),
        key=lambda c: geodesic(
            (latitude, longitude),
            (centroids[c]['Latitude'], centroids[c]['Longitude'])
        ).km
    )
    cluster_data = deployment_plan[closest_cluster]
    return jsonify({
        'closest_cluster': closest_cluster,
        'cluster_centroid': cluster_data['cluster_centroid'],
        'number_of_incidents': cluster_data['number_of_incidents'],
        'suggestions': cluster_data['suggestions'],
    }), 200

@app.route('/crime-prediction', methods=['GET', 'POST'])
def crime_prediction():
    year_selected = request.form.get('year')
    graph_html = None
    error = None
    if year_selected:
        if crime_data is None:
            error = "Data could not be loaded."
        else:
            year_selected = int(year_selected)
            if year_selected == 2025:
                predicted_df = crime_data['predicted_df_2025']
                start_date = '2025-01-01'
                steps = 12
            elif year_selected == 2026:
                predicted_df = crime_data['predicted_df_2026']
                start_date = '2026-01-01'
                steps = 24
            else:
                return render_template('crime_prediction.html', graph=None, error="Invalid year selected")
            predicted_df_long = predicted_df.melt(
                id_vars='CrimeGroup_Name', var_name='Month', value_name='Predicted_Count'
            )
            fig = px.line(
                predicted_df_long,
                x='Month', y='Predicted_Count', color='CrimeGroup_Name',
                title=f'Top 5 Predicted Crimes for Each Month in {year_selected}'
            )
            graph_html = fig.to_html(full_html=False)
    return render_template('crime_prediction.html', graph=graph_html, error=error)

@app.route('/future-crime-forecast', methods=['GET', 'POST'])
def future_crime_forecast():
    future_steps = 12
    forecast_plot = None
    forecast_results = None
    if request.method == 'POST':
        try:
            model_fit = load_arima_model()
            if request.form.get('future_steps'):
                future_steps = int(request.form.get('future_steps'))
            start_date_input = request.form.get('start_date', "2024-03-02")
            start_date = pd.to_datetime(start_date_input)
            future_forecast = model_fit.forecast(steps=future_steps)
            future_dates = pd.date_range(start=start_date, periods=future_steps, freq='M')
            plt.figure(figsize=(10, 6))
            plt.plot(future_dates, future_forecast, label="Forecast")
            plt.legend()
            plt.title("Future Crime Forecast")
            plt.xlabel("Date")
            plt.ylabel("Predicted Crime Count")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            forecast_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
            forecast_results = future_forecast.tolist()
            plt.close()
        except Exception as e:
            print(f"Error: {e}")
            forecast_plot = None
            forecast_results = None
    return render_template('future_crime.html', forecast_plot=forecast_plot, forecast_results=forecast_results, future_steps=future_steps)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
