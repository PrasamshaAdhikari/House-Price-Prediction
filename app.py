from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import pickle
import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.secret_key = "anything123"

# load model + scaler
model = keras.models.load_model("model.keras")
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route('/')
def index():
    result = session.pop('result', None)
    return render_template('index.html', result=result)


@app.route('/house', methods=['POST'])
def house():

    try:
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        houseage = float(request.form['houseage'])
        houserooms = float(request.form['houserooms'])
        totalbedrooms = float(request.form['totalbedrooms'])
        population = float(request.form['population'])
        households = float(request.form['households'])
        medianincome = float(request.form['medianincome'])
        oceanproximity = float(request.form['oceanproximity'])

    except ValueError:
        session['result'] = "❌ Please fill all fields correctly"
        return redirect(url_for('index'))

    features = np.array([
        longitude, latitude, houseage, houserooms,
        totalbedrooms, population, households,
        medianincome, oceanproximity
    ], dtype=float)

    features_scaled = scaler.transform([features])

    price = float(model.predict(features_scaled)[0][0])

    session['result'] = price

    return redirect(url_for('index'))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/doc')
def doc():
    return render_template('doc.html')


@app.route('/ann')
def ann():
    return render_template('ann.html')


if __name__ == "__main__":
    app.run(debug=True)