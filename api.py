import joblib
from flask import Flask
from flask import request
import json
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# loading the models
diabetes_model = joblib.load("diabetes_model.sav")
heart_model = joblib.load("heart_disease_model.sav")
parkinsons_model = joblib.load("parkinsons_model.sav")

@app.route("/api/diabetes", methods=['GET', 'POST'])
@cross_origin()
def diabetes_predict():
    data = request.data
    data = json.loads(data)

    # input parameters
    pregenancies = data["pregenancies"]
    glucose = data["glucose"]
    blood_pressure = data["blood_pressure"]
    skin_thickness = data["skin_thickness"]
    insulin = data["insulin"]
    bmi = data["bmi"]
    diabetes_pedigree_function = data["diabetes_pedigree_function"]
    age = data["age"]

    # getting the prediction from the model
    prediction = diabetes_model.predict([[pregenancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    if prediction[0] == 0:
        res = "No"
    else:
        res = "Yes"

    return res

@app.route("/api/heart", methods=['GET', 'POST'])
@cross_origin()
def heart_predict():
    data = request.data
    data = json.loads(data)

    # input parameters

    res = heart_model.predict()
    return res

@app.route("/api/parkinsons", methods=['GET', 'POST'])
@cross_origin()
def parkinsons_predict():
    data = request.data
    data = json.loads(data)

    # input parameters

    res = parkinsons_model.predict()
    return res

@app.route("/api")
@cross_origin()
def predict():
    return "Hello world"

if __name__ == "__main__":
    app.run(debug=True)
