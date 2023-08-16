import joblib
from flask import Flask
app = Flask(__name__)

# loading the models
diabetes_model = joblib.load("diabetes_model.sav")
heart_model = joblib.load("heart_model.sav")
parkinsons_model = joblib.load("parkinsons_model.sav")

@app.route("/api/diabetes")
def diabetes_predict():
    # input parameters
    # number of pregnancies
    # glucose level
    # blood pressure value
    # skin thickness value
    # Insulin level
    # BMI Value
    # diabetes pedigree function value
    # age

    res = diabetes_model.predict()
    return res

@app.route("/api/heart")
def diabetes_predict():
    # input parameters

    res = heart_model.predict()
    return res

@app.route("/api/parkinsons")
def diabetes_predict():
    # input parameters

    res = parkinsons_model.predict()
    return res

@app.route("/api")
def predict():
    """Call the model and get the desired output"""
    # res = test()
    res = "Hello world"
    return res

if __name__ == "__main__":
    app.run(debug=True)
