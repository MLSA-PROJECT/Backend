import joblib
from flask import Flask
from flask import request

app = Flask(__name__)

# loading the models
diabetes_model = joblib.load("diabetes_model.sav")
heart_model = joblib.load("heart_model.sav")
parkinsons_model = joblib.load("parkinsons_model.sav")

@app.route("/api/diabetes", methods=['GET', 'POST'])
def diabetes_predict():
    data = request.data

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

@app.route("/api/heart", methods=['GET', 'POST'])
def diabetes_predict():
    # input parameters

    res = heart_model.predict()
    return res

@app.route("/api/parkinsons", methods=['GET', 'POST'])
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
