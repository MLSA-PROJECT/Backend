from flask import Flask

app = Flask(__name__)

# from dummy import test

# @app.route("/")
# def home():
#     return "Hello, World!"

@app.route("/api")
def predict():
    """Call the model and get the desired output"""
    # res = test()
    res = "Hello world"
    return res

if __name__ == "__main__":
    app.run(debug=True)
