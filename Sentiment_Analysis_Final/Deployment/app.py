import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("Model/sentiment_model.pkl", "rb"))

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    prediction = model.predict(np.array([text]))[0]  # Reshape input
    sentiment = "Positive" if prediction == 1 else "Negative"
    return render_template("index.html", prediction_text="The sentiment of the text is {}".format(sentiment))


if __name__ == "__main__":
    flask_app.run(debug=True)


"""
NOTE FOR MYSELF:
While this web app running, it works until i enter text to analyze, but when i push to "analyze sentiment" i got an error
with name "ValueError: Expected 2D array, got scalar array instead"
Once i faced with that problem, and solve || now i am looking for to solve that problem - THIS NOT FOR MYSELF TO REMEMBER ERROR
THERE IS AN ERRORRRRRRRRRRRRR, HEY I A TALKING TO YOU

"""