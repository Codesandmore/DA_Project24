from flask import Flask, render_template, request
from src.predict import predict_spam

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    result_nb = predict_spam(email_text, model_type='naive_bayes')
    result_dt = predict_spam(email_text, model_type='decision_tree')
    return render_template('index.html', prediction_nb=result_nb, prediction_dt=result_dt)

if __name__ == "__main__":
    app.run(debug=True)
