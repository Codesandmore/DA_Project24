import joblib
import numpy as np

def predict_spam(email_text, model_type):
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    model = joblib.load(f'models/{model_type}_model.pkl')
    
    # Transform the input email text
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)

    return 'Spam' if prediction[0] == 1 else 'Not Spam'
