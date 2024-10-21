import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from src.preprocess import load_data, preprocess_data

def train_models():
    # Load the data
    data = load_data('data/spam_data.csv')
    
    # Preprocess data and balance it
    data = preprocess_data(data)
    
    # Split the data into features and labels
    X = data['v2']  # all rows, 'text' column
    y = data['v1']  # all rows, 'label' column

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Transform the test data
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    # Train Decision Tree model
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train_tfidf, y_train)

    # Save the models
    joblib.dump(nb_model, 'models/naive_bayes_model.pkl')
    joblib.dump(dt_model, 'models/decision_tree_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

    # Calculate accuracies
    nb_accuracy = nb_model.score(X_test_tfidf, y_test)
    dt_accuracy = dt_model.score(X_test_tfidf, y_test)

    return nb_accuracy, dt_accuracy

if __name__ == "__main__":
    nb_acc, dt_acc = train_models()
    print(f"Naive Bayes Accuracy: {nb_acc}")
    print(f"Decision Tree Accuracy: {dt_acc}")
