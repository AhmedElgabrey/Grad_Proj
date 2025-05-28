import joblib

def load_model_components():
    model = joblib.load("models/random_forest_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, vectorizer, label_encoder
