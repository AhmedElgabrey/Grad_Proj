import pandas as pd
from typing import Optional, List
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from preprocess import preprocess_text

def recommend_content_by_type(
    user_text: str,
    model: BaseEstimator,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    rec_db: pd.DataFrame,
    content_types: Optional[List[str]] = None,
    top_n: int = 5
) -> pd.DataFrame:
    processed = preprocess_text(user_text)
    features = vectorizer.transform([processed])
    predicted_label_num = model.predict(features)[0]
    predicted_emotion = label_encoder.inverse_transform([predicted_label_num])[0]

    filtered = rec_db[rec_db['emotion'] == predicted_emotion]

    if content_types:
        filtered = filtered[filtered['type'].isin(content_types)]

    return filtered.sort_values(by="score", ascending=False).head(top_n)


def predict_emotion(
    text: str,
    model: BaseEstimator,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder
) -> str:
    """توقع الشعور من نص المستخدم بعد معالجته"""
    processed = preprocess_text(text)
    features = vectorizer.transform([processed])
    prediction_num = model.predict(features)[0]
    prediction_label = label_encoder.inverse_transform([prediction_num])[0]
    return prediction_label

