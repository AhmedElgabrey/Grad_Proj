import pandas as pd
from typing import Optional, List
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from preprocess import preprocess_text


# خريطة من الشعور الحالي إلى المشاعر المستهدفة للتوصية
mood_map = {
    'sadness': ['joy', 'calm', 'amusement'],
    'anger': ['calm', 'joy'],
    'fear': ['calm', 'confidence'],
    'joy': ['joy'],
    'surprise': ['calm', 'joy'],
    'love': ['joy', 'love'],
    'misery': ['hope', 'calm', 'joy'],
    'gratitude': ['joy', 'love'],
    'disgust': ['calm', 'neutral']
    # يمكنك إضافة المزيد حسب الحاجة
}


def predict_emotion(
    text: str,
    model: BaseEstimator,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder
) -> str:
    """
    Predict emotion label for a given text.

    Args:
        text (str): Input raw text.
        model: Trained classification model.
        vectorizer: TF-IDF vectorizer.
        label_encoder: LabelEncoder to decode predicted labels.

    Returns:
        str: Predicted emotion label.
    """
    processed = preprocess_text(text)
    vect_text = vectorizer.transform([processed])
    prediction = model.predict(vect_text)
    return label_encoder.inverse_transform(prediction)[0]


def filter_by_keywords(
    df: pd.DataFrame,
    include_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter content based on included or excluded keywords.

    Args:
        df (pd.DataFrame): DataFrame with content.
        include_keywords (List[str], optional): Keywords that must be included.
        exclude_keywords (List[str], optional): Keywords to exclude.

    Returns:
        pd.DataFrame: Filtered content DataFrame.
    """
    if include_keywords:
        pattern = '|'.join(include_keywords)
        df = df[df['text'].str.contains(pattern, case=False, na=False)]

    if exclude_keywords:
        pattern = '|'.join(exclude_keywords)
        df = df[~df['text'].str.contains(pattern, case=False, na=False)]

    return df


def recommend_content_filtered(
    user_text: str,
    model: BaseEstimator,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    rec_db: pd.DataFrame,
    include_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
    sample_size: int = 3
) -> pd.DataFrame:
    """
    Recommend content filtered by predicted emotion and optional keywords.

    Args:
        user_text (str): User input text.
        model: Trained classification model.
        vectorizer: TF-IDF vectorizer.
        label_encoder: LabelEncoder to decode predicted labels.
        rec_db (pd.DataFrame): Recommendation database with columns ['text', 'emotion', ...].
        include_keywords (List[str], optional): Keywords to include.
        exclude_keywords (List[str], optional): Keywords to exclude.
        sample_size (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: Recommended content.
    """
    user_emotion = predict_emotion(user_text, model, vectorizer, label_encoder)
    print(f"🔍 Detected Emotion: {user_emotion}")

    target_emotions = mood_map.get(user_emotion, ['joy', 'calm', 'confidence'])
    recommended = rec_db[rec_db['emotion'].isin(target_emotions)]

    filtered = filter_by_keywords(recommended, include_keywords, exclude_keywords)

    if filtered.empty:
        print("⚠️ No content matches filter criteria. Returning unfiltered recommendations.")
        return recommended[['text', 'emotion']].sample(min(sample_size, len(recommended)))

    return filtered[['text', 'emotion']].sample(min(sample_size, len(filtered)))
