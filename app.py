from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from model_utils import load_model_components
from recommender import recommend_content_by_type, predict_emotion

# ========== تحميل النموذج والمعالجات ==========
model, vectorizer, label_encoder = load_model_components()

# تحميل قاعدة بيانات التوصية
recommendation_db = pd.read_csv("recommendation_database.csv")

# ========== إنشاء تطبيق FastAPI ==========
app = FastAPI(
    title="Emotion-Based Content Recommendation API",
    description="تحليل مشاعر النصوص وتقديم توصيات محتوى بناءً على الحالة الشعورية",
    version="1.0.0"
)

# ========== تعريف المدخلات باستخدام Pydantic ==========
class UserTextRequest(BaseModel):
    text: str
    content_types: Optional[List[str]] = None

# ========== نقطة اختبار ==========
@app.get("/")
def root():
    return {"message": "🎯 Emotion-based Recommendation API is running!"}

# ========== نقطة تحليل المشاعر ==========
@app.post("/predict-emotion")
def api_predict_emotion(request: UserTextRequest):
    emotion = predict_emotion(request.text, model, vectorizer, label_encoder)
    return {"predicted_emotion": emotion}

# ========== نقطة تقديم التوصيات ==========
@app.post("/recommend")
def api_recommend_content(request: UserTextRequest):
    results = recommend_content_by_type(
        user_text=request.text,
        model=model,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
        rec_db=recommendation_db,
        content_types=request.content_types
    )
    return {
        "recommended_content": results.to_dict(orient="records")
    }
