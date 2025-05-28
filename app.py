from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from model_utils import load_model_components
from recommender import recommend_content_filtered, predict_emotion
from fastapi.middleware.cors import CORSMiddleware

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
    include_keywords: Optional[List[str]] = None
    exclude_keywords: Optional[List[str]] = None

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
    results = recommend_content_filtered(
        user_text=request.text,
        model=model,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
        rec_db=recommendation_db,
        include_keywords=request.include_keywords,
        exclude_keywords=request.exclude_keywords,
        sample_size=5
    )
    return {
        "recommended_content": results.to_dict(orient="records")
    }

# ========== إعداد CORS ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكن تحديد دومين الـ Laravel هنا بدل *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
