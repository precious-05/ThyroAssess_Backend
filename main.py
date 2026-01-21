from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import json
import plotly.graph_objects as go
import plotly.utils
from database import save_prediction, get_prediction_history, get_statistics, check_database_health
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#------------------INITIALIZING THE FAST-API------------------------
app = FastAPI(
    title="Thyroid Disease Prediction API", 
    version="1.0",
    description="API for Thyroid Cancer Risk Prediction using Machine Learning",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware - Improved for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Load model and features
logger.info("Loading Thyroid Model...")
try:
    with open("ml_models/thyroid_model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("✅ Model loaded successfully!")
    
    # Log model details
    logger.info(f"Model Type: {type(model).__name__}")
    if hasattr(model, 'classes_'):
        logger.info(f"Model Classes: {model.classes_}")
    
except FileNotFoundError:
    logger.error("❌ Model file not found at 'ml_models/thyroid_model.pkl'")
    model = None
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None

# Load features list
try:
    with open("ml_models/features.txt", "r") as f:
        FEATURES = [line.strip() for line in f.readlines()]
    logger.info(f"✅ Features loaded: {len(FEATURES)} features")
except:
    FEATURES = ['Age', 'Family_History', 'Radiation_Exposure', 'Iodine_Deficiency',
                'Smoking', 'Obesity', 'Diabetes', 'TSH_Level', 'T3_Level', 'T4_Level',
                'Nodule_Size', 'Thyroid_Cancer_Risk', 'Gender_Male']
    logger.info(f"Using default features: {len(FEATURES)} features")

# DataTypes according to my dataset
class ThyroidData(BaseModel):
    Age: int  # Changed from float to int
    Family_History: int
    Radiation_Exposure: int
    Iodine_Deficiency: int
    Smoking: int
    Obesity: int
    Diabetes: int
    TSH_Level: float
    T3_Level: float
    T4_Level: float
    Nodule_Size: float
    Thyroid_Cancer_Risk: int
    Gender_Male: float  #------------Changed from int to float
    
    #-----------------------Validations----------------------
    @validator('Age')
    def validate_age(cls, v):
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v
    
    @validator('TSH_Level')
    def validate_tsh(cls, v):
        if v < 0 or v > 50:
            raise ValueError('TSH must be between 0 and 50')
        return v
    
    @validator('T3_Level')
    def validate_t3(cls, v):
        if v < 0 or v > 10:
            raise ValueError('T3 must be between 0 and 10')
        return v
    
    @validator('T4_Level')
    def validate_t4(cls, v):
        if v < 0 or v > 20:
            raise ValueError('T4 must be between 0 and 20')
        return v
    
    @validator('Gender_Male')
    def validate_gender(cls, v):
        # Accept both int (0/1) and float (0.0/1.0)
        if v not in [0, 1, 0.0, 1.0]:
            raise ValueError('Gender_Male must be 0 or 1')
        return float(v)  # Convert to float for consistency
    
    @validator('Thyroid_Cancer_Risk')
    def validate_cancer_risk(cls, v):
        # Tumhare dataset ke according validation
        if v not in [0, 1, 2]:
            raise ValueError('Thyroid_Cancer_Risk must be 0, 1, or 2 (based on your dataset)')
        return v

class PredictionResponse(BaseModel):
    prediction: str
    risk_percentage: float
    confidence: str
    features_importance: Dict[str, float]
    chart_data: str

# Helper Functions
def calculate_risk_percentage(prediction_prob: float) -> float:
    """Convert probability to risk percentage"""
    risk = prediction_prob * 100
    return round(risk, 2)

def get_confidence_level(risk_percentage: float) -> str:
    """Determine confidence level based on risk"""
    if risk_percentage >= 70:
        return "High"
    elif risk_percentage >= 40:
        return "Moderate"
    else:
        return "Low"

def create_risk_chart(risk_percentage: float) -> str:
    """Create Plotly gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Thyroid Cancer Risk", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "#90EE90"},  # lightgreen
                {'range': [30, 70], 'color': "#FFD700"},  # yellow
                {'range': [70, 100], 'color': "#FF6347"}  # tomato red
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_percentage
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def get_feature_importance() -> Dict[str, float]:
    """Get feature importance from model"""
    if model is None or not hasattr(model, 'coef_'):
        # Return default equal importance
        return {feature: 0.2 for feature in FEATURES[:5]}
    
    try:
        # Normalize coefficients for better interpretation
        coef = model.coef_[0]
        abs_coef = np.abs(coef)
        
        # Min-max normalization
        if abs_coef.max() > abs_coef.min():
            normalized = (abs_coef - abs_coef.min()) / (abs_coef.max() - abs_coef.min())
        else:
            normalized = abs_coef
        
        # Create importance dictionary
        importance_dict = {}
        for i, feature in enumerate(FEATURES):
            if i < len(normalized):
                importance_dict[feature] = round(float(normalized[i]), 4)
        
        # Get top 5 features
        top_features = dict(sorted(importance_dict.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:5])
        
        # Ensure sum is reasonable
        total = sum(top_features.values())
        if total > 0:
            normalized_top = {k: round(v/total, 4) for k, v in top_features.items()}
            return normalized_top
        
        return top_features
        
    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        return {"Age": 0.25, "TSH_Level": 0.20, "Nodule_Size": 0.20, 
                "T3_Level": 0.18, "Thyroid_Cancer_Risk": 0.17}

# ================= API ENDPOINTS =================

@app.get("/")
def read_root():
    """Root endpoint - API information"""
    return {
        "message": "Thyroid Disease Prediction API", 
        "status": "active",
        "version": "1.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "/predict",
            "history": "/history",
            "stats": "/stats",
            "health": "/health",
            "features": "/features"
        }
    }

@app.get("/features")
def get_features():
    """Get list of required features with data types"""
    feature_types = {
        "Age": "integer",
        "Family_History": "integer (0/1)",
        "Radiation_Exposure": "integer (0/1)",
        "Iodine_Deficiency": "integer (0/1)",
        "Smoking": "integer (0/1)",
        "Obesity": "integer (0/1)",
        "Diabetes": "integer (0/1)",
        "TSH_Level": "float",
        "T3_Level": "float",
        "T4_Level": "float",
        "Nodule_Size": "float",
        "Thyroid_Cancer_Risk": "integer (0-4)",
        "Gender_Male": "float (0.0/1.0)"
    }
    
    return {
        "features": FEATURES, 
        "count": len(FEATURES),
        "data_types": feature_types
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_thyroid(data: ThyroidData):
    """Make thyroid disease prediction"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service temporarily unavailable. Model not loaded."
        )
    
    try:
        # ✅ Ensure Gender_Male is float (frontend compatibility)
        input_dict = data.dict()
        
        # Validate all features are present
        for feature in FEATURES:
            if feature not in input_dict:
                raise ValueError(f"Missing feature: {feature}")
        
        # Convert to DataFrame with correct feature order
        df = pd.DataFrame([input_dict])[FEATURES]
        
        # Make prediction
        prediction_proba = model.predict_proba(df)
        prediction_class = model.predict(df)
        
        # Get probability of malignant (class 1)
        malignant_prob = float(prediction_proba[0][1])
        
        # Calculate risk
        risk_percentage = calculate_risk_percentage(malignant_prob)
        
        # Determine result
        prediction_result = "Malignant" if prediction_class[0] == 1 else "Benign"
        
        # Get confidence
        confidence = get_confidence_level(risk_percentage)
        
        # Get feature importance
        features_importance = get_feature_importance()
        
        # Create chart
        chart_data = create_risk_chart(risk_percentage)
        
        # Save to database
        try:
            save_prediction(input_dict, prediction_result, risk_percentage)
            logger.info(f"✅ Prediction saved: {prediction_result}, Risk: {risk_percentage}%")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save prediction: {e}")
            # Continue even if save fails
        
        return {
            "prediction": prediction_result,
            "risk_percentage": risk_percentage,
            "confidence": confidence,
            "features_importance": features_importance,
            "chart_data": chart_data
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(e)}"
        )

@app.get("/history")
def get_history(limit: int = 20):
    """Get prediction history"""
    try:
        # Validate limit parameter
        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        history = get_prediction_history(limit)
        
        # Add calculated fields for frontend
        for record in history:
            # Add confidence if missing
            if 'confidence' not in record and 'risk_percentage' in record:
                record['confidence'] = get_confidence_level(record['risk_percentage'])
            
            # Add patient info string for display
            if 'user_data' in record:
                age = record['user_data'].get('Age', 'N/A')
                gender = "Male" if record['user_data'].get('Gender_Male', 0) == 1 else "Female"
                record['patient_info'] = f"{age} yrs, {gender}"
        
        return {
            "history": history, 
            "count": len(history),
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Database error: {str(e)}"
        )

# ✅ NEW ENDPOINT: Statistics for history page
@app.get("/stats")
def get_stats():
    """Get statistics for history page charts"""
    try:
        statistics = get_statistics()
        
        # Add API metadata
        statistics.update({
            "api_version": "1.0",
            "model_accuracy": 0.8293,  # From your model metrics
            "model_type": "Logistic Regression",
            "timestamp": datetime.now().isoformat()
        })
        
        return statistics
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        
        # Return fallback statistics
        return {
            "total_predictions": 0,
            "malignant_count": 0,
            "benign_count": 0,
            "risk_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "risk_stats": {
                "average": 0,
                "minimum": 0,
                "maximum": 0
            },
            "last_updated": datetime.now().isoformat(),
            "api_version": "1.0",
            "model_accuracy": 0.8293,
            "model_type": "Logistic Regression",
            "note": "Using fallback statistics"
        }

@app.get("/health")
def health_check():
    """Health check endpoint with detailed status"""
    db_health = check_database_health()
    
    return {
        "status": "healthy" if model is not None else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "loaded": model is not None,
            "type": type(model).__name__ if model else None,
            "features_count": len(FEATURES) if model else 0
        },
        "database": db_health,
        "endpoints_available": [
            "/predict",
            "/history", 
            "/stats",
            "/health",
            "/features"
        ]
    }

# ✅ NEW ENDPOINT: Model information
@app.get("/model/info")
def model_info():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": type(model).__name__,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "training_accuracy": 0.8293,  # From your metrics
        "confusion_matrix": [[30767, 1873], [5388, 4511]],
        "classification_report": {
            "benign": {"precision": 0.850975, "recall": 0.942616, "f1_score": 0.894455},
            "malignant": {"precision": 0.706610, "recall": 0.455703, "f1_score": 0.554075}
        },
        "loaded_at": datetime.now().isoformat()
    }
    
    if hasattr(model, 'classes_'):
        info['classes'] = model.classes_.tolist()
    
    if hasattr(model, 'coef_'):
        info['coefficients'] = {feature: float(coef) for feature, coef in zip(FEATURES, model.coef_[0])}
    
    return info

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Thyroid Prediction API...")
    logger.info(f"📊 Model Status: {'Loaded' if model else 'Not Loaded'}")
    logger.info(f"🔌 Database Health: {check_database_health()}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=True  # Set to False in production
    )