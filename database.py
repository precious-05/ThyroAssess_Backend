"""
DATABASE.PY - MongoDB Connection Module
UPDATED FOR MongoDB Atlas Connection
100% Compatible with existing code
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, DuplicateKeyError
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# MongoDB Atlas Configuration from .env
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://alina_31:PROCODER@thyroscan.l8lmmpl.mongodb.net/health_prediction_db?retryWrites=true&w=majority&appName=thyroScan")
DB_NAME = "health_prediction_db"
COLLECTION_NAME = "thyroid_predictions"

# Connection objects
client = None
db = None
predictions_collection = None

# 1. CONNECTION INITIALIZATION WITH ERROR HANDLING
def initialize_database():
    """Initialize MongoDB Atlas connection with retry logic"""
    global client, db, predictions_collection
    
    try:
        logger.info("🔄 Attempting connection to MongoDB Atlas...")
        
        # Use the MONGO_URI from .env file
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,  # Increased timeout for Atlas
            connectTimeoutMS=10000,        # Longer connection timeout
            socketTimeoutMS=30000,         # Longer socket timeout for Atlas
            maxPoolSize=50,                # Increased pool size
            minPoolSize=10,                # Minimum pool size
            retryWrites=True,
            retryReads=True,
            tls=True,                      # Enable TLS for Atlas
            tlsAllowInvalidCertificates=False  # Strict TLS validation
        )
        
        # Test connection with a ping
        client.admin.command('ping')
        logger.info("✅ MongoDB Atlas connection successful!")
        
        # Initialize database and collection
        db = client[DB_NAME]
        predictions_collection = db[COLLECTION_NAME]
        
        # Create indexes (idempotent - safe to run multiple times)
        create_indexes()
        
        logger.info(f"✅ Database initialized: {DB_NAME}.{COLLECTION_NAME}")
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB Atlas connection failed: {e}")
        logger.info("🔄 Attempting fallback to local MongoDB...")
        
        # Fallback to local MongoDB
        try:
            client = MongoClient(
                "mongodb://localhost:27017",
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=5000
            )
            client.admin.command('ping')
            db = client[DB_NAME]
            predictions_collection = db[COLLECTION_NAME]
            create_indexes()
            logger.info("✅ Fallback to local MongoDB successful!")
            return True
        except Exception as local_error:
            logger.error(f"❌ Local MongoDB also failed: {local_error}")
            # Don't raise error - allow application to run in degraded mode
            return False

# ✅ 2. INDEX CREATION (PERFORMANCE OPTIMIZATION)
def create_indexes():
    """Create indexes for better query performance"""
    try:
        if predictions_collection is None:
            return
        
        # Single field indexes
        predictions_collection.create_index([("timestamp", -1)])  # For sorting by date
        predictions_collection.create_index([("prediction", 1)])   # For filtering by prediction type
        predictions_collection.create_index([("risk_percentage", -1)])  # For risk-based queries
        
        # Compound index for common queries
        predictions_collection.create_index([
            ("prediction", 1),
            ("timestamp", -1)
        ])
        
        # TTL index for automatic cleanup (optional - 90 days retention)
        try:
            predictions_collection.create_index(
                [("timestamp", 1)], 
                expireAfterSeconds=90 * 24 * 60 * 60  # 90 days
            )
            logger.info("✅ TTL index created for automatic cleanup")
        except:
            logger.info("ℹ️ TTL index may already exist")
        
        logger.info("✅ Database indexes created/verified")
        
    except Exception as e:
        logger.warning(f"⚠️ Index creation failed (may already exist): {e}")

# ✅ 3. SAVE PREDICTION (EXACT SAME SIGNATURE)
def save_prediction(user_data: Dict, prediction_result: str, risk_percentage: float) -> str:
    """
    Save prediction to MongoDB - 100% Compatible with existing code
    Returns: prediction_id as string
    """
    global predictions_collection
    
    # Initialize if not already done
    if predictions_collection is None:
        if not initialize_database():
            logger.warning("⚠️ Database not available, using fallback storage")
            return save_to_fallback(user_data, prediction_result, risk_percentage)
    
    try:
        # ✅ Validate input data types (critical fix)
        if not isinstance(user_data, dict):
            logger.error("❌ user_data must be a dictionary")
            return "invalid_data"
        
        if prediction_result not in ["Benign", "Malignant"]:
            logger.error(f"❌ Invalid prediction_result: {prediction_result}")
            return "invalid_prediction"
        
        if not isinstance(risk_percentage, (int, float)) or not (0 <= risk_percentage <= 100):
            logger.error(f"❌ Invalid risk_percentage: {risk_percentage}")
            return "invalid_risk"
        
        # ✅ Fix: Ensure Gender_Male is float (from your dataset analysis)
        if 'Gender_Male' in user_data:
            # Convert to float if it's integer
            if isinstance(user_data['Gender_Male'], int):
                user_data['Gender_Male'] = float(user_data['Gender_Male'])
        
        # ✅ Fix: Ensure all required fields exist
        required_fields = [
            'Age', 'Family_History', 'Radiation_Exposure', 'Iodine_Deficiency',
            'Smoking', 'Obesity', 'Diabetes', 'TSH_Level', 'T3_Level', 
            'T4_Level', 'Nodule_Size', 'Thyroid_Cancer_Risk', 'Gender_Male'
        ]
        
        for field in required_fields:
            if field not in user_data:
                logger.warning(f"⚠️ Missing field {field} in user_data")
                user_data[field] = 0  # Safe default
        
        # Create prediction record
        prediction_record = {
            "user_data": user_data,
            "prediction": prediction_result,
            "risk_percentage": float(risk_percentage),  # Ensure float
            "confidence": get_confidence_level(risk_percentage),  # Added field
            "timestamp": datetime.now(timezone.utc),  # Timezone aware
            "disease_type": "thyroid",
            "created_at": datetime.now(timezone.utc),
            "version": "1.0",
            "source": "thyroasses_ai"
        }
        
        # Insert into database
        result = predictions_collection.insert_one(prediction_record)
        
        # Generate readable ID
        prediction_id = str(result.inserted_id)
        
        logger.info(f"✅ Prediction saved successfully to MongoDB Atlas: {prediction_id}")
        
        # Also save to fallback for redundancy
        save_to_fallback(user_data, prediction_result, risk_percentage)
        
        return prediction_id
        
    except DuplicateKeyError as e:
        logger.error(f"❌ Duplicate key error: {e}")
        return "duplicate_error"
    except Exception as e:
        logger.error(f"❌ Error saving prediction to MongoDB Atlas: {e}")
        # Fallback to local storage
        return save_to_fallback(user_data, prediction_result, risk_percentage)

# ✅ 4. GET PREDICTION HISTORY (EXACT SAME SIGNATURE)
def get_prediction_history(limit: int = 20) -> List[Dict]:
    """
    Get recent predictions - 100% Compatible with existing code
    Returns: List of prediction records
    """
    global predictions_collection
    
    # Initialize if not already done
    if predictions_collection is None:
        if not initialize_database():
            logger.warning("⚠️ Database not available, returning fallback data")
            return get_fallback_history(limit)
    
    try:
        # Sanitize limit (prevent denial of service)
        limit = min(max(1, limit), 100)  # Between 1 and 100
        
        # Query with projection to exclude MongoDB internal fields
        cursor = predictions_collection.find(
            {},
            {
                "_id": 0,  # Exclude MongoDB ObjectId
                "user_data": 1,
                "prediction": 1,
                "risk_percentage": 1,
                "confidence": 1,
                "timestamp": 1,
                "disease_type": 1,
                "source": 1
            }
        ).sort("timestamp", -1).limit(limit)
        
        history = list(cursor)
        
        # ✅ Fix: Convert datetime to string for JSON serialization
        for record in history:
            if "timestamp" in record and record["timestamp"]:
                # Convert to ISO format string
                if isinstance(record["timestamp"], datetime):
                    record["timestamp"] = record["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"✅ Retrieved {len(history)} prediction records from MongoDB Atlas")
        return history
        
    except Exception as e:
        logger.error(f"❌ Error retrieving history from MongoDB Atlas: {e}")
        return get_fallback_history(limit)

# ✅ 5. NEW FUNCTION: GET STATISTICS (FOR HISTORY PAGE)
def get_statistics() -> Dict[str, Any]:
    """Get statistics for charts in history page"""
    if predictions_collection is None:
        if not initialize_database():
            return get_fallback_statistics()
    
    try:
        # Total predictions
        total = predictions_collection.count_documents({})
        
        # Malignant vs Benign counts
        malignant_count = predictions_collection.count_documents({"prediction": "Malignant"})
        benign_count = predictions_collection.count_documents({"prediction": "Benign"})
        
        # Risk distribution
        high_risk = predictions_collection.count_documents({
            "risk_percentage": {"$gte": 70}
        })
        medium_risk = predictions_collection.count_documents({
            "risk_percentage": {"$gte": 40, "$lt": 70}
        })
        low_risk = predictions_collection.count_documents({
            "risk_percentage": {"$lt": 40}
        })
        
        # Average risk
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_risk": {"$avg": "$risk_percentage"},
                    "min_risk": {"$min": "$risk_percentage"},
                    "max_risk": {"$max": "$risk_percentage"}
                }
            }
        ]
        
        result = list(predictions_collection.aggregate(pipeline))
        stats = result[0] if result else {
            "avg_risk": 0,
            "min_risk": 0,
            "max_risk": 0
        }
        
        statistics = {
            "total_predictions": total,
            "malignant_count": malignant_count,
            "benign_count": benign_count,
            "risk_distribution": {
                "high": high_risk,
                "medium": medium_risk,
                "low": low_risk
            },
            "risk_stats": {
                "average": round(float(stats.get("avg_risk", 0)), 2),
                "minimum": round(float(stats.get("min_risk", 0)), 2),
                "maximum": round(float(stats.get("max_risk", 0)), 2)
            },
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "database": "mongodb_atlas" if "atlas" in MONGO_URI.lower() else "local"
        }
        
        logger.info(f"✅ Statistics generated from MongoDB Atlas: {total} total predictions")
        return statistics
        
    except Exception as e:
        logger.error(f"❌ Error generating statistics from MongoDB Atlas: {e}")
        return get_fallback_statistics()

# ✅ 6. FALLBACK FUNCTIONS (FOR WHEN DATABASE IS UNAVAILABLE)
def save_to_fallback(user_data: Dict, prediction_result: str, risk_percentage: float) -> str:
    """Save prediction to local fallback storage"""
    try:
        import json
        from datetime import datetime
        
        # Create fallback record
        fallback_record = {
            "user_data": user_data,
            "prediction": prediction_result,
            "risk_percentage": risk_percentage,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "fallback"
        }
        
        # Load existing fallback data
        try:
            with open("fallback_predictions.json", "r") as f:
                fallback_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            fallback_data = []
        
        # Add new record
        fallback_data.append(fallback_record)
        
        # Keep only last 100 records
        if len(fallback_data) > 100:
            fallback_data = fallback_data[-100:]
        
        # Save back to file
        with open("fallback_predictions.json", "w") as f:
            json.dump(fallback_data, f, indent=2)
        
        fallback_id = f"fallback_{datetime.now().timestamp()}"
        logger.info(f"✅ Prediction saved to fallback: {fallback_id}")
        
        return fallback_id
        
    except Exception as e:
        logger.error(f"❌ Error saving to fallback: {e}")
        return f"fallback_error_{datetime.now().timestamp()}"

def get_fallback_history(limit: int = 20) -> List[Dict]:
    """Get history from fallback storage"""
    try:
        import json
        
        try:
            with open("fallback_predictions.json", "r") as f:
                fallback_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            fallback_data = []
        
        # Return limited recent records
        return fallback_data[-limit:]
        
    except Exception as e:
        logger.error(f"❌ Error reading fallback history: {e}")
        return []

def get_fallback_statistics() -> Dict[str, Any]:
    """Get statistics from fallback storage"""
    try:
        import json
        
        try:
            with open("fallback_predictions.json", "r") as f:
                fallback_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            fallback_data = []
        
        if not fallback_data:
            return {
                "total_predictions": 0,
                "malignant_count": 0,
                "benign_count": 0,
                "risk_distribution": {"high": 0, "medium": 0, "low": 0},
                "risk_stats": {"average": 0, "minimum": 0, "maximum": 0},
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "database": "fallback"
            }
        
        # Calculate statistics
        total = len(fallback_data)
        malignant_count = sum(1 for item in fallback_data if item.get("prediction") == "Malignant")
        benign_count = total - malignant_count
        
        risk_values = [item.get("risk_percentage", 0) for item in fallback_data]
        
        high_risk = sum(1 for risk in risk_values if risk >= 70)
        medium_risk = sum(1 for risk in risk_values if 40 <= risk < 70)
        low_risk = sum(1 for risk in risk_values if risk < 40)
        
        avg_risk = sum(risk_values) / total if total > 0 else 0
        
        return {
            "total_predictions": total,
            "malignant_count": malignant_count,
            "benign_count": benign_count,
            "risk_distribution": {
                "high": high_risk,
                "medium": medium_risk,
                "low": low_risk
            },
            "risk_stats": {
                "average": round(avg_risk, 2),
                "minimum": round(min(risk_values), 2) if risk_values else 0,
                "maximum": round(max(risk_values), 2) if risk_values else 0
            },
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "database": "fallback"
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating fallback statistics: {e}")
        return {
            "total_predictions": 0,
            "malignant_count": 0,
            "benign_count": 0,
            "risk_distribution": {"high": 0, "medium": 0, "low": 0},
            "risk_stats": {"average": 0, "minimum": 0, "maximum": 0},
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "database": "fallback"
        }

# ✅ 7. HELPER FUNCTION (MATCHES YOUR EXISTING CODE)
def get_confidence_level(risk_percentage: float) -> str:
    """Determine confidence level based on risk percentage"""
    if risk_percentage >= 70:
        return "High"
    elif risk_percentage >= 40:
        return "Moderate"
    else:
        return "Low"

# ✅ 8. CLEANUP FUNCTION (OPTIONAL)
def cleanup_old_records(days_to_keep: int = 90):
    """Cleanup old records (run periodically)"""
    try:
        if predictions_collection is None:
            return 0
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        result = predictions_collection.delete_many({
            "timestamp": {"$lt": cutoff_date}
        })
        
        logger.info(f"✅ Cleaned up {result.deleted_count} old records")
        return result.deleted_count
        
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
        return 0

# ✅ 9. HEALTH CHECK FUNCTION
def check_database_health() -> Dict[str, Any]:
    """Check database health and status"""
    try:
        if client is None or predictions_collection is None:
            initialize_database()
        
        if client is None:
            return {
                "status": "disconnected",
                "message": "Database not connected",
                "timestamp": datetime.now().isoformat(),
                "database_type": "none"
            }
        
        # Ping database
        client.admin.command('ping')
        
        # Get collection stats
        stats = predictions_collection.count_documents({})
        
        # Check if connected to Atlas or local
        database_type = "mongodb_atlas" if "atlas" in MONGO_URI.lower() else "local_mongodb"
        
        return {
            "status": "connected",
            "message": "Database is healthy",
            "collection_count": stats,
            "database_type": database_type,
            "connection_uri": MONGO_URI[:50] + "..." if len(MONGO_URI) > 50 else MONGO_URI,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "database_type": "unknown",
            "timestamp": datetime.now().isoformat()
        }

# ✅ 10. TEST CONNECTION FUNCTION
def test_connection():
    """Test MongoDB Atlas connection explicitly"""
    logger.info("🔍 Testing MongoDB Atlas connection...")
    logger.info(f"URI: {MONGO_URI[:30]}...")
    
    try:
        # Create a test client
        test_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000
        )
        
        # Test the connection
        test_client.admin.command('ping')
        
        # List databases (optional)
        databases = test_client.list_database_names()
        
        logger.info(f"✅ Connection successful!")
        logger.info(f"✅ Available databases: {databases[:5]}...")  # First 5 databases
        
        # Close test client
        test_client.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        return False

# ✅ 11. INITIALIZE ON IMPORT (SAFE)
try:
    logger.info("🚀 Initializing MongoDB Atlas connection for ThyroAssess AI...")
    # Try to initialize but don't crash if it fails
    if initialize_database():
        logger.info("✅ ThyroAssess AI database module ready!")
    else:
        logger.warning("⚠️ Database initialization deferred - using fallback mode")
except Exception as e:
    logger.warning(f"⚠️ Database initialization error: {e}")
    # Application will use fallback mode

# ✅ EXPORT FUNCTIONS (MAINTAIN COMPATIBILITY)
__all__ = [
    'save_prediction',
    'get_prediction_history',
    'get_statistics',
    'check_database_health',
    'test_connection',
    'cleanup_old_records'
]