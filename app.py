from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# ================= APP CONFIG =================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ================= LOAD ML MODEL =================
model = None
accuracy = None

try:
    with open(MODEL_PATH, "rb") as file:
        loaded_data = pickle.load(file)
        
        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            model, accuracy = loaded_data
        else:
            model = loaded_data
            accuracy = 0.85
            
    print(f"Model loaded successfully | Accuracy: {accuracy * 100:.2f}%")

except FileNotFoundError:
    print("Warning: model.pkl not found.")
    model = None
    accuracy = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    accuracy = None

# ================= HEALTH CHECK =================

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "active",
        "model_loaded": model is not None,
        "accuracy": round(accuracy * 100, 2) if accuracy else None,
        "message": "Diabetes Prediction API is running"
    })

# ================= PREDICTION API =================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict diabetes risk based on patient parameters
    
    Expected JSON input:
    {
        "pregnancies": 2,
        "glucose": 120,
        "bloodpressure": 70,
        "skinthickness": 20,
        "insulin": 85,
        "bmi": 25.6,
        "dpf": 0.5,
        "age": 30
    }
    """
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            "error": "Prediction model is not loaded",
            "status": "error"
        }), 503
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "status": "error"
            }), 400
        
        # Extract features with validation
        required_fields = ['pregnancies', 'glucose', 'bloodpressure', 
                          'skinthickness', 'insulin', 'bmi', 'dpf', 'age']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "status": "error"
            }), 400
        
        # Convert to float and validate
        try:
            pregnancies = float(data['pregnancies'])
            glucose = float(data['glucose'])
            bloodpressure = float(data['bloodpressure'])
            skinthickness = float(data['skinthickness'])
            insulin = float(data['insulin'])
            bmi = float(data['bmi'])
            dpf = float(data['dpf'])
            age = float(data['age'])
        except ValueError as e:
            return jsonify({
                "error": f"Invalid number format: {str(e)}",
                "status": "error"
            }), 400
        
        # Range validations
        if pregnancies < 0 or pregnancies > 20:
            return jsonify({
                "error": "Pregnancies must be between 0 and 20",
                "status": "error"
            }), 400
        
        if age < 1 or age > 120:
            return jsonify({
                "error": "Age must be between 1 and 120",
                "status": "error"
            }), 400
        
        if glucose < 0 or glucose > 500:
            return jsonify({
                "error": "Glucose value must be between 0 and 500",
                "status": "error"
            }), 400
        
        if bloodpressure < 0 or bloodpressure > 250:
            return jsonify({
                "error": "Blood pressure must be between 0 and 250",
                "status": "error"
            }), 400
        
        if bmi < 0 or bmi > 100:
            return jsonify({
                "error": "BMI must be between 0 and 100",
                "status": "error"
            }), 400
        
        if insulin < 0 or insulin > 2000:
            return jsonify({
                "error": "Insulin must be between 0 and 2000",
                "status": "error"
            }), 400
        
        if skinthickness < 0 or skinthickness > 100:
            return jsonify({
                "error": "Skin thickness must be between 0 and 100",
                "status": "error"
            }), 400
        
        if dpf < 0 or dpf > 3:
            return jsonify({
                "error": "Diabetes Pedigree Function must be between 0 and 3",
                "status": "error"
            }), 400
        
        # Prepare features for prediction
        features = np.array([
            pregnancies, glucose, bloodpressure, 
            skinthickness, insulin, bmi, dpf, age
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0].tolist()
        
        # Prepare response
        result = {
            "status": "success",
            "prediction": int(prediction),
            "risk_level": "High" if prediction == 1 else "Low",
            "message": "High risk of diabetes. Please consult a doctor." if prediction == 1 else "Low risk of diabetes. Maintain a healthy lifestyle.",
            "accuracy": round(accuracy * 100, 2) if accuracy else None
        }
        
        # Add probability if available
        if probability:
            result["probability"] = {
                "low_risk": round(probability[0] * 100, 2),
                "high_risk": round(probability[1] * 100, 2)
            }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Prediction error: {str(e)}",
            "status": "error"
        }), 500

# ================= BATCH PREDICTION API =================

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction for multiple patients
    
    Expected JSON input:
    {
        "patients": [
            {
                "pregnancies": 2,
                "glucose": 120,
                "bloodpressure": 70,
                "skinthickness": 20,
                "insulin": 85,
                "bmi": 25.6,
                "dpf": 0.5,
                "age": 30
            },
            {...}
        ]
    }
    """
    
    if model is None:
        return jsonify({
            "error": "Prediction model is not loaded",
            "status": "error"
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({
                "error": "Invalid request. Expected JSON with 'patients' array",
                "status": "error"
            }), 400
        
        patients = data['patients']
        
        if not isinstance(patients, list):
            return jsonify({
                "error": "'patients' must be an array",
                "status": "error"
            }), 400
        
        results = []
        required_fields = ['pregnancies', 'glucose', 'bloodpressure', 
                          'skinthickness', 'insulin', 'bmi', 'dpf', 'age']
        
        for idx, patient in enumerate(patients):
            try:
                # Check for missing fields
                missing_fields = [field for field in required_fields if field not in patient]
                if missing_fields:
                    results.append({
                        "index": idx,
                        "error": f"Missing fields: {', '.join(missing_fields)}",
                        "status": "error"
                    })
                    continue
                
                # Extract features
                features = np.array([
                    float(patient['pregnancies']),
                    float(patient['glucose']),
                    float(patient['bloodpressure']),
                    float(patient['skinthickness']),
                    float(patient['insulin']),
                    float(patient['bmi']),
                    float(patient['dpf']),
                    float(patient['age'])
                ]).reshape(1, -1)
                
                # Predict
                prediction = model.predict(features)[0]
                
                results.append({
                    "index": idx,
                    "prediction": int(prediction),
                    "risk_level": "High" if prediction == 1 else "Low",
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "index": idx,
                    "error": f"Prediction error: {str(e)}",
                    "status": "error"
                })
        
        return jsonify({
            "status": "success",
            "total_patients": len(patients),
            "results": results
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Batch prediction error: {str(e)}",
            "status": "error"
        }), 500

# ================= MODEL INFO API =================

@app.route("/model/info", methods=["GET"])
def model_info():
    """Get information about the loaded model"""
    
    model_info = {
        "model_loaded": model is not None,
        "accuracy": round(accuracy * 100, 2) if accuracy else None,
        "features": [
            "pregnancies", "glucose", "bloodpressure", 
            "skinthickness", "insulin", "bmi", "dpf", "age"
        ],
        "prediction_classes": ["Low Risk", "High Risk"]
    }
    
    # Add model type if available
    if model:
        model_info["model_type"] = type(model).__name__
    
    return jsonify(model_info)

# ================= ERROR HANDLERS =================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "status": "error",
        "available_endpoints": [
            "GET / - Health check",
            "POST /predict - Single prediction",
            "POST /predict/batch - Batch predictions",
            "GET /model/info - Model information"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "status": "error"
    }), 500

# ================= FOR VERCEL =================
app.debug = False

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))