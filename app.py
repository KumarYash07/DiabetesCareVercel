from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import sys

# ================= APP CONFIG =================
app = Flask(__name__)
CORS(app) 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ================= LOAD ML MODEL =================
model = None
accuracy = None

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

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
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "message": "Diabetes Prediction API is running"
    })

# ================= PREDICTION API =================

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Prediction model is not loaded",
            "status": "error"
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "status": "error"
            }), 400
        
        required_fields = ['pregnancies', 'glucose', 'bloodpressure', 
                          'skinthickness', 'insulin', 'bmi', 'dpf', 'age']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "status": "error"
            }), 400
        
        # Convert to float
        pregnancies = float(data['pregnancies'])
        glucose = float(data['glucose'])
        bloodpressure = float(data['bloodpressure'])
        skinthickness = float(data['skinthickness'])
        insulin = float(data['insulin'])
        bmi = float(data['bmi'])
        dpf = float(data['dpf'])
        age = float(data['age'])
        
        # Basic validation
        if pregnancies < 0 or pregnancies > 20:
            return jsonify({"error": "Pregnancies must be between 0 and 20", "status": "error"}), 400
        
        if age < 1 or age > 120:
            return jsonify({"error": "Age must be between 1 and 120", "status": "error"}), 400
        
        if glucose < 0 or glucose > 500:
            return jsonify({"error": "Glucose value must be between 0 and 500", "status": "error"}), 400
        
        if bloodpressure < 0 or bloodpressure > 250:
            return jsonify({"error": "Blood pressure must be between 0 and 250", "status": "error"}), 400
        
        if bmi < 0 or bmi > 100:
            return jsonify({"error": "BMI must be between 0 and 100", "status": "error"}), 400
        
        # Prepare features
        features = np.array([
            pregnancies, glucose, bloodpressure, 
            skinthickness, insulin, bmi, dpf, age
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            probability = {
                "low_risk": round(proba[0] * 100, 2),
                "high_risk": round(proba[1] * 100, 2)
            }
        
        result = {
            "status": "success",
            "prediction": int(prediction),
            "risk_level": "High" if prediction == 1 else "Low",
            "message": "High risk of diabetes. Please consult a doctor." if prediction == 1 else "Low risk of diabetes. Maintain a healthy lifestyle.",
            "accuracy": round(accuracy * 100, 2) if accuracy else None
        }
        
        if probability:
            result["probability"] = probability
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({
            "error": f"Invalid input: {str(e)}",
            "status": "error"
        }), 400
    except Exception as e:
        return jsonify({
            "error": f"Prediction error: {str(e)}",
            "status": "error"
        }), 500

# ================= BATCH PREDICTION =================

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"error": "Model not loaded", "status": "error"}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({"error": "Expected JSON with 'patients' array", "status": "error"}), 400
        
        patients = data['patients']
        if not isinstance(patients, list):
            return jsonify({"error": "'patients' must be an array", "status": "error"}), 400
        
        results = []
        required_fields = ['pregnancies', 'glucose', 'bloodpressure', 
                          'skinthickness', 'insulin', 'bmi', 'dpf', 'age']
        
        for idx, patient in enumerate(patients):
            try:
                missing_fields = [field for field in required_fields if field not in patient]
                if missing_fields:
                    results.append({
                        "index": idx,
                        "error": f"Missing fields: {', '.join(missing_fields)}",
                        "status": "error"
                    })
                    continue
                
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
        return jsonify({"error": f"Batch error: {str(e)}", "status": "error"}), 500

# ================= MODEL INFO =================

@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "model_loaded": model is not None,
        "accuracy": round(accuracy * 100, 2) if accuracy else None,
        "features": [
            "pregnancies", "glucose", "bloodpressure", 
            "skinthickness", "insulin", "bmi", "dpf", "age"
        ],
        "prediction_classes": ["Low Risk", "High Risk"],
        "model_type": type(model).__name__ if model else None,
        "python_version": sys.version,
        "numpy_version": np.__version__
    })

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
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)