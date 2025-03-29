
import os
import numpy as np
import json
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model_path = os.path.join('model', 'TheLastDance-Copy1.h5')
try:
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Load label mappings
labels_path = os.path.join('model', 'labels.json')
try:
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    logger.info(f"Labels loaded successfully: {labels}")
except Exception as e:
    logger.error(f"Error loading labels: {e}")
    labels = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the server is running properly."""
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    return jsonify({"status": "ok", "message": "Server is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions based on keypoints."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get keypoints from request
        data = request.json
        
        if not data or 'keypoints' not in data:
            return jsonify({"error": "No keypoints provided"}), 400
        
        keypoints = np.array(data['keypoints'])
        
        # Input shape validation (adjust based on your model's expected input)
        expected_shape = model.input_shape[1:]  # The expected shape excluding batch dimension
        
        # Reshape if needed (depends on your model)
        if len(keypoints.shape) == 1:
            # If keypoints is 1D, reshape to match model input
            # Assuming model expects [sequence_length, feature_dim]
            sequence_length = expected_shape[0] if len(expected_shape) > 1 else 1
            feature_dim = expected_shape[-1]
            
            # Reshape to match expected input
            if keypoints.size != sequence_length * feature_dim:
                return jsonify({"error": f"Invalid keypoints shape. Expected {sequence_length * feature_dim} values"}), 400
            
            keypoints = keypoints.reshape(1, sequence_length, feature_dim)
        else:
            # Add batch dimension if not present
            if len(keypoints.shape) == 2:
                keypoints = np.expand_dims(keypoints, axis=0)
        
        # Make prediction
        prediction = model.predict(keypoints)
        
        # Get the predicted class index
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Get the corresponding label
        predicted_label = labels.get(str(predicted_class), "Unknown")
        
        # Log and return the result
        logger.info(f"Prediction made: {predicted_label} (class {predicted_class})")
        return jsonify({
            "prediction": predicted_label,
            "confidence": float(prediction[0][predicted_class]),
            "class_index": int(predicted_class)
        })
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
