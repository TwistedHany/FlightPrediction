from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for model and encoders
model = None
encoders = {}
feature_names = []
model_accuracy = 0

def load_and_train_model():
    """Load dataset and train the model"""
    global model, encoders, feature_names, model_accuracy
    
    print("Loading dataset...")
    df = pd.read_csv('flights_sample_3m.csv')
    
    print("Available columns:", df.columns.tolist())
    
    # Extract date features from FL_DATE
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    df['MONTH'] = df['FL_DATE'].dt.month
    df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek + 1  # 1=Monday, 7=Sunday
    df['DAY_OF_MONTH'] = df['FL_DATE'].dt.day
    
    # Use AIRLINE_CODE as carrier (or AIRLINE if CODE doesn't exist)
    if 'AIRLINE_CODE' in df.columns:
        df['OP_CARRIER'] = df['AIRLINE_CODE']
    else:
        df['OP_CARRIER'] = df['AIRLINE']
    
    # Select features
    features = ['MONTH', 'DAY_OF_WEEK', 'DAY_OF_MONTH', 'OP_CARRIER', 
                'ORIGIN', 'DEST', 'DEP_TIME', 'DISTANCE']
    
    # Create delay indicator: delay > 15 minutes
    df['DELAY'] = (df['DEP_DELAY'] > 15).astype(int)
    target = 'DELAY'
    
    print(f"Using target column: {target}")
    
    # Remove rows with missing values
    df = df[features + [target]].dropna()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Delay rate: {df[target].mean()*100:.2f}%")
    
    # Encode categorical variables
    categorical_features = ['OP_CARRIER', 'ORIGIN', 'DEST']
    
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Prepare features and target
    X = df[features]
    y = df[target]
    feature_names = features
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Calculate model accuracy
    model_accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {model_accuracy*100:.2f}%")
    
    # Save model + encoders + features + accuracy
    with open('flight_delay_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'encoders': encoders,
            'features': feature_names,
            'accuracy': model_accuracy
        }, f)
    
    print("Model training complete!")

def load_trained_model():
    """Load pre-trained model from file"""
    global model, encoders, feature_names, model_accuracy
    
    if os.path.exists('flight_delay_model.pkl'):
        print("Loading pre-trained model...")
        with open('flight_delay_model.pkl', 'rb') as f:
            data = pickle.load(f)
            model = data.get('model')
            encoders = data.get('encoders', {})
            feature_names = data.get('features', [])
            model_accuracy = data.get('accuracy', 0)  # <-- added line to load accuracy
        print(f"Model loaded successfully! (accuracy={model_accuracy*100:.2f}%)")
        return True
    return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict flight delay probability"""
    try:
        data = request.json
        
        # Extract features from request
        features_dict = {
            'MONTH': int(data.get('MONTH')),
            'DAY_OF_WEEK': int(data.get('DAY_OF_WEEK')),
            'DAY_OF_MONTH': int(data.get('DAY_OF_MONTH')),
            'OP_CARRIER': data.get('OP_CARRIER'),
            'ORIGIN': data.get('ORIGIN'),
            'DEST': data.get('DEST'),
            'DEP_TIME': int(data.get('DEP_TIME')),
            'DISTANCE': float(data.get('DISTANCE'))
        }
        
        # Encode categorical features
        for col in ['OP_CARRIER', 'ORIGIN', 'DEST']:
            if col in encoders:
                try:
                    features_dict[col] = encoders[col].transform([features_dict[col]])[0]
                except ValueError:
                    # Unknown category, use most common value
                    features_dict[col] = 0
        
        # Create feature array in correct order
        feature_array = np.array([[features_dict[f] for f in feature_names]])
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0]
        delay_probability = probability[1] * 100  # Probability of delay
        
        # Get feature importance
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        # Determine risk level
        if delay_probability >= 60:
            risk = 'High'
        elif delay_probability >= 30:
            risk = 'Medium'
        else:
            risk = 'Low'
        
        return jsonify({
            'prediction': int(prediction),
            'delay_probability': round(delay_probability, 1),
            'on_time_probability': round(probability[0] * 100, 1),
            'risk': risk,
            'feature_importance': feature_importance,
            'model_info': {
                'model_type': 'Random Forest',
                'confidence': round(max(probability), 3)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/train', methods=['POST'])
def train():
    """Endpoint to trigger model training"""
    global model_accuracy
    try:
        load_and_train_model()
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'accuracy': f"{model_accuracy*100:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 404
    
    return jsonify({
        'features': feature_names,
        'feature_importance': dict(zip(feature_names, model.feature_importances_)),
        'n_estimators': model.n_estimators,
        'model_type': 'Random Forest Classifier'
    })

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get model analytics and statistics"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 404
    
    try:
        # Get feature importance
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate model metrics
        return jsonify({
            'model_metrics': {
                'accuracy': round(model_accuracy * 100, 2),
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth,
                'n_features': len(feature_names)
            },
            'feature_importance': {
                name: round(importance * 100, 2) 
                for name, importance in sorted_features
            },
            'top_features': [
                {
                    'name': name.replace('_', ' ').title(),
                    'importance': round(importance * 100, 2)
                }
                for name, importance in sorted_features
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not load_trained_model():
        print("No pre-trained model found. Training new model...")
        load_and_train_model()
    
    print("\nStarting Flask API server...")
    print("API will be available at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
