from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import os
import shap

app = Flask(__name__)

# Global variables for encoders and scalers
label_encoders = {}
scaler = None
model = None
feature_names = []
shap_explainer = None

# Fare validation ranges by passenger class
FARE_VALIDATION_RANGES = {
    1: {
        "min": 0.00,
        "max": 614.80,
        "strict_min": 0.00,
        "strict_max": 512.33,
        "mean": 84.15,
        "min_acceptable": 0.00,
        "description": "1st Class"
    },
    2: {
        "min": 0.00,
        "max": 88.20,
        "strict_min": 0.00,
        "strict_max": 73.50,
        "mean": 20.66,
        "min_acceptable": 0.00,
        "description": "2nd Class"
    },
    3: {
        "min": 0.00,
        "max": 83.46,
        "strict_min": 0.00,
        "strict_max": 69.55,
        "mean": 13.68,
        "min_acceptable": 0.00,
        "description": "3rd Class"
    },
}

def load_and_preprocess_data():
    """Load and preprocess the Titanic dataset"""
    global label_encoders, scaler, model
    
    # Load data
    train_data = pd.read_csv('train.csv')
    
    # Separate features and target
    X = train_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)
    y = train_data['Survived']
    
    # Handle missing values
    # Age: fill with median
    X = X.copy()  # Create a copy to avoid chained assignment warnings
    X.loc[:, 'Age'] = X['Age'].fillna(X['Age'].median())
    
    # Cabin: create a feature indicating if cabin is known
    X.loc[:, 'HasCabin'] = X['Cabin'].notna().astype(int)
    X = X.drop('Cabin', axis=1)
    
    # Embarked: fill with most frequent value
    X.loc[:, 'Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
    
    # Fare: fill with median
    X.loc[:, 'Fare'] = X['Fare'].fillna(X['Fare'].median())
    
    # Encode categorical variables
    categorical_features = ['Sex', 'Embarked']
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature])
        label_encoders[feature] = le
    
    # Scale numerical features
    numerical_features = ['Age', 'Fare']
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Store feature names globally
    global feature_names
    feature_names = X.columns.tolist()
    
    # Create SHAP explainer for feature importance
    global shap_explainer
    shap_explainer = shap.TreeExplainer(model)
    
    return feature_names

def validate_fare_by_class(pclass, fare):
    """Validate fare based on passenger class"""
    if pclass not in FARE_VALIDATION_RANGES:
        return False, "Invalid passenger class"
    
    ranges = FARE_VALIDATION_RANGES[pclass]
    
    # Check minimum acceptable value
    if fare < ranges["min_acceptable"]:
        return False, f"Fare ${fare:.2f} is below minimum acceptable value for Class {pclass} (${ranges['min_acceptable']:.2f})"
    
    # Check maximum value
    if fare > ranges["max"]:
        return False, f"Fare ${fare:.2f} is above maximum value for Class {pclass} (${ranges['max']:.2f})"
    
    return True, "Fare is valid for this class"

def preprocess_input(passenger_data):
    """Preprocess a single passenger's data for prediction"""
    # Create a DataFrame with the passenger data
    df = pd.DataFrame([passenger_data])
    
    # Handle missing values
    if pd.isna(df['Age'].iloc[0]):
        df['Age'] = df['Age'].fillna(30)  # Default age
    
    if pd.isna(df['Fare'].iloc[0]):
        df['Fare'] = df['Fare'].fillna(30)  # Default fare
    
    # Create HasCabin feature
    df['HasCabin'] = 0  # For new predictions, assume no cabin info
    
    # Encode categorical variables
    for feature, encoder in label_encoders.items():
        if feature in df.columns:
            # Handle unseen categories
            if df[feature].iloc[0] not in encoder.classes_:
                df[feature] = encoder.classes_[0]  # Use first class as default
            df[feature] = encoder.transform(df[feature])
    
    # Scale numerical features
    numerical_features = ['Age', 'Fare']
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    # Ensure correct column order
    expected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'HasCabin']
    df = df[expected_columns]
    
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract passenger information
        passenger_data = {
            'Pclass': int(data['pclass']),
            'Sex': data['sex'],
            'Age': float(data['age']) if data['age'] else None,
            'SibSp': int(data['sibsp']),
            'Parch': int(data['parch']),
            'Fare': float(data['fare']) if data['fare'] else None,
            'Embarked': data['embarked']
        }
        
        # Validate fare based on passenger class
        if passenger_data['Fare'] is not None:
            fare_valid, fare_message = validate_fare_by_class(
                passenger_data['Pclass'], 
                passenger_data['Fare']
            )
            if not fare_valid:
                return jsonify({
                    'error': fare_message,
                    'validation_error': True,
                    'fare_ranges': FARE_VALIDATION_RANGES[passenger_data['Pclass']]
                }), 400
        else:
            # If no fare provided, use a default value for validation
            passenger_data['Fare'] = FARE_VALIDATION_RANGES[passenger_data['Pclass']]['mean']
        
        # Preprocess the input
        processed_data = preprocess_input(passenger_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # Return results
        result = {
            'survived': bool(prediction),
            'survival_probability': float(probability[1]),
            'death_probability': float(probability[0])
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train')
def train_model():
    try:
        feature_names = load_and_preprocess_data()
        return jsonify({
            'message': 'Model trained successfully!',
            'features': feature_names
        })
    except Exception as e:
        return jsonify({'error': str(e)}, 500)

@app.route('/feature-importance')
def get_feature_importance():
    """Get feature importance and SHAP analysis"""
    try:
        if shap_explainer is None:
            return jsonify({'error': 'Model not trained yet'}, 400)
        
        # Get feature importance from Random Forest
        rf_importance = model.feature_importances_
        
        # Create feature importance dictionary
        importance_dict = {}
        for i, feature in enumerate(feature_names):
            importance_dict[feature] = {
                'importance': float(rf_importance[i]),
                'rank': 0  # Will be updated below
            }
        
        # Sort features by importance and add ranking
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1]['importance'], reverse=True)
        for rank, (feature, data) in enumerate(sorted_features):
            importance_dict[feature]['rank'] = rank + 1
        
        # Get SHAP values for a sample of training data to show feature effects
        # We'll use a small sample to avoid memory issues
        train_data = pd.read_csv('train.csv')
        X_sample = train_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)
        
        # Apply same preprocessing to sample
        X_sample['Age'].fillna(X_sample['Age'].median(), inplace=True)
        X_sample['HasCabin'] = X_sample['Cabin'].notna().astype(int)
        X_sample = X_sample.drop('Cabin', axis=1)
        X_sample['Embarked'].fillna(X_sample['Embarked'].mode()[0], inplace=True)
        X_sample['Fare'].fillna(X_sample['Fare'].median(), inplace=True)
        
        # Encode categorical variables
        for feature, encoder in label_encoders.items():
            X_sample[feature] = encoder.transform(X_sample[feature])
        
        # Scale numerical features
        numerical_features = ['Age', 'Fare']
        X_sample[numerical_features] = scaler.transform(X_sample[numerical_features])
        
        # Ensure correct column order
        X_sample = X_sample[feature_names]
        
        # Get SHAP values for a small sample
        sample_size = min(100, len(X_sample))
        X_sample_shap = X_sample.sample(n=sample_size, random_state=42)
        
        # Calculate SHAP values (simplified approach)
        try:
            shap_values = shap_explainer.shap_values(X_sample_shap)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification: use positive class (survived)
                shap_values = shap_values[1]
            
            # Convert to numpy array and calculate mean absolute values
            shap_values = np.array(shap_values)
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            
            # Add SHAP values to importance dict
            for i, feature in enumerate(feature_names):
                importance_dict[feature]['shap_importance'] = float(mean_shap_values[i])
                
        except Exception as e:
            print(f"SHAP calculation error: {e}")
            # Fallback: use Random Forest importance as SHAP importance
            for feature in feature_names:
                importance_dict[feature]['shap_importance'] = importance_dict[feature]['importance']
        
        # Create summary for the response
        most_important = sorted_features[:3]  # Top 3 features
        least_important = sorted_features[-3:]  # Bottom 3 features
        
        summary = {
            'most_important_features': [
                {
                    'feature': feature,
                    'importance': data['importance'],
                    'shap_importance': data['shap_importance'],
                    'rank': data['rank']
                }
                for feature, data in most_important
            ],
            'least_important_features': [
                {
                    'feature': feature,
                    'importance': data['importance'],
                    'shap_importance': data['shap_importance'],
                    'rank': data['rank']
                }
                for feature, data in least_important
            ],
            'all_features': importance_dict
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}, 500)

@app.route('/fare-ranges')
def get_fare_ranges():
    """Get fare ranges for each passenger class"""
    try:
        return jsonify({
            'fare_ranges': FARE_VALIDATION_RANGES,
            'message': 'Fare ranges retrieved successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}, 500)

@app.route('/feature-importance-chart')
def get_feature_importance_chart():
    """Generate a visual chart for feature importance"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Get feature importance data
        if model is None:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        # Get feature importance from the model
        feature_importance = model.feature_importances_
        
        # Create the chart
        plt.figure(figsize=(10, 6))
        
        # Sort features by importance
        sorted_indices = feature_importance.argsort()[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_importance = feature_importance[sorted_indices]
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(sorted_features)), sorted_importance, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'])
        
        # Customize the chart
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Feature Importance Score')
        plt.title('Titanic Survival Prediction - Feature Importance Analysis', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, sorted_importance)):
            plt.text(importance + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', va='center', fontweight='bold')
        
        # Add legend for color coding
        plt.figtext(0.02, 0.02, '🎯 Higher scores = More important features', 
                   fontsize=10, style='italic', color='#666')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        chart_path = 'static/feature_importance_chart.png'
        import os
        os.makedirs('static', exist_ok=True)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return jsonify({
            'chart_url': '/static/feature_importance_chart.png',
            'message': 'Feature importance chart generated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}, 500)

if __name__ == '__main__':
    # Train the model when starting the app
    print("Training the model...")
    feature_names = load_and_preprocess_data()
    print(f"Model trained with features: {feature_names}")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
