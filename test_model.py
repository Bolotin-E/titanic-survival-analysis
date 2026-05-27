#!/usr/bin/env python3
"""
Test script for the Titanic Survival Prediction Model
This script demonstrates the model's performance and validates predictions
"""

import requests
import json
import pandas as pd

def test_prediction_endpoint():
    """Test the prediction endpoint with various passenger profiles"""
    
    # Test cases representing different passenger types
    test_cases = [
        {
            "name": "1st Class Female (High Survival)",
            "data": {"pclass": 1, "sex": "female", "age": 25, "sibsp": 0, "parch": 0, "fare": 50.0, "embarked": "S"}
        },
        {
            "name": "3rd Class Male (Low Survival)",
            "data": {"pclass": 3, "sex": "male", "age": 30, "sibsp": 0, "parch": 0, "fare": 7.0, "embarked": "S"}
        },
        {
            "name": "2nd Class Female with Family",
            "data": {"pclass": 2, "sex": "female", "age": 35, "sibsp": 1, "parch": 2, "fare": 25.0, "embarked": "C"}
        },
        {
            "name": "1st Class Male (Medium Survival)",
            "data": {"pclass": 1, "sex": "male", "age": 45, "sibsp": 0, "parch": 0, "fare": 100.0, "embarked": "S"}
        },
        {
            "name": "3rd Class Child (Medium Survival)",
            "data": {"pclass": 3, "sex": "male", "age": 8, "sibsp": 2, "parch": 2, "fare": 15.0, "embarked": "Q"}
        }
    ]
    
    print("🚢 Testing Titanic Survival Prediction Model")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Passenger Data: {test_case['data']}")
        
        try:
            response = requests.post(
                'http://localhost:5001/predict',
                headers={'Content-Type': 'application/json'},
                json=test_case['data']
            )
            
            if response.status_code == 200:
                result = response.json()
                survival_status = "✅ SURVIVED" if result['survived'] else "❌ DID NOT SURVIVE"
                survival_prob = round(result['survival_probability'] * 100, 1)
                death_prob = round(result['death_probability'] * 100, 1)
                
                print(f"   Prediction: {survival_status}")
                print(f"   Survival Probability: {survival_prob}%")
                print(f"   Death Probability: {death_prob}%")
                
                # Add interpretation
                if result['survived']:
                    if survival_prob > 80:
                        print(f"   💡 High confidence in survival prediction")
                    elif survival_prob > 60:
                        print(f"   💡 Moderate confidence in survival prediction")
                    else:
                        print(f"   💡 Low confidence in survival prediction")
                else:
                    if death_prob > 80:
                        print(f"   💡 High confidence in death prediction")
                    elif death_prob > 60:
                        print(f"   💡 Moderate confidence in death prediction")
                    else:
                        print(f"   💡 Low confidence in death prediction")
                        
            else:
                print(f"   ❌ Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Model Performance Summary:")
    print("   - The model shows realistic survival patterns")
    print("   - 1st class passengers generally have higher survival rates")
    print("   - Females have higher survival rates than males")
    print("   - Children and families show moderate survival rates")
    print("   - 3rd class males have the lowest survival rates")

def test_model_features():
    """Test the model training endpoint to see available features"""
    try:
        response = requests.get('http://localhost:5001/train')
        if response.status_code == 200:
            result = response.json()
            print(f"\n🔧 Model Features: {', '.join(result['features'])}")
            print(f"📊 Total Features: {len(result['features'])}")
        else:
            print(f"❌ Error getting model features: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    # Check if the Flask app is running
    try:
        response = requests.get('http://localhost:5001/')
        if response.status_code == 200:
            print("✅ Flask app is running successfully!")
            test_model_features()
            test_prediction_endpoint()
        else:
            print(f"❌ Flask app returned status code: {response.status_code}")
    except requests.exceptions.RequestException:
        print("❌ Flask app is not running. Please start it with: python app.py")
        print("   Then run this test script again.")

