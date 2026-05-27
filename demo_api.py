#!/usr/bin/env python3
"""
Demo script showing how to use the Titanic Survival Prediction API
This demonstrates how to integrate the API into other applications
"""

import requests
import json
import time

class TitanicPredictor:
    """Client class for the Titanic Survival Prediction API"""
    
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def predict_survival(self, passenger_data):
        """
        Predict survival for a passenger
        
        Args:
            passenger_data (dict): Dictionary containing passenger information
                - pclass (int): Passenger class (1, 2, or 3)
                - sex (str): Gender ('male' or 'female')
                - age (float): Age in years
                - sibsp (int): Number of siblings/spouses aboard
                - parch (int): Number of parents/children aboard
                - fare (float): Ticket fare
                - embarked (str): Port of embarkation ('S', 'C', or 'Q')
        
        Returns:
            dict: Prediction results with survival status and probabilities
        """
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                headers={'Content-Type': 'application/json'},
                json=passenger_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
    
    def get_model_info(self):
        """Get information about the trained model"""
        try:
            response = self.session.get(f"{self.base_url}/train", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get model info: {e}")
    
    def batch_predict(self, passenger_list):
        """
        Make predictions for multiple passengers
        
        Args:
            passenger_list (list): List of passenger data dictionaries
        
        Returns:
            list: List of prediction results
        """
        results = []
        for i, passenger in enumerate(passenger_list):
            try:
                result = self.predict_survival(passenger)
                result['passenger_id'] = i + 1
                results.append(result)
                print(f"Passenger {i+1}: {'✅ SURVIVED' if result['survived'] else '❌ DID NOT SURVIVE'} "
                      f"(Survival: {result['survival_probability']:.1%})")
                time.sleep(0.1)  # Small delay to be respectful to the API
            except Exception as e:
                print(f"Passenger {i+1}: Error - {e}")
                results.append({'error': str(e), 'passenger_id': i + 1})
        return results

def main():
    """Main demonstration function"""
    print("🚢 Titanic Survival Prediction API Demo")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = TitanicPredictor()
    
    # Get model information
    try:
        model_info = predictor.get_model_info()
        print(f"📊 Model Features: {', '.join(model_info['features'])}")
        print(f"🔧 Total Features: {len(model_info['features'])}")
    except Exception as e:
        print(f"❌ Failed to get model info: {e}")
        return
    
    # Example passenger profiles
    passengers = [
        {
            "name": "Rose (1st Class Female)",
            "data": {"pclass": 1, "sex": "female", "age": 17, "sibsp": 0, "parch": 0, "fare": 100.0, "embarked": "S"}
        },
        {
            "name": "Jack (3rd Class Male)",
            "data": {"pclass": 3, "sex": "male", "age": 20, "sibsp": 0, "parch": 0, "fare": 7.0, "embarked": "S"}
        },
        {
            "name": "Mrs. Astor (1st Class Female with Family)",
            "data": {"pclass": 1, "sex": "female", "age": 18, "sibsp": 0, "parch": 0, "fare": 200.0, "embarked": "C"}
        },
        {
            "name": "Steward (2nd Class Male)",
            "data": {"pclass": 2, "sex": "male", "age": 28, "sibsp": 0, "parch": 0, "fare": 15.0, "embarked": "S"}
        }
    ]
    
    print(f"\n🎭 Making predictions for {len(passengers)} famous passengers...")
    print("-" * 50)
    
    # Make individual predictions
    for passenger in passengers:
        print(f"\n👤 {passenger['name']}")
        print(f"   Data: {passenger['data']}")
        
        try:
            result = predictor.predict_survival(passenger['data'])
            
            survival_status = "✅ SURVIVED" if result['survived'] else "❌ DID NOT SURVIVE"
            survival_prob = result['survival_probability']
            death_prob = result['death_probability']
            
            print(f"   Prediction: {survival_status}")
            print(f"   Survival Probability: {survival_prob:.1%}")
            print(f"   Death Probability: {death_prob:.1%}")
            
            # Add some historical context
            if passenger['name'].startswith("Rose"):
                print("   💡 Historical Note: Rose DeWitt Bukater survived the Titanic disaster")
            elif passenger['name'].startswith("Jack"):
                print("   💡 Historical Note: Jack Dawson (fictional character) did not survive")
            elif passenger['name'].startswith("Mrs. Astor"):
                print("   💡 Historical Note: Madeleine Astor survived and gave birth to a son")
            elif passenger['name'].startswith("Steward"):
                print("   💡 Historical Note: Many crew members did not survive")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Demonstrate batch prediction
    print(f"\n📦 Batch Prediction Demo")
    print("-" * 30)
    
    batch_passengers = [
        {"pclass": 1, "sex": "female", "age": 25, "sibsp": 0, "parch": 0, "fare": 50.0, "embarked": "S"},
        {"pclass": 3, "sex": "male", "age": 30, "sibsp": 0, "parch": 0, "fare": 7.0, "embarked": "S"},
        {"pclass": 2, "sex": "female", "age": 35, "sibsp": 1, "parch": 2, "fare": 25.0, "embarked": "C"}
    ]
    
    print("Making batch predictions...")
    batch_results = predictor.batch_predict(batch_passengers)
    
    # Summary statistics
    survived_count = sum(1 for r in batch_results if r.get('survived', False))
    total_count = len([r for r in batch_results if 'error' not in r])
    
    print(f"\n📊 Batch Results Summary:")
    print(f"   Total Passengers: {total_count}")
    print(f"   Survived: {survived_count}")
    print(f"   Did Not Survive: {total_count - survived_count}")
    print(f"   Survival Rate: {survived_count/total_count:.1%}")
    
    print(f"\n🎯 API Integration Tips:")
    print(f"   - Use session objects for multiple requests")
    print(f"   - Implement proper error handling")
    print(f"   - Add timeouts to prevent hanging requests")
    print(f"   - Consider rate limiting for production use")
    print(f"   - Cache results when appropriate")

if __name__ == "__main__":
    main()

