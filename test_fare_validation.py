#!/usr/bin/env python3
"""
Test script for the Titanic Fare Validation Feature
This script demonstrates the new fare validation based on passenger class
"""

import requests
import json

def test_fare_ranges_endpoint():
    """Test the fare ranges endpoint"""
    
    print("💰 Testing Titanic Fare Ranges Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:5001/fare-ranges')
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Fare ranges retrieved successfully!")
            
            for pclass, info in data['fare_ranges'].items():
                print(f"\n🏛️  {info['description']}:")
                print(f"   Flexible Range: ${info['min']:.2f} - ${info['max']:.2f}")
                print(f"   Strict Range:   ${info['strict_min']:.2f} - ${info['strict_max']:.2f}")
                print(f"   Typical Fare:   ${info['mean']:.2f}")
            
            return data['fare_ranges']
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return None

def test_fare_validation():
    """Test fare validation with various scenarios"""
    
    print(f"\n🔍 Testing Fare Validation Scenarios")
    print("=" * 50)
    
    # Test cases: (pclass, fare, expected_result, description)
    test_cases = [
        (1, 100.0, True, "1st Class with valid fare"),
        (1, 600.0, False, "1st Class with fare above max"),
        (1, 50.0, True, "1st Class with typical fare"),
        (2, 25.0, True, "2nd Class with valid fare"),
        (2, 90.0, False, "2nd Class with fare above max"),
        (2, 10.0, True, "2nd Class with low fare"),
        (3, 15.0, True, "3rd Class with valid fare"),
        (3, 100.0, False, "3rd Class with fare above max"),
        (3, 5.0, True, "3rd Class with low fare"),
        (1, 0.0, True, "1st Class with minimum fare"),
        (2, 0.0, True, "2nd Class with minimum fare"),
        (3, 0.0, True, "3rd Class with minimum fare"),
    ]
    
    results = []
    
    for pclass, fare, expected_valid, description in test_cases:
        print(f"\n🧪 Testing: {description}")
        print(f"   Class: {pclass}, Fare: ${fare:.2f}")
        
        try:
            response = requests.post(
                'http://localhost:5001/predict',
                headers={'Content-Type': 'application/json'},
                json={
                    'pclass': pclass,
                    'sex': 'male',
                    'age': 30,
                    'sibsp': 0,
                    'parch': 0,
                    'fare': fare,
                    'embarked': 'S'
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                actual_valid = True
                print(f"   ✅ Prediction successful: {'Survived' if result['survived'] else 'Did not survive'}")
                print(f"   📊 Survival probability: {result['survival_probability']:.1%}")
                
            elif response.status_code == 400:
                result = response.json()
                if result.get('validation_error'):
                    actual_valid = False
                    print(f"   ❌ Validation failed: {result['error']}")
                    if 'fare_ranges' in result:
                        ranges = result['fare_ranges']
                        print(f"   📋 Valid range: ${ranges['min']:.2f} - ${ranges['max']:.2f}")
                else:
                    actual_valid = False
                    print(f"   ❌ Other error: {result['error']}")
            else:
                actual_valid = False
                print(f"   ❌ Unexpected status: {response.status_code}")
            
            # Check if validation result matches expectation
            if actual_valid == expected_valid:
                print(f"   🎯 Validation result: {'✅ PASS' if actual_valid == expected_valid else '❌ FAIL'}")
            else:
                print(f"   🎯 Validation result: ❌ FAIL (Expected: {expected_valid}, Got: {actual_valid})")
            
            results.append({
                'test_case': description,
                'pclass': pclass,
                'fare': fare,
                'expected': expected_valid,
                'actual': actual_valid,
                'passed': actual_valid == expected_valid
            })
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection Error: {e}")
            results.append({
                'test_case': description,
                'pclass': pclass,
                'fare': fare,
                'expected': expected_valid,
                'actual': None,
                'passed': False
            })
    
    return results

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    
    print(f"\n🎯 Testing Edge Cases and Boundaries")
    print("=" * 50)
    
    edge_cases = [
        (1, 614.79, True, "1st Class at flexible max boundary"),
        (1, 614.81, False, "1st Class just above flexible max"),
        (2, 88.19, True, "2nd Class at flexible max boundary"),
        (2, 88.21, False, "2nd Class just above flexible max"),
        (3, 83.45, True, "3rd Class at flexible max boundary"),
        (3, 83.47, False, "3rd Class just above flexible max"),
        (1, 512.32, True, "1st Class at strict max boundary"),
        (1, 512.34, True, "1st Class above strict max but within flexible"),
        (2, 73.49, True, "2nd Class at strict max boundary"),
        (2, 73.51, True, "2nd Class above strict max but within flexible"),
        (3, 69.54, True, "3rd Class at strict max boundary"),
        (3, 69.56, True, "3rd Class above strict max but within flexible"),
    ]
    
    edge_results = []
    
    for pclass, fare, expected_valid, description in edge_cases:
        print(f"\n🔬 Testing: {description}")
        print(f"   Class: {pclass}, Fare: ${fare:.2f}")
        
        try:
            response = requests.post(
                'http://localhost:5001/predict',
                headers={'Content-Type': 'application/json'},
                json={
                    'pclass': pclass,
                    'sex': 'male',
                    'age': 30,
                    'sibsp': 0,
                    'parch': 0,
                    'fare': fare,
                    'embarked': 'S'
                }
            )
            
            if response.status_code == 200:
                actual_valid = True
                print(f"   ✅ Accepted (within flexible range)")
            elif response.status_code == 400:
                result = response.json()
                if result.get('validation_error'):
                    actual_valid = False
                    print(f"   ❌ Rejected: {result['error']}")
                else:
                    actual_valid = False
                    print(f"   ❌ Other error: {result['error']}")
            else:
                actual_valid = False
                print(f"   ❌ Unexpected status: {response.status_code}")
            
            if actual_valid == expected_valid:
                print(f"   🎯 Result: ✅ PASS")
            else:
                print(f"   🎯 Result: ❌ FAIL (Expected: {expected_valid}, Got: {actual_valid})")
            
            edge_results.append({
                'test_case': description,
                'pclass': pclass,
                'fare': fare,
                'expected': expected_valid,
                'actual': actual_valid,
                'passed': actual_valid == expected_valid
            })
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Connection Error: {e}")
            edge_results.append({
                'test_case': description,
                'pclass': pclass,
                'fare': fare,
                'expected': expected_valid,
                'actual': None,
                'passed': False
            })
    
    return edge_results

def main():
    """Main function to run all fare validation tests"""
    
    print("🚢 Titanic Fare Validation Test Suite")
    print("=" * 60)
    
    # Check if the Flask app is running
    try:
        response = requests.get('http://localhost:5001/')
        if response.status_code == 200:
            print("✅ Flask app is running successfully!")
            
            # Test fare ranges endpoint
            fare_ranges = test_fare_ranges_endpoint()
            
            if fare_ranges:
                # Test basic validation scenarios
                basic_results = test_fare_validation()
                
                # Test edge cases
                edge_results = test_edge_cases()
                
                # Summary
                print(f"\n📊 TEST SUMMARY")
                print("=" * 60)
                
                all_results = basic_results + edge_results
                passed = sum(1 for r in all_results if r['passed'])
                total = len(all_results)
                
                print(f"Total Tests: {total}")
                print(f"Passed: {passed}")
                print(f"Failed: {total - passed}")
                print(f"Success Rate: {passed/total*100:.1f}%")
                
                if passed == total:
                    print(f"\n🎉 All fare validation tests passed!")
                else:
                    print(f"\n⚠️  Some tests failed. Check the results above.")
                
                print(f"\n💡 Fare Validation Features:")
                print(f"   ✅ Real-time fare range display")
                print(f"   ✅ Class-based fare validation")
                print(f"   ✅ Flexible and strict range enforcement")
                print(f"   ✅ User-friendly error messages")
                print(f"   ✅ Dynamic form constraints")
                
        else:
            print(f"❌ Flask app returned status code: {response.status_code}")
            
    except requests.exceptions.RequestException:
        print("❌ Flask app is not running. Please start it with: python app.py")
        print("   Then run this test script again.")

if __name__ == "__main__":
    main()

