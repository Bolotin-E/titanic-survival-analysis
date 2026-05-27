#!/usr/bin/env python3
"""
Analyze fare ranges by passenger class for validation
"""

import pandas as pd
import numpy as np

def analyze_fares():
    """Analyze fare distribution by passenger class"""
    
    # Load training data
    train_data = pd.read_csv('train.csv')
    
    print('🚢 Titanic Fare Analysis by Passenger Class')
    print('=' * 60)
    
    fare_ranges = {}
    
    for pclass in [1, 2, 3]:
        class_data = train_data[train_data['Pclass'] == pclass]
        
        # Calculate fare statistics
        min_fare = class_data['Fare'].min()
        max_fare = class_data['Fare'].max()
        mean_fare = class_data['Fare'].mean()
        median_fare = class_data['Fare'].median()
        std_fare = class_data['Fare'].std()
        
        # Store for later use
        fare_ranges[pclass] = {
            'min': min_fare,
            'max': max_fare,
            'mean': mean_fare,
            'median': median_fare,
            'std': std_fare
        }
        
        print(f'🏛️  Class {pclass} ({"1st" if pclass == 1 else "2nd" if pclass == 2 else "3rd"} Class):')
        print(f'   📊 Fare Statistics:')
        print(f'      Min:     ${min_fare:8.2f}')
        print(f'      Max:     ${max_fare:8.2f}')
        print(f'      Mean:    ${mean_fare:8.2f}')
        print(f'      Median:  ${median_fare:8.2f}')
        print(f'      Std Dev: ${std_fare:8.2f}')
        print(f'   🎯 Valid Range: ${min_fare:.2f} - ${max_fare:.2f}')
        print(f'   📈 Sample Count: {len(class_data)} passengers')
        print()
    
            # Calculate reasonable validation ranges (with some flexibility)
        print('🔧 Recommended Validation Ranges (with flexibility):')
        print('=' * 60)
        
        validation_ranges = {}
        for pclass, stats in fare_ranges.items():
            # Add some flexibility to the ranges
            min_valid = max(0, stats['min'] * 0.8)  # Allow slightly below min
            max_valid = stats['max'] * 1.2  # Allow slightly above max
            min_acceptable = max(0, stats['min'] * 0.5)  # Minimum acceptable value
            
            validation_ranges[pclass] = {
                'min': min_valid,
                'max': max_valid,
                'strict_min': stats['min'],
                'strict_max': stats['max'],
                'min_acceptable': min_acceptable
            }
            
            print(f'Class {pclass}:')
            print(f'  Strict Range:  ${stats["min"]:.2f} - ${stats["max"]:.2f}')
            print(f'  Flexible Range: ${min_valid:.2f} - ${max_valid:.2f}')
            print(f'  Min Acceptable: ${min_acceptable:.2f}')
            print()
        
        return fare_ranges, validation_ranges

def generate_validation_code(fare_ranges):
    """Generate Python code for fare validation"""
    
    print('💻 Generated Validation Code:')
    print('=' * 60)
    
    print('FARE_VALIDATION_RANGES = {')
    for pclass, stats in fare_ranges.items():
        min_valid = max(0, stats['min'] * 0.8)
        max_valid = stats['max'] * 1.2
        min_acceptable = max(0, stats['min'] * 0.5)
        print(f'    {pclass}: {{')
        print(f'        "min": {min_valid:.2f},')
        print(f'        "max": {max_valid:.2f},')
        print(f'        "strict_min": {stats["min"]:.2f},')
        print(f'        "strict_max": {stats["max"]:.2f},')
        print(f'        "mean": {stats["mean"]:.2f},')
        print(f'        "min_acceptable": {min_acceptable:.2f},')
        print(f'        "description": "{"1st" if pclass == 1 else "2nd" if pclass == 2 else "3rd"} Class"')
        print(f'    }},')
    print('}')
    
    print('\n# Validation function:')
    print('def validate_fare_by_class(pclass, fare):')
    print('    """Validate fare based on passenger class"""')
    print('    if pclass not in FARE_VALIDATION_RANGES:')
    print('        return False, "Invalid passenger class"')
    print('    ')
    print('    ranges = FARE_VALIDATION_RANGES[pclass]')
    print('    ')
    print('    # Check minimum acceptable value')
    print('    if fare < ranges["min_acceptable"]:')
    print('        return False, f"Fare ${fare:.2f} is below minimum acceptable value for Class {pclass} (${ranges[\"min_acceptable\"]:.2f})"')
    print('    ')
    print('    # Check maximum value')
    print('    if fare > ranges["max"]:')
    print('        return False, f"Fare ${fare:.2f} is above maximum value for Class {pclass} (${ranges[\"max\"]:.2f})"')
    print('    ')
    print('    return True, "Fare is valid for this class"')

if __name__ == "__main__":
    fare_ranges, validation_ranges = analyze_fares()
    generate_validation_code(fare_ranges)
