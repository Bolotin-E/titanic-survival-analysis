#!/usr/bin/env python3
"""
Test script for the Titanic Feature Importance Analysis
This script demonstrates the new SHAP-based feature importance functionality
"""

import requests
import json
import matplotlib.pyplot as plt
import numpy as np

def test_feature_importance():
    """Test the feature importance endpoint"""
    
    print("🔍 Testing Titanic Feature Importance Analysis")
    print("=" * 60)
    
    try:
        # Get feature importance data
        response = requests.get('http://localhost:5001/feature-importance')
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Feature importance analysis successful!")
            print(f"📊 Total features analyzed: {len(data['all_features'])}")
            
            # Display most important features
            print(f"\n🏆 TOP 3 MOST IMPORTANT FEATURES:")
            print("-" * 40)
            for feature in data['most_important_features']:
                print(f"#{feature['rank']}: {feature['feature']}")
                print(f"   Random Forest Importance: {feature['importance']:.4f} ({feature['importance']*100:.1f}%)")
                print(f"   SHAP Impact: {feature['shap_importance']:.4f} ({feature['shap_importance']*100:.1f}%)")
                print()
            
            # Display least important features
            print(f"📉 TOP 3 LEAST IMPORTANT FEATURES:")
            print("-" * 40)
            for feature in data['least_important_features']:
                print(f"#{feature['rank']}: {feature['feature']}")
                print(f"   Random Forest Importance: {feature['importance']:.4f} ({feature['importance']*100:.1f}%)")
                print(f"   SHAP Impact: {feature['shap_importance']:.4f} ({feature['shap_importance']*100:.1f}%)")
                print()
            
            # Display complete ranking
            print(f"📋 COMPLETE FEATURE RANKING:")
            print("-" * 40)
            
            # Sort all features by importance
            all_features = list(data['all_features'].items())
            all_features.sort(key=lambda x: x[1]['importance'], reverse=True)
            
            for rank, (feature_name, feature_data) in enumerate(all_features, 1):
                importance_pct = feature_data['importance'] * 100
                shap_pct = feature_data['shap_importance'] * 100
                
                # Add emoji based on rank
                if rank == 1:
                    emoji = "🥇"
                elif rank == 2:
                    emoji = "🥈"
                elif rank == 3:
                    emoji = "🥉"
                elif rank <= 5:
                    emoji = "🏅"
                elif rank >= len(all_features) - 2:
                    emoji = "📉"
                else:
                    emoji = "📊"
                
                print(f"{emoji} #{rank:2d}: {feature_name:12} | RF: {importance_pct:5.1f}% | SHAP: {shap_pct:5.1f}%")
            
            # Analysis insights
            print(f"\n💡 ANALYSIS INSIGHTS:")
            print("-" * 40)
            
            top_feature = data['most_important_features'][0]
            bottom_feature = data['least_important_features'][-1]
            
            print(f"• Most influential feature: '{top_feature['feature']}' with {top_feature['importance']*100:.1f}% importance")
            print(f"• Least influential feature: '{bottom_feature['feature']}' with {bottom_feature['importance']*100:.1f}% importance")
            
            # Calculate importance spread
            importance_values = [f['importance'] for f in data['all_features'].values()]
            importance_spread = max(importance_values) - min(importance_values)
            print(f"• Importance spread: {importance_spread:.4f} ({importance_spread*100:.1f}%)")
            
            # Check for feature importance patterns
            high_importance_features = [f for f in data['all_features'].values() if f['importance'] > 0.1]
            low_importance_features = [f for f in data['all_features'].values() if f['importance'] < 0.05]
            
            print(f"• High importance features (>10%): {len(high_importance_features)}")
            print(f"• Low importance features (<5%): {len(low_importance_features)}")
            
            # SHAP vs Random Forest comparison
            print(f"\n🔬 SHAP vs RANDOM FOREST COMPARISON:")
            print("-" * 40)
            
            shap_values = [f['shap_importance'] for f in data['all_features'].values()]
            rf_values = [f['importance'] for f in data['all_features'].values()]
            
            # Calculate correlation between SHAP and RF importance
            correlation = np.corrcoef(shap_values, rf_values)[0, 1]
            print(f"• Correlation between SHAP and RF importance: {correlation:.3f}")
            
            if correlation > 0.8:
                print("  ✅ Strong agreement between SHAP and Random Forest importance")
            elif correlation > 0.6:
                print("  ⚠️  Moderate agreement between SHAP and Random Forest importance")
            else:
                print("  ❌ Weak agreement between SHAP and Random Forest importance")
            
            return data
            
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return None

def create_feature_importance_chart(data):
    """Create a visual chart of feature importance"""
    try:
        # Extract data for plotting
        features = list(data['all_features'].keys())
        rf_importance = [data['all_features'][f]['importance'] for f in features]
        shap_importance = [data['all_features'][f]['shap_importance'] for f in features]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Random Forest Importance
        bars1 = ax1.barh(features, rf_importance, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Random Forest Importance')
        ax1.set_title('Feature Importance (Random Forest)')
        ax1.set_xlim(0, max(rf_importance) * 1.1)
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # SHAP Importance
        bars2 = ax2.barh(features, shap_importance, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('SHAP Impact')
        ax2.set_title('Feature Impact (SHAP)')
        ax2.set_xlim(0, max(shap_importance) * 1.1)
        
        # Add value labels on bars
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Feature importance chart saved as 'feature_importance_analysis.png'")
        
    except Exception as e:
        print(f"❌ Error creating chart: {e}")

def main():
    """Main function to run the feature importance test"""
    
    print("🚢 Titanic Feature Importance Analysis Test")
    print("=" * 60)
    
    # Check if the Flask app is running
    try:
        response = requests.get('http://localhost:5001/')
        if response.status_code == 200:
            print("✅ Flask app is running successfully!")
            
            # Test feature importance
            data = test_feature_importance()
            
            if data:
                # Create visualization
                try:
                    create_feature_importance_chart(data)
                except ImportError:
                    print("\n📊 Chart creation skipped (matplotlib not available)")
                except Exception as e:
                    print(f"\n❌ Chart creation failed: {e}")
                
                print(f"\n🎯 Feature importance analysis complete!")
                print(f"   Access the web interface at: http://localhost:5001")
                print(f"   Use the 'Refresh Analysis' button to update results")
                
        else:
            print(f"❌ Flask app returned status code: {response.status_code}")
            
    except requests.exceptions.RequestException:
        print("❌ Flask app is not running. Please start it with: python app.py")
        print("   Then run this test script again.")

if __name__ == "__main__":
    main()

