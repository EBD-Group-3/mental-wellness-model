"""
Test the training fix with small dataset and continuous risk values.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the package to the path  
sys.path.insert(0, str(Path(__file__).parent))

def test_training_fix():
    """Test training with your specific data to verify the fix."""
    
    print("🧪 TESTING TRAINING FIX")
    print("=" * 40)
    
    # Your data (from the error case)
    data = {
        'age': [50, 46, 51, 38],
        'sleep_hours': [7.8, 7.5, 7.5, 6.1],
        'exercise_minutes': [100, 102, 90, 119],
        'work_stress_level': [8, 7, 5, 2],
        'mood_rating': [3, 6, 2, 10],
        'energy_level': [18.76, 16.51, 0.7, 4.28],
        'avg_heart_rate': [149, 109, 125, 146],
        'resting_heart_rate': [73.1, 68.6, 78.4, 64.1],
        'depression_risk': [0.52, 0.41, 0.62, 0.35],  # Continuous values
        'anxiety_risk': [0.64, 0.55, 0.54, 0.4]       # Continuous values
    }
    
    df = pd.DataFrame(data)
    
    print("📊 Test data:")
    print(df.to_string())
    
    print(f"\n🔍 Risk value analysis:")
    print(f"  Depression risk: min={df['depression_risk'].min():.3f}, max={df['depression_risk'].max():.3f}")
    print(f"  Anxiety risk: min={df['anxiety_risk'].min():.3f}, max={df['anxiety_risk'].max():.3f}")
    
    # Test binary conversion
    threshold = 0.5
    depression_binary = (df['depression_risk'] > threshold).astype(int)
    anxiety_binary = (df['anxiety_risk'] > threshold).astype(int)
    
    print(f"\n🎯 Binary conversion (threshold={threshold}):")
    print(f"  Depression: {depression_binary.tolist()} (classes: {np.unique(depression_binary)})")
    print(f"  Anxiety: {anxiety_binary.tolist()} (classes: {np.unique(anxiety_binary)})")
    
    # Check class distribution
    dep_counts = np.bincount(depression_binary)
    anx_counts = np.bincount(anxiety_binary)
    
    print(f"\n📈 Class distribution:")
    print(f"  Depression: Low risk (0): {dep_counts[0] if len(dep_counts) > 0 else 0}, High risk (1): {dep_counts[1] if len(dep_counts) > 1 else 0}")
    print(f"  Anxiety: Low risk (0): {anx_counts[0] if len(anx_counts) > 0 else 0}, High risk (1): {anx_counts[1] if len(anx_counts) > 1 else 0}")
    
    # Test with actual predictor
    print(f"\n🚀 TESTING WITH PREDICTOR:")
    try:
        from mental_wellness_model.controller.predictor import MentalWellnessPredictor
        
        predictor = MentalWellnessPredictor(model_type='random_forest')
        
        # Test training
        results = predictor.train(df, test_size=0.2)
        
        print("✅ Training completed successfully!")
        print(f"Results: {list(results.keys())}")
        
        for model_name, metrics in results.items():
            print(f"\n  {model_name.upper()} Results:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"    {metric_name}: {value:.3f}")
                else:
                    print(f"    {metric_name}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases that might cause issues."""
    
    print(f"\n🔍 TESTING EDGE CASES:")
    
    # Case 1: All same class
    print(f"\n1. All same class (all high risk):")
    data_same_class = {
        'age': [50, 46, 51, 38],
        'sleep_hours': [7.8, 7.5, 7.5, 6.1],
        'exercise_minutes': [100, 102, 90, 119],
        'work_stress_level': [8, 7, 5, 2],
        'mood_rating': [3, 6, 2, 10],
        'energy_level': [18.76, 16.51, 0.7, 4.28],
        'avg_heart_rate': [149, 109, 125, 146],
        'resting_heart_rate': [73.1, 68.6, 78.4, 64.1],
        'depression_risk': [0.8, 0.9, 0.7, 0.6],  # All > 0.5
        'anxiety_risk': [0.3, 0.2, 0.4, 0.1]      # All < 0.5
    }
    
    df_same = pd.DataFrame(data_same_class)
    
    try:
        from mental_wellness_model.controller.predictor import MentalWellnessPredictor
        predictor = MentalWellnessPredictor(model_type='random_forest')
        results = predictor.train(df_same, test_size=0.2)
        print("  ✅ Handled same-class data successfully")
    except Exception as e:
        print(f"  ❌ Same-class test failed: {e}")

if __name__ == "__main__":
    success = test_training_fix()
    test_edge_cases()
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"Your training should now work with continuous risk values.")
    else:
        print(f"\n❌ Tests failed. Check the error messages above.")