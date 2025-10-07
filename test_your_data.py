"""
Test your specific data against the validation rules.
"""

import pandas as pd
import sys
from pathlib import Path

# Add the package to the path  
sys.path.insert(0, str(Path(__file__).parent))

def test_your_data():
    """Test the data you provided."""
    
    print("🧪 TESTING YOUR SPECIFIC DATA")
    print("=" * 40)
    
    # Your data
    data = {
        'age': [50, 46, 51, 38],
        'sleep_hours': [7.8, 7.5, 7.5, 6.1],
        'exercise_minutes': [100, 102, 90, 119],
        'work_stress_level': [8, 7, 5, 2],
        'mood_rating': [3, 6, 2, 10],
        'energy_level': [18.76, 16.51, 0.7, 4.28],
        'avg_heart_rate': [149, 109, 125, 146],
        'resting_heart_rate': [73.1, 68.6, 78.4, 64.1],
        'depression_risk': [0.52, 0.41, 0.62, 0.35],
        'anxiety_risk': [0.64, 0.55, 0.54, 0.4]
    }
    
    df = pd.DataFrame(data)
    
    print("📊 Your data:")
    print(df.to_string())
    
    print("\n🔍 VALIDATION CHECKS:")
    
    # Test each validation rule
    validations = [
        ("age >= 0", (df['age'] >= 0).all()),
        ("sleep_hours >= 0", (df['sleep_hours'] >= 0).all()),
        ("exercise_minutes >= 0", (df['exercise_minutes'] >= 0).all()),
        ("work_stress_level >= 0", (df['work_stress_level'] >= 0).all()),
        ("mood_rating >= 0", (df['mood_rating'] >= 0).all()),
        ("energy_level >= 0", (df['energy_level'] >= 0).all()),
        ("avg_heart_rate 30-250", (df['avg_heart_rate'] >= 30).all() and (df['avg_heart_rate'] <= 250).all()),
        ("resting_heart_rate 30-150", (df['resting_heart_rate'] >= 30).all() and (df['resting_heart_rate'] <= 150).all()),
    ]
    
    all_passed = True
    for validation_name, result in validations:
        status = "✅" if result else "❌"
        print(f"  {status} {validation_name}")
        if not result:
            all_passed = False
            # Show problematic values
            column_name = validation_name.split()[0]
            if column_name in df.columns:
                values = df[column_name].tolist()
                print(f"      Values: {values}")
    
    print(f"\n📋 REQUIRED COLUMNS CHECK:")
    required_columns = ['age', 'sleep_hours', 'exercise_minutes', 'work_stress_level', 'mood_rating', 'energy_level', 'avg_heart_rate', 'resting_heart_rate']
    target_columns = ['depression_risk', 'anxiety_risk']
    
    missing_features = set(required_columns) - set(df.columns)
    missing_targets = set(target_columns) - set(df.columns)
    
    if missing_features:
        print(f"  ❌ Missing features: {missing_features}")
        all_passed = False
    else:
        print(f"  ✅ All required feature columns present")
        
    if missing_targets:
        print(f"  ❌ Missing targets: {missing_targets}")
        all_passed = False
    else:
        print(f"  ✅ All required target columns present")
    
    print(f"\n🎯 OVERALL RESULT:")
    if all_passed:
        print("  ✅ Your data should pass validation with the relaxed rules!")
    else:
        print("  ❌ Your data still has some validation issues")
    
    # Test with the actual data processor
    print(f"\n🧪 TESTING WITH ACTUAL PROCESSOR:")
    try:
        from mental_wellness_model.controller.processor import DataProcessor
        processor = DataProcessor()
        
        result = processor.validate_data(df)
        if result:
            print("  ✅ Data processor validation PASSED!")
        else:
            print("  ❌ Data processor validation FAILED!")
            
    except Exception as e:
        print(f"  ❌ Error testing with processor: {e}")
    
    # Save a test CSV
    test_file = "test_data_sample.csv"
    df.to_csv(test_file, index=False)
    print(f"\n💾 Saved test data to: {test_file}")
    print("  You can upload this file to test training")

if __name__ == "__main__":
    test_your_data()