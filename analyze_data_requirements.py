"""
Data Validation Analysis for Mental Wellness Model Training
===========================================================

Based on your data sample and the validation error, here's what your data needs:

YOUR CURRENT DATA:
Your data appears to have 8 columns but no column headers visible.
Sample row: 36, 8.5, 37, 6, 2, 5.86, 156, 66.6

REQUIRED DATA STRUCTURE:
"""

import pandas as pd

def analyze_data_requirements():
    """Analyze what your data needs for successful training."""
    
    print("🔍 MENTAL WELLNESS MODEL - DATA REQUIREMENTS ANALYSIS")
    print("=" * 60)
    
    print("\n📋 REQUIRED FEATURE COLUMNS:")
    required_features = [
        'age',
        'sleep_hours', 
        'exercise_minutes',
        'work_stress_level',
        'mood_rating',
        'energy_level',
        'avg_heart_rate',
        'resting_heart_rate'
    ]
    
    for i, col in enumerate(required_features, 1):
        print(f"  {i}. {col}")
    
    print("\n🎯 REQUIRED TARGET COLUMNS:")
    required_targets = [
        'depression_risk',
        'anxiety_risk'
    ]
    
    for i, col in enumerate(required_targets, 1):
        print(f"  {i}. {col}")
    
    print("\n📊 YOUR CURRENT DATA ANALYSIS:")
    print("Based on your sample data:")
    print("  Row example: 36, 8.5, 37, 6, 2, 5.86, 156, 66.6")
    print("  Number of columns: 8")
    print("  Missing: Column headers and target columns")
    
    print("\n🔧 POSSIBLE COLUMN MAPPING:")
    your_data = [36, 8.5, 37, 6, 2, 5.86, 156, 66.6]
    possible_mapping = [
        "age (36)",
        "sleep_hours (8.5)", 
        "exercise_minutes (37)",
        "work_stress_level (6)",
        "mood_rating (2)",
        "energy_level (5.86)",
        "avg_heart_rate (156)",
        "resting_heart_rate (66.6)"
    ]
    
    print("  Your data might be:")
    for i, (value, mapping) in enumerate(zip(your_data, possible_mapping), 1):
        print(f"    Column {i}: {value} → {mapping}")
    
    print("\n❌ WHAT'S MISSING:")
    print("  1. Column headers (CSV needs header row)")
    print("  2. Target columns: depression_risk, anxiety_risk")
    print("  3. Proper value ranges validation")
    
    print("\n✅ SOLUTIONS:")
    print("  1. Add CSV header row with column names")
    print("  2. Add depression_risk and anxiety_risk columns")
    print("  3. Ensure value ranges are correct")
    
    print("\n📝 EXAMPLE CORRECT CSV FORMAT:")
    example_data = {
        'age': [36, 47, 32, 28],
        'sleep_hours': [8.5, 8.2, 7.0, 6.5],
        'exercise_minutes': [37, 66, 45, 30], 
        'work_stress_level': [6, 2, 5, 7],
        'mood_rating': [2, 5, 7, 4],
        'energy_level': [5, 4, 8, 6],
        'avg_heart_rate': [156, 142, 145, 160],
        'resting_heart_rate': [66, 74, 65, 70],
        'depression_risk': [0.3, 0.1, 0.2, 0.4],  # 0.0 to 1.0
        'anxiety_risk': [0.2, 0.1, 0.3, 0.5]      # 0.0 to 1.0
    }
    
    df = pd.DataFrame(example_data)
    print("\n" + df.to_csv(index=False))
    
    print("\n📋 VALUE RANGES VALIDATION:")
    print("  - age: >= 0")
    print("  - sleep_hours: >= 0") 
    print("  - exercise_minutes: >= 0")
    print("  - work_stress_level: 1-10")
    print("  - mood_rating: 1-10")
    print("  - energy_level: 1-10")
    print("  - avg_heart_rate: 40-200")
    print("  - resting_heart_rate: 40-120")
    print("  - depression_risk: 0.0-1.0")
    print("  - anxiety_risk: 0.0-1.0")

def create_template_csv():
    """Create a template CSV file with proper format."""
    
    # Create template data based on user's sample
    template_data = {
        'age': [36, 36, 36, 36, 36, 47, 47, 47, 47, 47],
        'sleep_hours': [8.5, 8.5, 8.5, 8.5, 8.5, 8.2, 8.2, 8.2, 8.2, 8.2],
        'exercise_minutes': [37, 37, 37, 37, 37, 66, 66, 66, 66, 66],
        'work_stress_level': [6, 6, 6, 6, 6, 2, 2, 2, 2, 2],
        'mood_rating': [2, 5, 5, 7, 6, 5, 5, 5, 5, 5],
        'energy_level': [6, 6, 6, 6, 6, 4, 4, 10, 4, 4], # Fixed: was 5.86, now integer 1-10
        'avg_heart_rate': [156, 156, 156, 156, 156, 142, 142, 142, 142, 142],
        'resting_heart_rate': [67, 67, 67, 67, 67, 75, 75, 75, 75, 75], # Fixed: was 66.6/74.7, now integers
        'depression_risk': [0.3, 0.2, 0.2, 0.4, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1],
        'anxiety_risk': [0.2, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2]
    }
    
    df = pd.DataFrame(template_data)
    
    # Save template
    template_file = "wellness_data_template.csv"
    df.to_csv(template_file, index=False)
    
    print(f"\n💾 CREATED TEMPLATE FILE: {template_file}")
    print(f"   This file contains {len(df)} sample rows with proper format")
    print(f"   Upload this to your GCS bucket to test training")

if __name__ == "__main__":
    analyze_data_requirements()
    create_template_csv()
    
    print("\n" + "="*60)
    print("🎯 NEXT STEPS:")
    print("1. Create a CSV file with the correct column headers")
    print("2. Add depression_risk and anxiety_risk columns")
    print("3. Ensure all values are within the specified ranges")
    print("4. Upload the corrected CSV to your GCS bucket")
    print("5. Use the CSV for training instead of Parquet")
    print("\nExample training request with CSV:")
    print('{')
    print('  "model_type": "random_forest",')
    print('  "use_sample_data": false,')
    print('  "use_gcs_data": true,')
    print('  "gcs_data_folder": "RawData",')
    print('  "gcs_data_filename": "wellness_data_corrected.csv"')
    print('}')