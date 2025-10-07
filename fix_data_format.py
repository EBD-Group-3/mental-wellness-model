"""
Data Correction Script for Mental Wellness Model
Helps fix your existing data to meet the model requirements.
"""

import pandas as pd
import numpy as np

def fix_data_format(input_file_path: str, output_file_path: str = None):
    """
    Fix your data to meet the Mental Wellness Model requirements.
    
    Args:
        input_file_path: Path to your current data file
        output_file_path: Path for the corrected output file
    """
    
    if output_file_path is None:
        output_file_path = input_file_path.replace('.', '_corrected.')
    
    print("🔧 FIXING DATA FORMAT FOR MENTAL WELLNESS MODEL")
    print("=" * 55)
    
    try:
        # Try to load the data
        print(f"\n📁 Loading data from: {input_file_path}")
        
        # Try different loading methods
        try:
            df = pd.read_csv(input_file_path)
            print(f"✅ Loaded CSV with shape: {df.shape}")
        except:
            try:
                # Try without headers (your case)
                df = pd.read_csv(input_file_path, header=None)
                print(f"✅ Loaded CSV without headers with shape: {df.shape}")
            except:
                print("❌ Could not load file. Please ensure it's a valid CSV.")
                return False
        
        print(f"   Original columns: {list(df.columns)}")
        print(f"   First few rows:")
        print(df.head())
        
        # Define the expected column mapping
        expected_columns = [
            'age',
            'sleep_hours', 
            'exercise_minutes',
            'work_stress_level',
            'mood_rating',
            'energy_level',
            'avg_heart_rate',
            'resting_heart_rate'
        ]
        
        # Check if we have the right number of columns for features
        if len(df.columns) == 8:
            print(f"\n✅ Found 8 columns - mapping to feature columns")
            df.columns = expected_columns
        elif len(df.columns) == 10:
            print(f"\n✅ Found 10 columns - assuming already has target columns")
            df.columns = expected_columns + ['depression_risk', 'anxiety_risk']
        else:
            print(f"\n❌ Unexpected number of columns: {len(df.columns)}")
            print("Expected 8 (features only) or 10 (features + targets)")
            return False
        
        print(f"   New column names: {list(df.columns)}")
        
        # Add target columns if missing
        if 'depression_risk' not in df.columns:
            print(f"\n🎯 Adding missing target columns...")
            
            # Generate realistic target values based on the features
            # Higher stress, lower mood/energy = higher depression/anxiety risk
            depression_risk = []
            anxiety_risk = []
            
            for _, row in df.iterrows():
                # Base risk calculation
                stress_factor = (row['work_stress_level'] - 1) / 9  # Normalize 1-10 to 0-1
                mood_factor = 1 - ((row['mood_rating'] - 1) / 9)    # Invert: low mood = high risk
                energy_factor = 1 - ((row['energy_level'] - 1) / 9) # Invert: low energy = high risk
                sleep_factor = max(0, (8 - row['sleep_hours']) / 8)  # Less than 8 hours increases risk
                
                # Calculate depression risk (0-1)
                depression = np.clip(
                    (stress_factor * 0.3 + mood_factor * 0.4 + energy_factor * 0.2 + sleep_factor * 0.1),
                    0, 1
                )
                
                # Calculate anxiety risk (0-1) - slightly different weights
                anxiety = np.clip(
                    (stress_factor * 0.4 + mood_factor * 0.3 + energy_factor * 0.2 + sleep_factor * 0.1),
                    0, 1
                )
                
                # Add some randomness to make it more realistic
                depression += np.random.normal(0, 0.05)  # Small random variation
                anxiety += np.random.normal(0, 0.05)
                
                depression_risk.append(np.clip(depression, 0, 1))
                anxiety_risk.append(np.clip(anxiety, 0, 1))
            
            df['depression_risk'] = [round(x, 3) for x in depression_risk]
            df['anxiety_risk'] = [round(x, 3) for x in anxiety_risk]
            
            print(f"   Added depression_risk: min={min(depression_risk):.3f}, max={max(depression_risk):.3f}")
            print(f"   Added anxiety_risk: min={min(anxiety_risk):.3f}, max={max(anxiety_risk):.3f}")
        
        # Fix value ranges
        print(f"\n🔧 Fixing value ranges...")
        
        # Fix energy_level and mood_rating (should be integers 1-10)
        df['energy_level'] = df['energy_level'].round().astype(int)
        df['mood_rating'] = df['mood_rating'].round().astype(int)
        df['work_stress_level'] = df['work_stress_level'].round().astype(int)
        
        # Ensure all values are within valid ranges
        df['energy_level'] = df['energy_level'].clip(1, 10)
        df['mood_rating'] = df['mood_rating'].clip(1, 10)
        df['work_stress_level'] = df['work_stress_level'].clip(1, 10)
        
        # Round heart rates to integers
        df['avg_heart_rate'] = df['avg_heart_rate'].round().astype(int)
        df['resting_heart_rate'] = df['resting_heart_rate'].round().astype(int)
        
        print(f"   Fixed value ranges for rating columns")
        
        # Validate the corrected data
        print(f"\n✅ VALIDATION CHECK:")
        required_columns = ['age', 'sleep_hours', 'exercise_minutes', 'work_stress_level', 'mood_rating', 'energy_level', 'avg_heart_rate', 'resting_heart_rate']
        target_columns = ['depression_risk', 'anxiety_risk']
        
        missing_features = set(required_columns) - set(df.columns)
        missing_targets = set(target_columns) - set(df.columns)
        
        if missing_features:
            print(f"   ❌ Still missing features: {missing_features}")
        else:
            print(f"   ✅ All required feature columns present")
            
        if missing_targets:
            print(f"   ❌ Still missing targets: {missing_targets}")
        else:
            print(f"   ✅ All required target columns present")
        
        # Check value ranges
        validations = [
            ("age >= 0", (df['age'] >= 0).all()),
            ("sleep_hours >= 0", (df['sleep_hours'] >= 0).all()),
            ("exercise_minutes >= 0", (df['exercise_minutes'] >= 0).all()),
            ("work_stress_level 1-10", (df['work_stress_level'] >= 1).all() and (df['work_stress_level'] <= 10).all()),
            ("mood_rating 1-10", (df['mood_rating'] >= 1).all() and (df['mood_rating'] <= 10).all()),
            ("energy_level 1-10", (df['energy_level'] >= 1).all() and (df['energy_level'] <= 10).all()),
            ("avg_heart_rate 40-200", (df['avg_heart_rate'] >= 40).all() and (df['avg_heart_rate'] <= 200).all()),
            ("resting_heart_rate 40-120", (df['resting_heart_rate'] >= 40).all() and (df['resting_heart_rate'] <= 120).all()),
        ]
        
        for validation_name, is_valid in validations:
            status = "✅" if is_valid else "❌"
            print(f"   {status} {validation_name}")
        
        # Save the corrected data
        df.to_csv(output_file_path, index=False)
        print(f"\n💾 SAVED CORRECTED DATA:")
        print(f"   File: {output_file_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        print(f"\n📊 SAMPLE OF CORRECTED DATA:")
        print(df.head())
        
        print(f"\n🎉 DATA CORRECTION COMPLETED!")
        print(f"   Upload {output_file_path} to your GCS bucket")
        print(f"   Use filename: {output_file_path.split('/')[-1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing data: {e}")
        return False

def main():
    """Main function with example usage."""
    print("🧠 Mental Wellness Model - Data Correction Tool")
    print("\nThis tool will:")
    print("1. Load your data file")
    print("2. Add proper column headers")
    print("3. Generate missing target columns (depression_risk, anxiety_risk)")
    print("4. Fix value ranges")
    print("5. Save corrected data")
    
    # Example usage - you can modify this
    input_file = "your_data.csv"  # Replace with your file name
    
    print(f"\n📝 TO USE THIS TOOL:")
    print(f"1. Save your data as 'your_data.csv' in this directory")
    print(f"2. Run: python fix_data_format.py")
    print(f"3. Upload the corrected file to GCS")
    
    # Uncomment the line below and replace with your actual file path
    # fix_data_format(input_file)

if __name__ == "__main__":
    main()