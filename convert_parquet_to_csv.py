"""
Parquet to CSV Conversion Utility
Helps users convert their Parquet files to CSV format if needed.
"""

import pandas as pd
import sys
from pathlib import Path

def convert_parquet_to_csv(parquet_file: str, csv_file: str = None):
    """Convert a Parquet file to CSV format."""
    
    try:
        # Check if input file exists
        if not Path(parquet_file).exists():
            print(f"❌ Error: File not found: {parquet_file}")
            return False
        
        # Generate output filename if not provided
        if csv_file is None:
            csv_file = str(Path(parquet_file).with_suffix('.csv'))
        
        print(f"🔄 Converting {parquet_file} to {csv_file}...")
        
        # Try to load the Parquet file
        try:
            df = pd.read_parquet(parquet_file)
            print(f"✅ Loaded Parquet file: {len(df)} rows, {len(df.columns)} columns")
        except ImportError as e:
            print(f"❌ Error: Missing Parquet support.")
            print(f"   pyarrow is required for Parquet file reading.")
            print(f"   Error details: {e}")
            return False
        except Exception as e:
            print(f"❌ Error reading Parquet file: {e}")
            return False
        
        # Save as CSV
        try:
            df.to_csv(csv_file, index=False)
            print(f"✅ Successfully converted to CSV: {csv_file}")
            
            # Show preview
            print(f"\n📊 Preview of converted data:")
            print(f"Columns: {', '.join(df.columns.tolist())}")
            print(f"Data types: {df.dtypes.to_dict()}")
            print(f"\nFirst 3 rows:")
            print(df.head(3).to_string())
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving CSV file: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function to handle command line arguments."""
    
    print("🔄 Parquet to CSV Conversion Utility")
    print("="*50)
    
    if len(sys.argv) < 2:
        print("Usage: python convert_parquet_to_csv.py <input.parquet> [output.csv]")
        print("\nExamples:")
        print("  python convert_parquet_to_csv.py wellness_sample.parquet")
        print("  python convert_parquet_to_csv.py data/wellness_sample.parquet data/wellness_sample.csv")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_parquet_to_csv(parquet_file, csv_file)
    
    if success:
        print(f"\n🎉 Conversion completed successfully!")
        print(f"You can now upload the CSV file to your GCS bucket and use:")
        print(f'  "gcs_data_filename": "{Path(csv_file or parquet_file).with_suffix(\".csv\").name}"')
    else:
        print(f"\n❌ Conversion failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()