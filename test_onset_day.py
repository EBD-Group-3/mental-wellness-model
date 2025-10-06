#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mental_wellness_model.controller.processor import DataProcessor

def test_onset_day_calculation():
    # Create test data
    processor = DataProcessor()
    test_df = processor.generate_synthetic_data(100)
    
    # Check the onset_day distribution
    print('Onset Day Statistics:')
    print(f'Min: {test_df["onset_day"].min()}')
    print(f'Max: {test_df["onset_day"].max()}')
    print(f'Mean: {test_df["onset_day"].mean():.1f}')
    print(f'Median: {test_df["onset_day"].median():.1f}')
    print(f'Std: {test_df["onset_day"].std():.1f}')
    print()
    print('Sample values (showing first 10 rows):')
    print(test_df[['depression_risk', 'anxiety_risk', 'onset_day']].head(10))
    print()
    
    # Show distribution by risk levels
    high_risk = test_df[(test_df['depression_risk'] == 1) | (test_df['anxiety_risk'] == 1)]
    low_risk = test_df[(test_df['depression_risk'] == 0) & (test_df['anxiety_risk'] == 0)]
    
    if len(high_risk) > 0:
        print(f'High risk cases (n={len(high_risk)}): Mean onset_day = {high_risk["onset_day"].mean():.1f}')
    if len(low_risk) > 0:
        print(f'Low risk cases (n={len(low_risk)}): Mean onset_day = {low_risk["onset_day"].mean():.1f}')

if __name__ == "__main__":
    test_onset_day_calculation()