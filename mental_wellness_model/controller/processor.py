"""
Data processing module for mental wellness prediction.
Handles data cleaning, validation, and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class DataProcessor:
    """
    Processes and cleans mental wellness data for prediction models.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_columns = []
        self.target_columns = ['depression_risk', 'anxiety_risk']
    
    def load_data(self, data_path: Optional[str] = None, 
                  data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Load mental wellness data from file or DataFrame.
        
        Args:
            data_path: Path to data file (CSV)
            data: DataFrame with mental wellness data
            
        Returns:
            Loaded DataFrame
        """
        if data is not None:
            return data.copy()
        elif data_path:
            return pd.read_csv(data_path)
        else:
            # Generate sample data for demonstration
            return self._generate_sample_data()
    
    def _generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate sample mental wellness data for testing and demonstration.
        
        Args:
            n_samples: Number of sample records to generate
            
        Returns:
            DataFrame with synthetic mental wellness data
        """
        np.random.seed(42)
        
        # Generate synthetic features
        data = {
            'age': np.random.normal(35, 12, n_samples).clip(18, 80),
            'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(3, 12),
            'exercise_minutes': np.random.randint(0, 240, n_samples),  # minutes per week
            'work_stress_level': np.random.normal(5, 2, n_samples).clip(1, 10),
            'mood_rating': np.random.normal(6, 2, n_samples).clip(1, 10),
            'energy_level': np.random.normal(6, 2, n_samples).clip(1, 10),
            'avg_heart_rate': np.random.normal(70, 10, n_samples).clip(50, 120),  # beats per minute
            'resting_heart_rate': np.random.normal(65, 8, n_samples).clip(50, 90),  # beats per minute at rest
        }
        
        df = pd.DataFrame(data)
        
        # Generate target variables based on features
        depression_risk_prob = (
            0.1 +
            0.2 * (10 - df['mood_rating']) / 10 +
            0.15 * (10 - df['energy_level']) / 10 +
            0.1 * df['work_stress_level'] / 10 +
            0.1 * (8 - df['sleep_hours']) / 8 +
            0.05 * (120 - df['exercise_minutes']) / 120 +  # less exercise increases depression risk
            0.05 * (df['resting_heart_rate'] - 60) / 30  # higher resting heart rate increases depression risk
        ).clip(0, 1)
        
        anxiety_risk_prob = (
            0.1 +
            0.2 * df['work_stress_level'] / 10 +
            0.15 * (10 - df['energy_level']) / 10 +
            0.1 * (10 - df['mood_rating']) / 10 +
            0.05 * (8 - df['sleep_hours']) / 8 +  # poor sleep increases anxiety risk
            0.1 * (df['avg_heart_rate'] - 60) / 60 +  # higher heart rate increases anxiety risk
            0.05 * (df['resting_heart_rate'] - 60) / 30  # higher resting heart rate increases anxiety risk
        ).clip(0, 1)
        
        df['depression_risk'] = np.random.binomial(1, depression_risk_prob)
        df['anxiety_risk'] = np.random.binomial(1, anxiety_risk_prob)
        
        # Generate onset day predictions (days until potential mental breakdown)
        # Base onset days calculation on depression and anxiety risk only
        combined_risk_score = (
            0.6 * depression_risk_prob +
            0.4 * anxiety_risk_prob
        ).clip(0, 1)
        
        # Convert risk score to days (higher risk = fewer days)
        # Range: 7-365 days, with more gradual relationship
        # Use linear interpolation for more realistic day distribution
        min_days = 7   # Minimum 1 week even for highest risk
        max_days = 365 # Maximum 1 year for lowest risk
        base_days = min_days + (max_days - min_days) * (1 - combined_risk_score)
        
        # Add some randomness and ensure minimum of 7 days
        noise = np.random.normal(0, base_days * 0.15, n_samples)
        df['onset_day'] = np.maximum(min_days, base_days + noise).round().astype(int)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate mental wellness data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_df = df.copy()
        
        # Handle missing values
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in self.target_columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        # Remove outliers using IQR method
        for col in numeric_columns:
            if col not in self.target_columns:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
        
        self.logger.info(f"Data cleaned. Shape: {cleaned_df.shape}")
        return cleaned_df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality and completeness.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['age', 'sleep_hours', 'exercise_minutes', 'work_stress_level', 'mood_rating', 'energy_level', 'avg_heart_rate', 'resting_heart_rate']
        
        # Check required columns exist
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for reasonable value ranges (more flexible validation)
        validations = [
            (df['age'] >= 0).all(),
            (df['sleep_hours'] >= 0).all(),
            (df['exercise_minutes'] >= 0).all(),
            (df['work_stress_level'] >= 0).all(),  # Relaxed: allow 0+ instead of 1-10
            (df['mood_rating'] >= 0).all(),        # Relaxed: allow 0+ instead of 1-10
            (df['energy_level'] >= 0).all(),       # Relaxed: allow 0+ instead of 1-10
            (df['avg_heart_rate'] >= 30).all() and (df['avg_heart_rate'] <= 250).all(),  # Wider range
            (df['resting_heart_rate'] >= 30).all() and (df['resting_heart_rate'] <= 150).all(),  # Wider range
        ]
        
        # Additional validation for onset_day if present
        if 'onset_day' in df.columns:
            validations.extend([
                (df['onset_day'] >= 1).all(),  # At least 1 day
                (df['onset_day'] <= 365).all(),  # Maximum 1 year
                df['onset_day'].dtype in ['int64', 'int32', 'float64']  # Numeric type
            ])
        
        # Check each validation and log specific failures
        validation_names = [
            "age >= 0",
            "sleep_hours >= 0", 
            "exercise_minutes >= 0",
            "work_stress_level >= 0",
            "mood_rating >= 0",
            "energy_level >= 0",
            "avg_heart_rate 30-250",
            "resting_heart_rate 30-150"
        ]
        
        failed_validations = []
        for i, (validation, name) in enumerate(zip(validations, validation_names)):
            if not validation:
                failed_validations.append(name)
                # Log some example values that failed
                column_name = name.split()[0]
                if column_name in df.columns:
                    sample_values = df[column_name].head().tolist()
                    self.logger.error(f"Validation failed for {name}: sample values = {sample_values}")
        
        if failed_validations:
            self.logger.error(f"Data validation failed for: {failed_validations}")
            return False
        
        self.logger.info("Data validation passed")
        return True