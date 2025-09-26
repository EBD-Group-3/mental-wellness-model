"""
Basic tests for mental wellness model components.
"""

import unittest
import pandas as pd
import numpy as np
from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer


class TestDataProcessor(unittest.TestCase):
    """Test the DataProcessor class."""
    
    def setUp(self):
        self.processor = DataProcessor()
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        df = self.processor.load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('depression_risk', df.columns)
        self.assertIn('anxiety_risk', df.columns)
    
    def test_data_validation(self):
        """Test data validation."""
        df = self.processor.load_data()
        self.assertTrue(self.processor.validate_data(df))
    
    def test_data_cleaning(self):
        """Test data cleaning."""
        df = self.processor.load_data()
        cleaned_df = self.processor.clean_data(df)
        self.assertEqual(len(df), len(cleaned_df))
        self.assertIsInstance(cleaned_df, pd.DataFrame)


class TestFeatureEngineer(unittest.TestCase):
    """Test the FeatureEngineer class."""
    
    def setUp(self):
        self.engineer = FeatureEngineer()
        self.processor = DataProcessor()
        self.sample_df = self.processor.load_data()
    
    def test_create_features(self):
        """Test feature creation."""
        original_cols = len(self.sample_df.columns)
        featured_df = self.engineer.create_features(self.sample_df)
        self.assertGreaterEqual(len(featured_df.columns), original_cols)
        self.assertIn('sleep_deficit', featured_df.columns)
        self.assertIn('total_stress_score', featured_df.columns)
    
    def test_feature_scaling(self):
        """Test feature scaling."""
        featured_df = self.engineer.create_features(self.sample_df)
        self.engineer.fit_scaler(featured_df)
        scaled_df = self.engineer.transform_features(featured_df)
        self.assertEqual(len(featured_df), len(scaled_df))
        self.assertTrue(self.engineer.is_fitted)


class TestMentalWellnessPredictor(unittest.TestCase):
    """Test the MentalWellnessPredictor class."""
    
    def setUp(self):
        self.predictor = MentalWellnessPredictor()
        self.processor = DataProcessor()
        self.engineer = FeatureEngineer()
        
        # Prepare test data
        df = self.processor.load_data()
        df = self.processor.clean_data(df)
        df = self.engineer.create_features(df)
        self.engineer.fit_scaler(df)
        self.test_df = self.engineer.transform_features(df)
    
    def test_model_training(self):
        """Test model training."""
        results = self.predictor.train(self.test_df)
        self.assertTrue(self.predictor.is_trained)
        self.assertIn('depression', results)
        self.assertIn('anxiety', results)
    
    def test_individual_prediction(self):
        """Test individual prediction."""
        # Train the model first
        self.predictor.train(self.test_df)
        
        features = {
            'age': 30,
            'sleep_hours': 7,
            'exercise_minutes': 90,
            'work_stress_level': 5,
            'mood_rating': 6,
            'energy_level': 6,
            'avg_heart_rate': 70,
            'resting_heart_rate': 65
        }
        
        results = self.predictor.predict_individual(features)
        self.assertIn('depression', results)
        self.assertIn('anxiety', results)
        self.assertIn('probability', results['depression'])
        self.assertIn('risk_level', results['depression'])
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        # Train the model first
        self.predictor.train(self.test_df)
        
        importance_df = self.predictor.get_feature_importance('depression')
        self.assertGreater(len(importance_df), 0)
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)


if __name__ == '__main__':
    unittest.main()