"""
Feature engineering module for mental wellness prediction.
Creates and transforms features for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging


class FeatureEngineer:
    """
    Handles feature engineering for mental wellness prediction models.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw mental wellness data.
        
        Args:
            df: Raw DataFrame with mental wellness data
            
        Returns:
            DataFrame with engineered features
        """
        featured_df = df.copy()
        
        # Sleep-related features
        if 'sleep_hours' in df.columns:
            featured_df['sleep_deficit'] = np.maximum(0, 8 - featured_df['sleep_hours'])
            featured_df['sleep_excess'] = np.maximum(0, featured_df['sleep_hours'] - 9)
            featured_df['sleep_quality_score'] = self._calculate_sleep_quality(featured_df['sleep_hours'])
        
        # Stress compound features
        if all(col in df.columns for col in ['work_stress_level', 'financial_stress']):
            featured_df['total_stress_score'] = (
                featured_df['work_stress_level'] + featured_df['financial_stress']
            ) / 2
        
        # Mood and energy interaction
        if all(col in df.columns for col in ['mood_rating', 'energy_level']):
            featured_df['mood_energy_interaction'] = (
                featured_df['mood_rating'] * featured_df['energy_level']
            )
            featured_df['mood_energy_ratio'] = (
                featured_df['mood_rating'] / (featured_df['energy_level'] + 0.1)
            )
        
        # Age-based features
        if 'age' in df.columns:
            featured_df['age_group'] = pd.cut(
                featured_df['age'], 
                bins=[0, 25, 35, 50, 65, 100],
                labels=['young_adult', 'adult', 'middle_aged', 'senior', 'elderly']
            )
            # One-hot encode age groups
            age_dummies = pd.get_dummies(featured_df['age_group'], prefix='age')
            featured_df = pd.concat([featured_df, age_dummies], axis=1)
            featured_df.drop('age_group', axis=1, inplace=True)
        
        # Social and lifestyle balance
        if all(col in df.columns for col in ['social_interaction_score', 'exercise_frequency']):
            featured_df['lifestyle_balance_score'] = (
                featured_df['social_interaction_score'] + 
                featured_df['exercise_frequency']
            ) / 2
        
        # Risk indicators
        if 'concentration_difficulty' in df.columns:
            featured_df['cognitive_impairment_flag'] = (
                featured_df['concentration_difficulty'] > 7
            ).astype(int)
        
        self.feature_names = [col for col in featured_df.columns 
                             if col not in ['depression_risk', 'anxiety_risk']]
        
        self.logger.info(f"Created {len(self.feature_names)} features")
        return featured_df
    
    def _calculate_sleep_quality(self, sleep_hours: pd.Series) -> pd.Series:
        """
        Calculate sleep quality score based on sleep hours.
        
        Args:
            sleep_hours: Series of sleep hours
            
        Returns:
            Sleep quality scores (1-10)
        """
        # Optimal sleep is around 7-8 hours
        optimal_range = (sleep_hours >= 7) & (sleep_hours <= 8)
        quality_score = np.where(optimal_range, 10, 
                                np.maximum(1, 10 - np.abs(sleep_hours - 7.5)))
        return pd.Series(quality_score, index=sleep_hours.index)
    
    def fit_scaler(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit the feature scaler on training data.
        
        Args:
            df: DataFrame with features to fit scaler on
            
        Returns:
            Self for method chaining
        """
        feature_df = df[self.feature_names] if self.feature_names else df
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            self.scaler.fit(feature_df[numeric_columns])
            self.is_fitted = True
            self.logger.info(f"Fitted scaler on {len(numeric_columns)} numeric features")
        
        return self
    
    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            df: DataFrame with features to transform
            
        Returns:
            DataFrame with scaled features
        """
        if not self.is_fitted:
            self.logger.warning("Scaler not fitted. Call fit_scaler() first.")
            return df
        
        transformed_df = df.copy()
        feature_df = transformed_df[self.feature_names] if self.feature_names else transformed_df
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            transformed_df[numeric_columns] = self.scaler.transform(feature_df[numeric_columns])
        
        return transformed_df
    
    def get_feature_importance_names(self) -> List[str]:
        """
        Get the names of all engineered features.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy() if self.feature_names else []