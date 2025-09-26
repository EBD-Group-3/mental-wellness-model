"""
Mental wellness prediction model.
Implements machine learning models for early detection of depression and anxiety.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import logging
import os


class MentalWellnessPredictor:
    """
    Machine learning model for predicting mental wellness risks.
    Focuses on early detection of depression and anxiety.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the mental wellness predictor.
        
        Args:
            model_type: Type of model to use ('random_forest' or 'logistic_regression')
        """
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.models = {}
        self.is_trained = False
        self.feature_columns = []
        
        # Initialize models
        if model_type == 'random_forest':
            self.models['depression'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.models['anxiety'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'logistic_regression':
            self.models['depression'] = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
            self.models['anxiety'] = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Dict]:
        """
        Train the mental wellness prediction models.
        
        Args:
            df: DataFrame with features and target variables
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with training metrics for each model
        """
        # Prepare features and targets
        target_columns = ['depression_risk', 'anxiety_risk']
        self.feature_columns = [col for col in df.columns if col not in target_columns]
        
        X = df[self.feature_columns]
        
        results = {}
        
        # Train separate models for depression and anxiety
        for condition in ['depression', 'anxiety']:
            target_col = f'{condition}_risk'
            
            if target_col not in df.columns:
                self.logger.warning(f"Target column {target_col} not found. Skipping {condition} model.")
                continue
            
            y = df[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train model
            self.logger.info(f"Training {condition} model...")
            self.models[condition].fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.models[condition].score(X_train, y_train)
            test_score = self.models[condition].score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.models[condition], X_train, y_train, cv=5)
            
            # Predictions for detailed metrics
            y_pred = self.models[condition].predict(X_test)
            y_pred_proba = self.models[condition].predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[condition] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std_accuracy': cv_scores.std(),
                'auc_score': auc_score,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            self.logger.info(f"{condition.capitalize()} model - Test Accuracy: {test_score:.3f}, AUC: {auc_score:.3f}")
        
        self.is_trained = True
        return results
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for mental wellness risks.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions.")
        
        X = df[self.feature_columns]
        predictions = df.copy()
        
        for condition in ['depression', 'anxiety']:
            if condition in self.models:
                # Predictions
                pred_col = f'{condition}_prediction'
                prob_col = f'{condition}_probability'
                
                predictions[pred_col] = self.models[condition].predict(X)
                predictions[prob_col] = self.models[condition].predict_proba(X)[:, 1]
        
        return predictions
    
    def predict_individual(self, features: Dict[str, Union[int, float]]) -> Dict[str, Dict]:
        """
        Make predictions for a single individual.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Dictionary with predictions for each condition
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions.")
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        X = df[self.feature_columns]
        
        results = {}
        for condition in ['depression', 'anxiety']:
            if condition in self.models:
                prediction = self.models[condition].predict(X)[0]
                probability = self.models[condition].predict_proba(X)[0, 1]
                
                results[condition] = {
                    'prediction': bool(prediction),
                    'probability': float(probability),
                    'risk_level': self._get_risk_level(probability)
                }
        
        return results
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Convert probability to risk level description.
        
        Args:
            probability: Risk probability (0-1)
            
        Returns:
            Risk level description
        """
        if probability < 0.3:
            return 'Low Risk'
        elif probability < 0.6:
            return 'Moderate Risk'
        elif probability < 0.8:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    def get_feature_importance(self, condition: str = 'depression') -> pd.DataFrame:
        """
        Get feature importance for a specific condition.
        
        Args:
            condition: Condition to get importance for ('depression' or 'anxiety')
            
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_trained or condition not in self.models:
            raise ValueError(f"Model for {condition} not trained.")
        
        if hasattr(self.models[condition], 'feature_importances_'):
            importance = self.models[condition].feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            self.logger.warning(f"Feature importance not available for {self.model_type}")
            return pd.DataFrame()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained models to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("No trained models to save.")
        
        model_data = {
            'models': self.models,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Models saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained models from file.
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data.get('model_type', self.model_type)
        self.is_trained = model_data.get('is_trained', True)  # Default to True if successfully loaded
        
        # Ensure the model is marked as trained if we have valid models
        if self.models and len(self.models) > 0:
            self.is_trained = True
        
        self.logger.info(f"Models loaded from {filepath}, is_trained: {self.is_trained}")