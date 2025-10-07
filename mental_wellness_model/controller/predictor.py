"""
Mental wellness prediction model.
Implements machine learning models for early detection of depression and anxiety.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_error
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
            # Regression model for onset day prediction
            self.models['onset_day'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
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
            # Linear regression model for onset day prediction
            self.models['onset_day'] = LinearRegression()
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
        target_columns = ['depression_risk', 'anxiety_risk', 'onset_day']
        self.feature_columns = [col for col in df.columns if col not in target_columns]
        
        X = df[self.feature_columns]
        
        results = {}
        
        # Train classification models for depression and anxiety
        for condition in ['depression', 'anxiety']:
            target_col = f'{condition}_risk'
            
            if target_col not in df.columns:
                self.logger.warning(f"Target column {target_col} not found. Skipping {condition} model.")
                continue
            
            y = df[target_col]
            
            # Convert continuous risk values to binary classes using threshold
            risk_threshold = 0.5  # Above 0.5 = high risk (1), below = low risk (0)
            if y.dtype in ['float64', 'float32'] and y.min() >= 0 and y.max() <= 1:
                y_binary = (y > risk_threshold).astype(int)
                self.logger.info(f"Converted continuous {condition} risk to binary: {sum(y_binary)}/{len(y_binary)} high risk samples")
            else:
                y_binary = y.astype(int)  # Assume already binary
            
            # Check if we have both classes and enough samples for test split
            unique_classes = np.unique(y_binary)
            n_samples = len(y_binary)
            min_test_samples = max(1, int(n_samples * test_size))
            
            if len(unique_classes) < 2:
                self.logger.warning(f"Only one class found for {condition}: {unique_classes}. Skipping stratified split.")
                stratify_param = None
            elif min_test_samples < len(unique_classes):
                self.logger.warning(f"Not enough samples for stratified split (need {len(unique_classes)}, got {min_test_samples}). Using random split.")
                stratify_param = None
            else:
                stratify_param = y_binary
            
            # Adjust test_size for very small datasets
            if n_samples <= 10:
                adjusted_test_size = max(1, n_samples // 3)  # At least 1, but not more than 1/3
                test_size_ratio = adjusted_test_size / n_samples
                self.logger.info(f"Small dataset detected ({n_samples} samples). Adjusting test_size to {adjusted_test_size} samples ({test_size_ratio:.2f})")
            else:
                test_size_ratio = test_size
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binary, test_size=test_size_ratio, random_state=42, stratify=stratify_param
            )
            
            # Train model
            self.logger.info(f"Training {condition} model...")
            self.models[condition].fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.models[condition].score(X_train, y_train)
            test_score = self.models[condition].score(X_test, y_test)
            
            # Adaptive cross-validation - handle small datasets
            try:
                # Determine optimal CV folds based on data size and class distribution
                n_samples = len(X_train)
                n_classes = len(np.unique(y_train))
                min_class_count = min([sum(y_train == cls) for cls in np.unique(y_train)])
                
                # Choose CV folds: min of 5, data_size//10, or min_class_count
                cv_folds = min(5, max(2, n_samples // 10), min_class_count)
                
                if cv_folds < 2:
                    # Too few samples for CV, skip it
                    cv_scores = np.array([test_score])  # Use test score as proxy
                    self.logger.warning(f"Skipping cross-validation for {condition}: insufficient data (min_class_count={min_class_count})")
                else:
                    cv_scores = cross_val_score(self.models[condition], X_train, y_train, cv=cv_folds)
            except Exception as e:
                # Fallback: use test score if CV fails
                cv_scores = np.array([test_score])
                self.logger.warning(f"Cross-validation failed for {condition}: {e}. Using test score as fallback.")
            
            # Predictions for detailed metrics
            y_pred = self.models[condition].predict(X_test)
            
            # Handle probability prediction for single class case
            y_pred_proba_full = self.models[condition].predict_proba(X_test)
            if y_pred_proba_full.shape[1] == 1:
                # Only one class - use that probability
                y_pred_proba = y_pred_proba_full[:, 0]
                auc_score = None  # Can't calculate AUC with only one class
                self.logger.warning(f"Only one class in test set for {condition}. AUC calculation skipped.")
            else:
                # Normal case - use positive class probability
                y_pred_proba = y_pred_proba_full[:, 1]
                try:
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                except ValueError as e:
                    auc_score = None
                    self.logger.warning(f"AUC calculation failed for {condition}: {e}")
            
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
        
        # Train regression model for onset day prediction
        if 'onset_day' in df.columns:
            self.logger.info("Training onset day prediction model...")
            y_onset = df['onset_day']
            
            # Split data for regression (no stratification needed)
            X_train, X_test, y_train_onset, y_test_onset = train_test_split(
                X, y_onset, test_size=test_size, random_state=42
            )
            
            # Train onset day model
            self.models['onset_day'].fit(X_train, y_train_onset)
            
            # Evaluate regression model
            train_score_onset = self.models['onset_day'].score(X_train, y_train_onset)
            test_score_onset = self.models['onset_day'].score(X_test, y_test_onset)
            
            # Adaptive cross-validation for regression
            try:
                # Determine optimal CV folds for regression
                n_samples = len(X_train)
                cv_folds = min(5, max(2, n_samples // 10))
                
                if cv_folds < 2:
                    # Too few samples for CV, skip it
                    cv_scores_onset = np.array([test_score_onset])
                    self.logger.warning(f"Skipping cross-validation for onset_day: insufficient data (n_samples={n_samples})")
                else:
                    cv_scores_onset = cross_val_score(self.models['onset_day'], X_train, y_train_onset, cv=cv_folds)
            except Exception as e:
                # Fallback: use test score if CV fails
                cv_scores_onset = np.array([test_score_onset])
                self.logger.warning(f"Cross-validation failed for onset_day: {e}. Using test score as fallback.")
            
            # Predictions for detailed metrics
            y_pred_onset = self.models['onset_day'].predict(X_test)
            
            # Calculate regression metrics
            mse = mean_squared_error(y_test_onset, y_pred_onset)
            mae = mean_absolute_error(y_test_onset, y_pred_onset)
            rmse = np.sqrt(mse)
            
            results['onset_day'] = {
                'train_r2_score': train_score_onset,
                'test_r2_score': test_score_onset,
                'cv_mean_r2_score': cv_scores_onset.mean(),
                'cv_std_r2_score': cv_scores_onset.std(),
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'prediction_range': f"{y_pred_onset.min():.1f} - {y_pred_onset.max():.1f} days"
            }
            
            self.logger.info(f"Onset day model - Test R² Score: {test_score_onset:.3f}, RMSE: {rmse:.2f} days")
        else:
            self.logger.warning("onset_day column not found. Skipping onset day model training.")
        
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
        
        # Add onset day predictions
        if 'onset_day' in self.models:
            predictions['onset_day_prediction'] = self.models['onset_day'].predict(X)
            # Round to nearest day and ensure minimum of 1 day
            predictions['onset_day_prediction'] = np.maximum(1, np.round(predictions['onset_day_prediction']))
        
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
        
        # Add onset day prediction
        if 'onset_day' in self.models:
            onset_prediction = self.models['onset_day'].predict(X)[0]
            # Round to nearest day and ensure minimum of 1 day
            onset_days = max(1, round(onset_prediction))
            
            results['onset_day'] = {
                'days_until_breakdown': int(onset_days),
                'severity_level': self._get_onset_severity(onset_days),
                'raw_prediction': float(onset_prediction)
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
    
    def _get_onset_severity(self, days: int) -> str:
        """
        Convert onset days to severity level description.
        
        Args:
            days: Number of days until potential breakdown
            
        Returns:
            Severity level description
        """
        if days <= 7:
            return 'Critical - Immediate Attention Required'
        elif days <= 14:
            return 'High Urgency - Schedule Intervention Soon'
        elif days <= 30:
            return 'Moderate Urgency - Monitor Closely'
        elif days <= 60:
            return 'Low Urgency - Preventive Measures Recommended'
        else:
            return 'Stable - Continue Regular Monitoring'
    
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