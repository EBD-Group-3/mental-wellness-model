# Mental Wellness Model Package
"""
A predictive model for early detection of mental wellness issues,
focusing on depression and anxiety prediction.
"""

__version__ = "0.1.0"
__author__ = "EBD-Group-3"

from .models.predictor import MentalWellnessPredictor
from .controller.processor import DataProcessor
from .utils.feature_engineering import FeatureEngineer

__all__ = ['MentalWellnessPredictor', 'DataProcessor', 'FeatureEngineer']