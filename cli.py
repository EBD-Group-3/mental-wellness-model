#!/usr/bin/env python3
"""
Command line interface for Mental Wellness Prediction Model.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer
from mental_wellness_model.config import load_config


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def train_model(args):
    """Train the mental wellness prediction model."""
    config = load_config(args.config)
    setup_logging(config.get('logging', {}).get('level', 'INFO'))
    
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    # Initialize components
    processor = DataProcessor()
    engineer = FeatureEngineer()
    predictor = MentalWellnessPredictor(model_type=config['model']['type'])
    
    # Load and process data
    if args.data:
        df = processor.load_data(data_path=args.data)
    else:
        df = processor.load_data()  # Use sample data
        logger.info(f"Generated {len(df)} samples of synthetic data")
    
    # Clean and validate data
    df = processor.clean_data(df)
    if not processor.validate_data(df):
        logger.error("Data validation failed!")
        return
    
    # Feature engineering
    df = engineer.create_features(df)
    engineer.fit_scaler(df)
    df = engineer.transform_features(df)
    
    # Train model
    results = predictor.train(df, test_size=config['model']['test_size'])
    
    # Print results
    for condition, metrics in results.items():
        logger.info(f"\n{condition.upper()} MODEL RESULTS:")
        logger.info(f"  Test Accuracy: {metrics['test_accuracy']:.3f}")
        logger.info(f"  AUC Score: {metrics['auc_score']:.3f}")
        logger.info(f"  CV Mean Accuracy: {metrics['cv_mean_accuracy']:.3f} (+/- {metrics['cv_std_accuracy']:.3f})")
    
    # Save model if requested
    if args.output:
        predictor.save_model(args.output)
        logger.info(f"Model saved to {args.output}")


def predict_individual(args):
    """Make prediction for an individual."""
    config = load_config(args.config)
    setup_logging(config.get('logging', {}).get('level', 'INFO'))
    
    logger = logging.getLogger(__name__)
    
    # Load model
    predictor = MentalWellnessPredictor()
    try:
        predictor.load_model(args.model)
        logger.info(f"Model loaded from {args.model}")
    except FileNotFoundError:
        logger.error(f"Model file not found: {args.model}")
        return
    
    # Get features from command line arguments
    features = {
        'age': args.age,
        'sleep_hours': args.sleep_hours,
        'exercise_minutes': args.exercise_minutes,
        'avg_heart_rate': args.avg_heart_rate,
        'work_stress_level': args.work_stress_level,
        'mood_score': args.mood_score,
        'fitness_level': args.fitness_level,
        'resting_heart_rate': args.resting_heart_rate,
    }
    
    # Make prediction
    results = predictor.predict_individual(features)
    
    # Display results
    print("\n=== MENTAL WELLNESS PREDICTION RESULTS ===")
    for condition, result in results.items():
        print(f"\n{condition.upper()}:")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Probability: {result['probability']:.3f}")
        print(f"  Prediction: {'At Risk' if result['prediction'] else 'Low Risk'}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Mental Wellness Prediction Model CLI")
    parser.add_argument('--config', '-c', help='Path to configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the prediction model')
    train_parser.add_argument('--data', '-d', help='Path to training data CSV file')
    train_parser.add_argument('--output', '-o', help='Path to save trained model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make prediction for an individual')
    predict_parser.add_argument('--model', '-m', required=True, help='Path to trained model file')
    predict_parser.add_argument('--age', type=float, default=30, help='Age')
    predict_parser.add_argument('--sleep-hours', type=float, default=7, help='Hours of sleep per night')
    predict_parser.add_argument('--exercise-minutes', type=int, default=90, help='Exercise minutes per week')
    predict_parser.add_argument('--avg-heart-rate', type=float, default=70, help='Average heart rate (beats per minute)')
    predict_parser.add_argument('--work-stress-level', type=float, default=5, help='Work stress level (1-10)')
    predict_parser.add_argument('--mood-score', type=float, default=6, help='Mood score (1-10)')
    predict_parser.add_argument('--fitness-level', type=float, default=6, help='Fitness level (1-10)')
    predict_parser.add_argument('--resting-heart-rate', type=float, default=65, help='Resting heart rate (beats per minute)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'predict':
        predict_individual(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()