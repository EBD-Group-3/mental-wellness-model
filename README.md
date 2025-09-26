# Mental Wellness Prediction Model

A machine learning-based system for early detection and prediction of mental wellness risks, focusing on depression and anxiety. This project addresses the reactive nature of current mental healthcare systems by providing proactive prediction capabilities.

## 🎯 Problem Statement

Mental health disorders affect many people globally, with depression and anxiety being leading causes of disability worldwide. Current mental healthcare systems are predominantly reactive, addressing issues only after symptoms become severe enough to prompt help-seeking behavior. This model aims to enable early detection and intervention.

## 🚀 Features

- **Predictive Models**: Machine learning models for depression and anxiety risk prediction
- **Feature Engineering**: Comprehensive feature extraction from mental wellness indicators
- **Data Processing**: Robust data cleaning, validation, and preprocessing pipelines
- **Risk Assessment**: Multi-level risk categorization (Low, Moderate, High, Very High)
- **CLI Interface**: Command-line tool for training models and making predictions
- **Configurable**: YAML-based configuration system
- **Scalable**: Designed for big data processing and real-time predictions

## 📊 Supported Features

The model processes the following mental wellness indicators:

- **Demographics**: Age, lifestyle factors
- **Sleep Patterns**: Sleep hours, quality metrics
- **Physical Activity**: Exercise frequency
- **Social Factors**: Social interaction scores
- **Stress Levels**: Work stress, financial stress
- **Psychological Indicators**: Mood ratings, energy levels, concentration difficulty

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/EBD-Group-3/mental-wellness-model.git
cd mental-wellness-model

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## 📚 Quick Start

### 1. Run the Example Demo
```bash
python example.py
```

### 2. Train a Model via CLI
```bash
# Train with synthetic data
python cli.py train --output my_model.joblib

# Train with your own data
python cli.py train --data your_data.csv --output my_model.joblib
```

### 3. Make Predictions
```bash
python cli.py predict --model my_model.joblib \
    --age 30 \
    --sleep-hours 7 \
    --exercise-minutes 90 \
    --work-stress-level 5 \
    --mood-rating 6 \
    --energy-level 6 \
    --avg-heart-rate 70 \
    --resting-heart-rate 65
```

## 💻 Programmatic Usage

```python
from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer

# Initialize components
processor = DataProcessor()
engineer = FeatureEngineer()
predictor = MentalWellnessPredictor(model_type='random_forest')

# Load and process data
df = processor.load_data()  # or load_data('your_data.csv')
df = processor.clean_data(df)

# Feature engineering
df = engineer.create_features(df)
engineer.fit_scaler(df)
df = engineer.transform_features(df)

# Train model
results = predictor.train(df)

# Make predictions
individual_features = {
    'age': 30,
    'sleep_hours': 7,
    'exercise_minutes': 90,
    'work_stress_level': 5,
    'mood_rating': 6,
    'energy_level': 6,
    'avg_heart_rate': 70,
    'resting_heart_rate': 65
}

predictions = predictor.predict_individual(individual_features)
print(predictions)
```

## 📁 Project Structure

```
mental-wellness-model/
├── mental_wellness_model/          # Main package
│   ├── data/                      # Data processing modules
│   │   ├── __init__.py
│   │   └── processor.py
│   ├── models/                    # ML models
│   │   ├── __init__.py
│   │   └── predictor.py
│   ├── utils/                     # Utilities
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── config/                    # Configuration
│   │   ├── __init__.py
│   │   └── default.yaml
│   └── __init__.py
├── cli.py                         # Command line interface
├── example.py                     # Usage example
├── requirements.txt               # Dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## ⚙️ Configuration

Modify `mental_wellness_model/config/default.yaml` to customize:

- Model types and parameters
- Feature engineering settings
- Risk assessment thresholds
- Data processing options

## 📈 Model Performance

The system includes built-in evaluation metrics:

- **Accuracy**: Classification accuracy on test data
- **AUC Score**: Area Under the ROC Curve
- **Cross-Validation**: 5-fold cross-validation scores
- **Feature Importance**: Ranking of most predictive features

## 🔬 Methodology

### Data Processing
1. **Data Validation**: Ensures data quality and completeness
2. **Cleaning**: Handles missing values and outliers
3. **Feature Engineering**: Creates composite and interaction features

### Machine Learning
1. **Multi-target Prediction**: Separate models for depression and anxiety
2. **Algorithms**: Random Forest and Logistic Regression
3. **Class Balancing**: Handles imbalanced datasets
4. **Cross-Validation**: Robust model evaluation

### Risk Assessment
- **Probability-based**: Uses prediction probabilities for nuanced risk levels
- **Interpretable**: Clear risk categories with actionable thresholds
- **Comprehensive**: Covers both depression and anxiety risks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This model is designed for research and early screening purposes only. It should not be used as a substitute for professional mental health diagnosis or treatment. Always consult qualified healthcare professionals for mental health concerns.

## 📞 Support

For questions, issues, or contributions, please open an issue on GitHub or contact the EBD-Group-3 team.
