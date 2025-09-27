# Onset Day Prediction Feature

## Overview
The mental wellness model now includes a new **onset_day** prediction feature that estimates how many days from today a potential mental breakdown might occur. This feature provides temporal insights into mental health risks, enabling proactive intervention planning.

## How It Works

### 1. **Data Generation**
- The onset_day target is generated based on simplified risk factors:
  - Depression risk probability (60% weight)
  - Anxiety risk probability (40% weight)

### 2. **Model Training**
- Uses **Random Forest Regressor** (for random_forest model type)
- Uses **Linear Regression** (for logistic_regression model type)
- Trained alongside depression and anxiety classification models
- Predicts continuous values from 1 to 365 days

### 3. **Prediction Output**
The onset_day prediction returns:
- `days_until_breakdown`: Integer number of days (minimum 1)
- `severity_level`: Categorical urgency level
- `raw_prediction`: Raw model output before rounding

## Severity Levels

| Days Until Breakdown | Severity Level | Action Required |
|---------------------|----------------|----------------|
| 1-7 days | Critical - Immediate Attention Required | Emergency intervention |
| 8-14 days | High Urgency - Schedule Intervention Soon | Schedule within 1-2 days |
| 15-30 days | Moderate Urgency - Monitor Closely | Weekly check-ins |
| 31-60 days | Low Urgency - Preventive Measures Recommended | Monthly monitoring |
| 61+ days | Stable - Continue Regular Monitoring | Routine care |

## API Usage

### Individual Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 35,
       "sleep_hours": 5,
       "exercise_minutes": 30,
       "work_stress_level": 8,
       "mood_rating": 3,
       "energy_level": 3,
       "avg_heart_rate": 85,
       "resting_heart_rate": 75
     }'
```

### Response Format
```json
{
  "depression": {
    "prediction": false,
    "probability": 0.26,
    "risk_level": "Low Risk"
  },
  "anxiety": {
    "prediction": false,
    "probability": 0.28,
    "risk_level": "Low Risk"
  },
  "onset_day": {
    "days_until_breakdown": 1,
    "severity_level": "Critical - Immediate Attention Required",
    "raw_prediction": 0.46
  }
}
```

## Training Data Requirements

Your CSV training data must now include the `onset_day` column:

```csv
age,sleep_hours,exercise_minutes,work_stress_level,mood_rating,energy_level,avg_heart_rate,resting_heart_rate,depression_risk,anxiety_risk,onset_day
25,8,150,3,8,8,65,60,0,0,180
35,5,30,8,3,3,85,75,1,1,14
```

## Use Cases

1. **Crisis Prevention**: Identify individuals at risk of immediate breakdown
2. **Resource Allocation**: Prioritize mental health resources based on urgency
3. **Intervention Planning**: Schedule appropriate interventions based on timeline
4. **Progress Monitoring**: Track changes in breakdown timeline over time
5. **Early Warning System**: Alert healthcare providers of deteriorating conditions

## Model Performance Metrics

The onset_day model is evaluated using regression metrics:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error (in days)
- **MAE**: Mean Absolute Error (in days)
- **Cross-validation**: 5-fold CV for robust evaluation

## Integration with Existing Features

The onset_day prediction complements existing depression and anxiety predictions:
- High depression/anxiety risk often correlates with fewer onset days
- Combined predictions provide comprehensive mental health assessment
- All three predictions use the same input features for consistency

## Technical Implementation

- **Regression Models**: RandomForestRegressor, LinearRegression
- **Feature Engineering**: Uses same engineered features as classification models
- **Data Validation**: Ensures onset_day values are between 1-365 days
- **API Integration**: Seamlessly integrated with existing prediction endpoints
- **Testing**: Comprehensive unit tests verify functionality

This feature transforms the mental wellness model from a static risk assessment tool into a dynamic temporal predictor, enabling more proactive and timely mental health interventions.