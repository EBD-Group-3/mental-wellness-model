# Custom Data Training Guide

## CSV File Format

Your custom training data CSV file should include the following columns:

### Required Feature Columns:
- `age`: Age of the individual (numeric, 18-100)
- `sleep_hours`: Hours of sleep per night (numeric, 0-24)
- `exercise_minutes`: Exercise minutes per week (numeric, 0-1440)
- `work_stress_level`: Work stress level (numeric, 1-10)
- `mood_rating`: Mood rating (numeric, 1-10)  
- `energy_level`: Energy level (numeric, 1-10)
- `avg_heart_rate`: Average heart rate in beats per minute (numeric, 50-200)
- `resting_heart_rate`: Resting heart rate in beats per minute (numeric, 40-120)

### Required Target Columns:
- `depression_risk`: Depression risk label (0 = No Risk, 1 = Risk)
- `anxiety_risk`: Anxiety risk label (0 = No Risk, 1 = Risk)
- `onset_day`: Number of days until potential mental breakdown (1-365 days)

## Sample CSV Format:
```csv
age,sleep_hours,exercise_minutes,work_stress_level,mood_rating,energy_level,avg_heart_rate,resting_heart_rate,depression_risk,anxiety_risk,onset_day
25,8.2,150,3,8,8,65,60,0,0,180
30,7.1,90,5,6,6,70,65,0,0,120
35,5.5,30,8,3,3,85,75,1,1,14
```

## API Usage

### 1. Place your CSV file in the data folder
- Local development: `./data/your_file.csv`
- Docker: Files are automatically mounted to `/app/data/`

### 2. Train with custom data via API:
```bash
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "random_forest",
       "test_size": 0.2,
       "use_sample_data": false,
       "data_file": "your_file.csv"
     }'
```

### 3. Train with custom data via PowerShell:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/train" -Method POST -ContentType "application/json" -Body '{
  "model_type": "random_forest",
  "test_size": 0.2,
  "use_sample_data": false,
  "data_file": "your_file.csv"
}'
```

## CLI Usage

You can also use the CLI to train with custom data:
```bash
python cli.py train --data ./data/your_file.csv --output my_custom_model.joblib
```

## Data Quality Guidelines

1. **Minimum rows**: At least 100 rows for meaningful training
2. **Balanced data**: Include both positive (1) and negative (0) examples for both depression_risk and anxiety_risk
3. **Valid ranges**: Ensure all values are within the specified ranges
4. **No missing values**: Fill in any missing data before training
5. **Realistic values**: Use medically reasonable heart rate values and logical combinations

## Example Files Provided

- `sample_training_data.csv`: Small sample with 10 rows
- `custom_training_data.csv`: Larger sample with 50 rows for better training

Feel free to use these as templates for your own data!