# Engine Predictive Maintenance Model

## Model Description
Sklearn pipeline (StandardScaler + LOGREG) for engine predictive maintenance.

## Performance
- F1 Score: 0.7657

## Features
- engine_rpm
- lub_oil_pressure
- fuel_pressure
- coolant_pressure
- lub_oil_temperature
- coolant_temperature

## Target
- engine_condition (0=Normal, 1=Needs Maintenance)

## Usage
```python
import joblib
model = joblib.load('best_model.joblib')
prediction = model.predict(X)
```
