"""
HemoSmart - Dataset Simulation
--------------------------------
Generates two datasets since real hospital patient data cannot be used
(privacy / HIPAA constraints):

1. patient_data.csv   -> for the transfusion prediction model (XGBoost)
2. blood_demand.csv   -> for the forecasting models (Prophet / LSTM)

Both are simulated using medically plausible ranges (based on WHO
reference ranges), NOT real patient records.

Run:
    python data/create_dataset.py
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# ----------------------------------------------------------------------
# 1. Patient transfusion dataset
# ----------------------------------------------------------------------
N_PATIENTS = 5000

# Most patients are clinically "normal" with a minority presenting
# abnormal values -> use normal distributions centered on healthy
# reference values (clipped to valid physiological ranges), instead
# of uniform, which unrealistically overloads the abnormal end.
hemoglobin = np.clip(np.random.normal(12.5, 2.2, N_PATIENTS), 5.0, 16.0)
platelets = np.clip(np.random.normal(250000, 70000, N_PATIENTS), 50000, 400000).astype(int)
inr = np.clip(np.random.normal(1.1, 0.35, N_PATIENTS), 0.8, 4.0)
age = np.random.randint(18, 90, N_PATIENTS)
surgery_type = np.random.choice(
    ["Cardiac", "Orthopedic", "General", "Emergency"],
    N_PATIENTS,
    p=[0.20, 0.25, 0.40, 0.15],  # Emergency should be a minority of cases
)

# Transfusion label is derived from clinically meaningful thresholds
# (WHO guideline style rules) instead of being pure random noise,
# so the model has real patterns to learn.
transfusion_needed = (
    (hemoglobin < 8.0) |
    (platelets < 80000) |
    (inr > 1.5) |
    ((surgery_type == "Emergency") & (hemoglobin < 10.0))
).astype(int)

patient_data = pd.DataFrame({
    "hemoglobin": np.round(hemoglobin, 2),
    "platelets": platelets,
    "INR": np.round(inr, 2),
    "age": age,
    "surgery_type": surgery_type,
    "transfusion_needed": transfusion_needed,
})

patient_data.to_csv("data/patient_data.csv", index=False)
print(f"patient_data.csv created -> {patient_data.shape[0]} rows")
print("Transfusion rate:", round(patient_data['transfusion_needed'].mean() * 100, 1), "%")

# ----------------------------------------------------------------------
# 2. Blood demand time-series dataset (2 years, daily)
# ----------------------------------------------------------------------
dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
n_days = len(dates)

# Base demand + weekly seasonality (higher on weekdays, scheduled surgeries)
# + yearly seasonality (slight seasonal wave) + random noise
day_of_week_effect = np.where(pd.Series(dates).dt.dayofweek < 5, 5, -3)
yearly_wave = 5 * np.sin(np.arange(n_days) * (2 * np.pi / 365))
weekly_wave = 3 * np.sin(np.arange(n_days) * (2 * np.pi / 7))
noise = np.random.normal(0, 4, n_days)

base_demand = 30
units_used = (
    base_demand + day_of_week_effect + yearly_wave + weekly_wave + noise
)
units_used = np.clip(units_used, 5, None).round().astype(int)

blood_demand = pd.DataFrame({
    "ds": dates,       # Prophet requires columns named 'ds' and 'y'
    "y": units_used,
})

blood_demand.to_csv("data/blood_demand.csv", index=False)
print(f"blood_demand.csv created -> {blood_demand.shape[0]} rows")
print("Average daily units used:", round(blood_demand['y'].mean(), 1))