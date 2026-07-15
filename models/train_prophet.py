"""
HemoSmart - Blood Demand Forecasting (Prophet)
---------------------------------------------------
Trains a Prophet model on simulated blood consumption history.
Runs entirely on CPU -- no GPU required, which is why this is the
primary forecasting model on laptops without a dedicated NVIDIA GPU.

(LSTM is trained separately on Colab/Kaggle GPU and compared against
this as models/train_lstm_colab.ipynb -- see docs/lstm_on_colab.md)

Requires: data/blood_demand.csv (run data/create_dataset.py first)

Run:
    python models/train_prophet.py

Outputs:
    models/prophet_model.pkl
    models/prophet_forecast_plot.png
    models/prophet_metrics.txt
"""

import pandas as pd
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
data = pd.read_csv("data/blood_demand.csv")
data["ds"] = pd.to_datetime(data["ds"])
print("Loaded blood demand data:", data.shape)

# ----------------------------------------------------------------------
# 2. Train/test split (last 30 days held out for evaluation)
# ----------------------------------------------------------------------
train = data.iloc[:-30]
test = data.iloc[-30:]

# ----------------------------------------------------------------------
# 3. Train Prophet
# ----------------------------------------------------------------------
print("Training Prophet...")
model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
)
model.fit(train)

# ----------------------------------------------------------------------
# 4. Evaluate against held-out 30 days
# ----------------------------------------------------------------------
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
predicted = forecast.set_index("ds").loc[test["ds"], "yhat"].values
actual = test["y"].values

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print(f"MAE:  {mae:.2f} units")
print(f"RMSE: {rmse:.2f} units")

with open("models/prophet_metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\n")

# ----------------------------------------------------------------------
# 5. Refit on FULL data and produce the real 7-day forecast
# ----------------------------------------------------------------------
model_full = Prophet(weekly_seasonality=True, yearly_seasonality=True)
model_full.fit(data)
future_7d = model_full.make_future_dataframe(periods=7)
forecast_7d = model_full.predict(future_7d)

print("\n7-Day Blood Demand Forecast:")
print(forecast_7d[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7))

fig = model_full.plot(forecast_7d)
plt.savefig("models/prophet_forecast_plot.png", dpi=150, bbox_inches="tight")
plt.close()

with open("models/prophet_model.pkl", "wb") as f:
    pickle.dump(model_full, f)

print("\nSaved models/prophet_model.pkl -- ready for backend to load.")