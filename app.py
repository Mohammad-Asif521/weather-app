import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. Generate Synthetic Weather Data
# -----------------------------
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
temperature = 25 + np.sin(np.linspace(0, 12, 365)) * 10 + np.random.normal(0, 2, 365)
df = pd.DataFrame({"Date": dates, "Temperature": temperature})
df.set_index("Date", inplace=True)

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.title("🌤 Weather Forecasting using ARIMA")
st.write("This demo generates synthetic temperature data and forecasts future values using ARIMA.")

# Show dataset
st.subheader("Synthetic Dataset")
st.line_chart(df["Temperature"])

# -----------------------------
# 3. Train-Test Split
# -----------------------------
train = df.iloc[:-30]
test = df.iloc[-30:]

# -----------------------------
# 4. Auto ARIMA Model
# -----------------------------
model = auto_arima(train["Temperature"], seasonal=False, trace=False)
forecast = model.predict(n_periods=30)

# -----------------------------
# 5. Evaluation
# -----------------------------
rmse = np.sqrt(mean_squared_error(test["Temperature"], forecast))
st.subheader("Model Evaluation")
st.write(f"**RMSE:** {rmse:.2f}")

# -----------------------------
# 6. Visualization
# -----------------------------
st.subheader("Forecast vs Actual")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(test.index, test["Temperature"], label="Actual")
ax.plot(test.index, forecast, label="Forecast", linestyle="--")
ax.legend()
st.pyplot(fig)

# -----------------------------
# 7. Future Forecast
# -----------------------------
future_forecast = model.predict(n_periods=15)
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=15, freq="D")
future_df = pd.DataFrame({"Date": future_dates, "Forecasted_Temperature": future_forecast})
future_df.set_index("Date", inplace=True)

st.subheader("Future Forecast (Next 15 Days)")
st.line_chart(future_df["Forecasted_Temperature"])

