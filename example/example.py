import numpy as np
from nbeats_tensorflow.model import NBeats

# Generate a time series
N = 1000
t = np.linspace(0, 1, N)
trend = 30 + 20 * t + 10 * (t ** 2)
seasonality = 5 * np.cos(2 * np.pi * (10 * t - 0.5))
noise = np.random.normal(0, 1, N)
y = trend + seasonality + noise

# Fit the model
model = NBeats(
    y=y,
    forecast_period=200,
    lookback_period=400,
    units=30,
    stacks=['trend', 'seasonality'],
    num_trend_coefficients=3,
    num_seasonal_coefficients=5,
    num_blocks_per_stack=2,
    share_weights=True,
    share_coefficients=False,
)

model.fit(
    loss='mse',
    epochs=100,
    batch_size=32,
    learning_rate=0.003,
    backcast_loss_weight=0.5,
    verbose=True
)

# Plot the in-sample predictions
predictions = model.predict(index=800, return_backcast=True)
fig = model.plot_predictions()
fig.write_image('predictions.png', width=650, height=400)

# Plot the out-of-sample forecasts
forecasts = model.forecast(return_backcast=True)
fig = model.plot_forecasts()
fig.write_image('forecasts.png', width=650, height=400)
