"""
TemporalNet — Sample / Demo Datasets
======================================
Synthetic time series generators for stocks, weather, and energy consumption.
"""

import numpy as np
import pandas as pd


def generate_stock_data(days: int = 500) -> pd.DataFrame:
    """
    Simulated stock price with trend, volatility clustering, and seasonal effects.

    Returns DataFrame with columns: date, value
    """
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    # Geometric Brownian Motion base
    drift = 0.0003
    volatility = 0.018
    returns = np.random.normal(drift, volatility, days)

    # Add volatility clustering (GARCH-like)
    vol_factor = np.ones(days)
    for i in range(1, days):
        vol_factor[i] = 0.9 * vol_factor[i - 1] + 0.1 * abs(returns[i - 1]) / volatility
    returns = returns * vol_factor

    # Price from cumulative returns
    price = 150 * np.exp(np.cumsum(returns))

    # Add subtle weekly seasonality
    weekly = 1.5 * np.sin(2 * np.pi * np.arange(days) / 5)
    price += weekly

    return pd.DataFrame({"date": dates, "value": np.round(price, 2)})


def generate_weather_data(days: int = 730) -> pd.DataFrame:
    """
    Simulated daily temperature with strong annual seasonality and noise.

    Returns DataFrame with columns: date, value
    """
    np.random.seed(123)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    t = np.arange(days)

    # Base temperature with annual cycle
    base = 65 + 25 * np.sin(2 * np.pi * (t - 80) / 365.25)

    # Multi-year trend (slight warming)
    trend = 0.003 * t

    # Day-to-day noise (autocorrelated)
    noise = np.zeros(days)
    noise[0] = np.random.normal(0, 3)
    for i in range(1, days):
        noise[i] = 0.7 * noise[i - 1] + np.random.normal(0, 3)

    temp = base + trend + noise

    return pd.DataFrame({"date": dates, "value": np.round(temp, 1)})


def generate_energy_data(days: int = 365) -> pd.DataFrame:
    """
    Simulated daily energy consumption with daily and weekly patterns.

    Returns DataFrame with columns: date, value
    """
    np.random.seed(77)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    t = np.arange(days)

    # Base load
    base = 500

    # Weekly pattern (lower on weekends)
    day_of_week = t % 7
    weekly_effect = np.where(day_of_week >= 5, -80, 0)  # weekend dip

    # Seasonal (higher in summer/winter for cooling/heating)
    seasonal = 100 * np.cos(2 * np.pi * (t - 180) / 365.25) ** 2

    # Trend (growing demand)
    trend = 0.15 * t

    # Noise
    noise = np.random.normal(0, 25, days)

    consumption = base + weekly_effect + seasonal + trend + noise
    consumption = np.maximum(consumption, 200)  # floor

    return pd.DataFrame({"date": dates, "value": np.round(consumption, 1)})


# Registry of demo datasets 
DEMO_DATASETS = {
    "Stock Price (Simulated)": generate_stock_data,
    "Daily Temperature": generate_weather_data,
    "Energy Consumption": generate_energy_data,
}
