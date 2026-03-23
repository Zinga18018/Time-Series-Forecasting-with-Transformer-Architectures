"""
TemporalNet — Feature Engineering for Time Series
===================================================
Lag features, rolling statistics, seasonality detection, and stationarity tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings("ignore")


def create_features(series: pd.Series) -> pd.DataFrame:
    """
    Generate engineered features from a univariate time series.

    Returns a DataFrame with:
      - lag_1, lag_7, lag_14, lag_30
      - rolling_mean_7, rolling_std_7
      - rolling_mean_14, rolling_std_14
      - rolling_mean_30, rolling_std_30
      - day_of_week, month (if datetime index)
      - trend, seasonal, residual (STL decomposition)
    """
    df = pd.DataFrame({"value": series.values}, index=series.index)

    # Lag features 
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df["value"].shift(lag)

    # Rolling statistics 
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = df["value"].rolling(window=window).mean()
        df[f"rolling_std_{window}"] = df["value"].rolling(window=window).std()

    # Calendar features 
    if hasattr(df.index, "dayofweek"):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
    else:
        df["day_of_week"] = np.arange(len(df)) % 7
        df["month"] = (np.arange(len(df)) // 30) % 12 + 1

    # Decomposition (trend / seasonal / residual) 
    try:
        period = detect_seasonality(series)
        if period < 2:
            period = 7
        decomp = seasonal_decompose(
            series.dropna(), model="additive", period=min(period, len(series) // 2)
        )
        df["trend"] = decomp.trend.values if len(decomp.trend) == len(df) else np.nan
        df["seasonal"] = decomp.seasonal.values if len(decomp.seasonal) == len(df) else np.nan
        df["residual"] = decomp.resid.values if len(decomp.resid) == len(df) else np.nan
    except Exception:
        df["trend"] = np.nan
        df["seasonal"] = np.nan
        df["residual"] = np.nan

    return df


def detect_seasonality(series: pd.Series, max_lag: int = 90) -> int:
    """
    Auto-detect the dominant seasonal period using ACF peaks.

    Returns the lag (integer) with the highest ACF value (excluding lag 0).
    Falls back to 7 if detection fails.
    """
    try:
        clean = series.dropna().values.astype(float)
        if len(clean) < max_lag + 1:
            max_lag = len(clean) // 2

        acf_values = acf(clean, nlags=max_lag, fft=True)
        # Skip lag 0 (always 1.0), find dominant peak
        acf_subset = acf_values[2:]  # skip 0 and 1
        if len(acf_subset) == 0:
            return 7

        peaks = []
        for i in range(1, len(acf_subset) - 1):
            if acf_subset[i] > acf_subset[i - 1] and acf_subset[i] > acf_subset[i + 1]:
                peaks.append((i + 2, acf_subset[i]))  # +2 because we skipped first 2

        if peaks:
            best_lag = max(peaks, key=lambda x: x[1])[0]
            return max(best_lag, 2)
        return 7

    except Exception:
        return 7


def stationarity_test(series: pd.Series) -> Dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Returns dict with:
      - test_statistic
      - p_value
      - critical_values (1%, 5%, 10%)
      - is_stationary (bool, at 5% level)
      - n_lags_used
      - n_observations
    """
    try:
        clean = series.dropna().values.astype(float)
        result = adfuller(clean, autolag="AIC")

        return {
            "test_statistic": round(result[0], 4),
            "p_value": round(result[1], 6),
            "n_lags_used": result[2],
            "n_observations": result[3],
            "critical_values": {k: round(v, 4) for k, v in result[4].items()},
            "is_stationary": result[1] < 0.05,
        }
    except Exception as e:
        return {
            "test_statistic": None,
            "p_value": None,
            "n_lags_used": None,
            "n_observations": None,
            "critical_values": {},
            "is_stationary": None,
            "error": str(e),
        }


def get_decomposition(
    series: pd.Series, period: int = None
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Decompose series into trend, seasonal, and residual components.

    Returns (trend, seasonal, residual) as pandas Series.
    """
    if period is None:
        period = detect_seasonality(series)
    if period < 2:
        period = 7

    period = min(period, len(series.dropna()) // 2)
    decomp = seasonal_decompose(series.dropna(), model="additive", period=period)
    return decomp.trend, decomp.seasonal, decomp.resid
