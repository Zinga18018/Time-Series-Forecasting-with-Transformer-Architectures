"""
TemporalNet — Time Series Forecasting Models
=============================================
Multiple statistical and ML-based forecasting models with unified interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and MAPE for forecast evaluation."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    n = min(len(actual), len(predicted))
    actual = actual[:n]
    predicted = predicted[:n]

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # MAPE — guard against zero division
    mask = actual != 0
    if mask.any():
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = float("inf")

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}


class TimeSeriesForecaster:
    """Unified interface for multiple time-series forecasting models."""

    def __init__(self):
        self.results: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # ARIMA
    # ------------------------------------------------------------------
    def fit_arima(
        self,
        series: pd.Series,
        horizon: int = 30,
        order: Tuple[int, int, int] = (5, 1, 0),
    ) -> dict:
        """Fit ARIMA model and produce forecasts with confidence intervals."""
        try:
            train = series.values[: -horizon] if horizon < len(series) else series.values
            test = series.values[-horizon:] if horizon < len(series) else series.values[-horizon:]

            model = ARIMA(train, order=order)
            fitted = model.fit()

            forecast_obj = fitted.get_forecast(steps=horizon)
            predictions = forecast_obj.predicted_mean
            ci = forecast_obj.conf_int(alpha=0.05)

            lower = ci[:, 0] if ci.ndim == 2 else predictions * 0.95
            upper = ci[:, 1] if ci.ndim == 2 else predictions * 1.05

            metrics = _compute_metrics(test, predictions)

            result = {
                "model_name": f"ARIMA{order}",
                "predictions": predictions,
                "lower": lower,
                "upper": upper,
                "metrics": metrics,
            }
            self.results["ARIMA"] = result
            return result

        except Exception as e:
            return self._fallback_result("ARIMA", series, horizon, str(e))

    # ------------------------------------------------------------------
    # Exponential Smoothing (Holt-Winters)
    # ------------------------------------------------------------------
    def fit_exponential_smoothing(
        self,
        series: pd.Series,
        horizon: int = 30,
        seasonal_periods: int = 7,
    ) -> dict:
        """Fit Holt-Winters Exponential Smoothing model."""
        try:
            train = series.values[: -horizon] if horizon < len(series) else series.values
            test = series.values[-horizon:] if horizon < len(series) else series.values[-horizon:]

            # Choose additive vs multiplicative based on data
            seasonal = "add" if (series.min() <= 0) else "mul"
            trend = "add"

            sp = min(seasonal_periods, len(train) // 3)
            if sp < 2:
                sp = 2

            model = ExponentialSmoothing(
                train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=sp,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)

            predictions = fitted.forecast(horizon)
            residual_std = np.std(fitted.resid)
            lower = predictions - 1.96 * residual_std
            upper = predictions + 1.96 * residual_std

            metrics = _compute_metrics(test, predictions)

            result = {
                "model_name": "Holt-Winters",
                "predictions": predictions,
                "lower": lower,
                "upper": upper,
                "metrics": metrics,
            }
            self.results["Holt-Winters"] = result
            return result

        except Exception as e:
            return self._fallback_result("Holt-Winters", series, horizon, str(e))

    # ------------------------------------------------------------------
    # Linear Regression (Trend + Seasonality)
    # ------------------------------------------------------------------
    def fit_linear_regression(
        self,
        series: pd.Series,
        horizon: int = 30,
    ) -> dict:
        """Fit Linear Regression with trend and cyclical features."""
        try:
            n = len(series)
            train_n = n - horizon if horizon < n else n
            values = series.values

            # Feature matrix: time index, sin/cos seasonality
            t = np.arange(n + horizon).reshape(-1, 1)
            sin_7 = np.sin(2 * np.pi * t / 7)
            cos_7 = np.cos(2 * np.pi * t / 7)
            sin_30 = np.sin(2 * np.pi * t / 30)
            cos_30 = np.cos(2 * np.pi * t / 30)
            sin_365 = np.sin(2 * np.pi * t / 365)
            cos_365 = np.cos(2 * np.pi * t / 365)

            X = np.hstack([t, sin_7, cos_7, sin_30, cos_30, sin_365, cos_365])

            X_train = X[:train_n]
            y_train = values[:train_n]
            X_future = X[n : n + horizon]

            model = LinearRegression()
            model.fit(X_train, y_train)

            predictions = model.predict(X_future)

            # Confidence interval based on training residuals
            train_pred = model.predict(X_train)
            residual_std = np.std(y_train - train_pred)
            lower = predictions - 1.96 * residual_std
            upper = predictions + 1.96 * residual_std

            test = values[-horizon:] if horizon < n else values
            metrics = _compute_metrics(test, predictions)

            result = {
                "model_name": "Linear Regression",
                "predictions": predictions,
                "lower": lower,
                "upper": upper,
                "metrics": metrics,
            }
            self.results["Linear Regression"] = result
            return result

        except Exception as e:
            return self._fallback_result("Linear Regression", series, horizon, str(e))

    # ------------------------------------------------------------------
    # Moving Average
    # ------------------------------------------------------------------
    def fit_moving_average(
        self,
        series: pd.Series,
        horizon: int = 30,
        window: int = 7,
    ) -> dict:
        """Simple Moving Average + Exponential Moving Average forecast."""
        try:
            values = series.values
            n = len(values)
            test = values[-horizon:] if horizon < n else values

            # Exponential moving average on full series
            alpha = 2 / (window + 1)
            ema = np.zeros(n)
            ema[0] = values[0]
            for i in range(1, n):
                ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]

            # Forecast: project last EMA trend forward
            last_ema = ema[-1]
            if n > window:
                trend = (ema[-1] - ema[-window]) / window
            else:
                trend = 0

            predictions = np.array([last_ema + trend * (i + 1) for i in range(horizon)])

            residuals = values - ema
            residual_std = np.std(residuals)
            lower = predictions - 1.96 * residual_std
            upper = predictions + 1.96 * residual_std

            metrics = _compute_metrics(test, predictions)

            result = {
                "model_name": f"Moving Avg (w={window})",
                "predictions": predictions,
                "lower": lower,
                "upper": upper,
                "metrics": metrics,
            }
            self.results["Moving Average"] = result
            return result

        except Exception as e:
            return self._fallback_result("Moving Average", series, horizon, str(e))

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    def benchmark(
        self,
        series: pd.Series,
        horizon: int = 30,
        models_to_run: Optional[List[str]] = None,
    ) -> Dict[str, dict]:
        """Run selected models and return a comparison dictionary."""
        if models_to_run is None:
            models_to_run = ["ARIMA", "Holt-Winters", "Linear Regression", "Moving Average"]

        dispatch = {
            "ARIMA": lambda: self.fit_arima(series, horizon),
            "Holt-Winters": lambda: self.fit_exponential_smoothing(series, horizon),
            "Linear Regression": lambda: self.fit_linear_regression(series, horizon),
            "Moving Average": lambda: self.fit_moving_average(series, horizon),
        }

        results = {}
        for name in models_to_run:
            if name in dispatch:
                results[name] = dispatch[name]()

        self.results = results
        return results

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    def _fallback_result(self, name: str, series: pd.Series, horizon: int, error: str) -> dict:
        """Return a naive forecast when a model fails."""
        last_val = series.values[-1] if len(series) > 0 else 0
        predictions = np.full(horizon, last_val)
        return {
            "model_name": f"{name} (fallback)",
            "predictions": predictions,
            "lower": predictions * 0.9,
            "upper": predictions * 1.1,
            "metrics": {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")},
            "error": error,
        }
