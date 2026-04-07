# TemporalNet -- Time-Series Forecasting Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14%2B-4051B5?style=flat-square)](https://www.statsmodels.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

An interactive time-series forecasting dashboard that compares multiple statistical and ML models side-by-side. Includes automated feature engineering, stationarity testing, and decomposition analysis. Zero API keys required.

---

## Supported Models

| Model | Type | Description |
|-------|------|-------------|
| **ARIMA** | Statistical | Auto-Regressive Integrated Moving Average -- captures trends and autocorrelation |
| **Holt-Winters** | Statistical | Exponential Smoothing with trend and seasonal components |
| **Linear Regression** | ML | Trend + cyclical (sin/cos) features with OLS regression |
| **Moving Average** | Statistical | Exponential Moving Average with trend projection |

All models produce **point forecasts**, **95% confidence intervals**, and **evaluation metrics** (MAE, RMSE, MAPE).

---

## Features

- **Zero API Keys** -- Fully local computation, no external services needed
- **Interactive Dashboard** -- Streamlit-powered UI with dark theme
- **Multi-Model Benchmarking** -- Run all models simultaneously and compare results
- **Feature Engineering** -- Lag features, rolling statistics, seasonality decomposition
- **Stationarity Analysis** -- Augmented Dickey-Fuller test with ACF/PACF visualization
- **Time Series Decomposition** -- Trend, seasonal, and residual separation
- **CSV Upload** -- Bring your own data with automatic column detection

---

## Quick Start

```bash
git clone https://github.com/Zinga18018/Time-Series-Forecasting-with-Transformer-Architectures.git
cd Time-Series-Forecasting-with-Transformer-Architectures
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## Demo Datasets

| Dataset | Description | Size | Key Patterns |
|---------|-------------|------|-------------|
| **Stock Price** | Simulated equity price via Geometric Brownian Motion | 500 days | Trend, volatility clustering, weekly seasonality |
| **Daily Temperature** | Simulated weather station readings | 730 days | Strong annual cycle, autocorrelated noise, slight warming trend |
| **Energy Consumption** | Simulated grid demand | 365 days | Weekly pattern (weekend dip), seasonal heating/cooling, growing demand |

---

## Project Structure

```
Time-Series-Forecasting-with-Transformer-Architectures/
|-- app.py                     # Main Streamlit application
|-- requirements.txt           # Python dependencies
|-- README.md
|-- .gitignore
|-- src/
|   |-- __init__.py
|   |-- models.py              # Forecasting models (ARIMA, Holt-Winters, etc.)
|   |-- feature_engineering.py # Feature extraction and analysis
|   |-- visualizer.py          # Plotly chart generators
|   +-- sample_data.py         # Synthetic dataset generators
```

---

## Technical Details

### Feature Engineering Pipeline

- **Lag Features**: lag_1, lag_7, lag_14, lag_30 -- capture autocorrelation structure
- **Rolling Statistics**: Mean and standard deviation for windows of 7, 14, 30 -- smooth noise and capture local trends
- **Calendar Features**: Day of week, month -- encode cyclical temporal patterns
- **Decomposition**: STL decomposition into trend, seasonal, and residual -- isolate signal components

### Model Evaluation

All models are evaluated on a holdout window equal to the forecast horizon using:

- **MAE** (Mean Absolute Error) -- interpretable, robust to outliers
- **RMSE** (Root Mean Squared Error) -- penalizes large errors
- **MAPE** (Mean Absolute Percentage Error) -- scale-independent comparison

---

## License

MIT License -- Yogesh Kuchimanchi
