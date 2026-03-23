# 📈 TemporalNet — Time-Series Forecasting with Transformer Architectures

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14%2B-4051B5?style=for-the-badge)](https://www.statsmodels.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

> **A research-grade, interactive time-series forecasting dashboard.**
> Compare multiple statistical and ML models side-by-side with zero API keys.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        TEMPORALNET PIPELINE                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐  │
│   │  Data Input  │────▶│ Feature Engineer  │────▶│   Models    │  │
│   │             │     │                  │     │             │  │
│   │ • Demo Data │     │ • Lag Features   │     │ • ARIMA     │  │
│   │ • CSV Upload│     │ • Rolling Stats  │     │ • Holt-Win  │  │
│   │             │     │ • Seasonality    │     │ • Lin. Reg  │  │
│   │             │     │ • Stationarity   │     │ • Mov. Avg  │  │
│   └─────────────┘     └──────────────────┘     └──────┬──────┘  │
│                                                       │         │
│                                                       ▼         │
│   ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐  │
│   │  Dashboard   │◀───│   Visualizer     │◀───│  Benchmark  │  │
│   │             │     │                  │     │             │  │
│   │ • Forecast  │     │ • Forecast Plot  │     │ • MAE       │  │
│   │ • Compare   │     │ • Comparison Bar │     │ • RMSE      │  │
│   │ • Decompose │     │ • Decomposition  │     │ • MAPE      │  │
│   │ • ACF/PACF  │     │ • ACF/PACF Plot  │     │             │  │
│   └─────────────┘     └──────────────────┘     └─────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Supported Models

| Model | Type | Description |
|-------|------|-------------|
| **ARIMA** | Statistical | Auto-Regressive Integrated Moving Average — captures trends and autocorrelation |
| **Holt-Winters** | Statistical | Exponential Smoothing with trend and seasonal components |
| **Linear Regression** | ML | Trend + cyclical (sin/cos) features with OLS regression |
| **Moving Average** | Statistical | Exponential Moving Average with trend projection |

All models produce **point forecasts**, **95% confidence intervals**, and **evaluation metrics** (MAE, RMSE, MAPE).

---

## ✨ Features

- **Zero API Keys** — Fully local computation, no external services needed
- **Interactive Dashboard** — Streamlit-powered UI with dark theme
- **Multi-Model Benchmarking** — Run all models simultaneously and compare results
- **Feature Engineering** — Lag features, rolling statistics, seasonality decomposition
- **Stationarity Analysis** — Augmented Dickey-Fuller test with ACF/PACF visualization
- **Time Series Decomposition** — Trend, seasonal, and residual separation
- **CSV Upload** — Bring your own data with automatic column detection
- **Publication-Quality Charts** — Plotly dark-themed visualizations with JetBrains Mono font

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Time-Series-Forecasting-with-Transformer-Architectures.git
cd Time-Series-Forecasting-with-Transformer-Architectures
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## 📊 Demo Datasets

| Dataset | Description | Size | Key Patterns |
|---------|-------------|------|-------------|
| **Stock Price** | Simulated equity price via Geometric Brownian Motion | 500 days | Trend, volatility clustering, weekly seasonality |
| **Daily Temperature** | Simulated weather station readings | 730 days | Strong annual cycle, autocorrelated noise, slight warming trend |
| **Energy Consumption** | Simulated grid demand | 365 days | Weekly pattern (weekend dip), seasonal heating/cooling, growing demand |

---

## 📁 Project Structure

```
Time-Series-Forecasting-with-Transformer-Architectures/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore                # Git ignore rules
├── src/
│   ├── __init__.py           # Package init
│   ├── models.py             # Forecasting models (ARIMA, Holt-Winters, etc.)
│   ├── feature_engineering.py# Feature extraction and analysis
│   ├── visualizer.py         # Plotly chart generators
│   └── sample_data.py        # Synthetic dataset generators
├── data/                     # Data directory (for user datasets)
└── assets/                   # Static assets
```

---

## 🛠️ Technical Details

### Feature Engineering Pipeline

- **Lag Features**: lag_1, lag_7, lag_14, lag_30 — capture autocorrelation structure
- **Rolling Statistics**: Mean and standard deviation for windows of 7, 14, 30 — smooth noise and capture local trends
- **Calendar Features**: Day of week, month — encode cyclical temporal patterns
- **Decomposition**: STL decomposition into trend, seasonal, and residual — isolate signal components

### Model Evaluation

All models are evaluated on a holdout window equal to the forecast horizon using:

- **MAE** (Mean Absolute Error) — interpretable, robust to outliers
- **RMSE** (Root Mean Squared Error) — penalizes large errors
- **MAPE** (Mean Absolute Percentage Error) — scale-independent comparison

---

## 📄 License

MIT License

Copyright (c) 2026 Yogesh Kuchimanchi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
