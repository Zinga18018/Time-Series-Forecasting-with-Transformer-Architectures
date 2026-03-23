"""
TemporalNet — Time-Series Forecasting with Transformer Architectures
=====================================================================
A Streamlit dashboard for interactive time-series analysis and forecasting.
No API keys required. Fully local computation.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.models import TimeSeriesForecaster
from src.feature_engineering import (
    create_features,
    detect_seasonality,
    stationarity_test,
    get_decomposition,
)
from src.visualizer import (
    forecast_plot,
    model_comparison_bar,
    decomposition_plot,
    acf_plot,
)
from src.sample_data import DEMO_DATASETS

# Page config 
st.set_page_config(
    page_title="TemporalNet | Time Series Forecasting",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');

    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #111111;
        --bg-card: #161616;
        --text-primary: #e0e0e0;
        --accent-green: #00ff88;
        --accent-blue: #00d4ff;
        --accent-purple: #a855f7;
        --accent-amber: #f59e0b;
    }

    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid #222;
    }

    .metric-card {
        background: linear-gradient(135deg, var(--bg-card), #1a1a2e);
        border: 1px solid #2a2a3e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .metric-card:hover {
        border-color: var(--accent-green);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent-green);
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .header-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .header-subtitle {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-card);
        border-radius: 8px;
        padding: 8px 16px;
        color: var(--text-primary);
        border: 1px solid #2a2a3e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a2a1a !important;
        border-color: var(--accent-green) !important;
    }

    div[data-testid="stExpander"] {
        background-color: var(--bg-card);
        border: 1px solid #2a2a3e;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header 
st.markdown('<p class="header-title"> TemporalNet</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="header-subtitle">Time-Series Forecasting with Transformer Architectures &mdash; '
    "Statistical & ML Models &bull; Interactive Analysis &bull; Zero API Keys</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# 
# SIDEBAR
# 
with st.sidebar:
    st.markdown("### Configuration")

    # Data source 
    st.markdown("#### Data Source")
    data_source = st.radio(
        "Choose data source",
        options=["Demo Dataset", "Upload CSV"],
        label_visibility="collapsed",
    )

    series = None
    data_name = ""

    if data_source == "Demo Dataset":
        dataset_name = st.selectbox("Select demo dataset", list(DEMO_DATASETS.keys()))
        df = DEMO_DATASETS[dataset_name]()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        series = df["value"]
        data_name = dataset_name
    else:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            # Attempt to find date and value columns
            date_col = None
            value_col = None
            for col in df.columns:
                if col.lower() in ("date", "datetime", "timestamp", "time", "ds"):
                    date_col = col
                elif col.lower() in ("value", "y", "close", "price", "target"):
                    value_col = col

            if date_col is None:
                date_col = df.columns[0]
            if value_col is None:
                value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col, value_col])
            df = df.set_index(date_col)
            series = df[value_col].astype(float)
            data_name = uploaded.name

    st.markdown("---")

    # Model selection 
    st.markdown("#### Models")
    models_available = ["ARIMA", "Holt-Winters", "Linear Regression", "Moving Average"]
    selected_models = []
    for m in models_available:
        if st.checkbox(m, value=True, key=f"model_{m}"):
            selected_models.append(m)

    st.markdown("---")

    # Forecast horizon 
    st.markdown("#### Forecast Horizon")
    horizon = st.slider("Days to forecast", min_value=7, max_value=90, value=30, step=1)

    st.markdown("---")

    # Feature engineering toggles 
    st.markdown("#### Feature Engineering")
    use_lag_features = st.toggle("Lag Features", value=True)
    use_rolling_stats = st.toggle("Rolling Statistics", value=True)
    use_decomposition = st.toggle("Seasonal Decomposition", value=True)

# 
# MAIN CONTENT
# 
if series is not None and len(series) > 20:

    # Run forecasting models 
    forecaster = TimeSeriesForecaster()
    results = forecaster.benchmark(series, horizon=horizon, models_to_run=selected_models)

    # Find best model 
    best_model = "N/A"
    best_rmse = float("inf")
    for name, res in results.items():
        rmse = res["metrics"].get("RMSE", float("inf"))
        if rmse < best_rmse and not np.isnan(rmse):
            best_rmse = rmse
            best_model = name

    # Metrics row 
    c1, c2, c3, c4 = st.columns(4)

    def metric_card(label: str, value: str, accent: str = "var(--accent-green)"):
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent}">{value}</div>
        </div>
        """

    with c1:
        st.markdown(metric_card("Data Points", f"{len(series):,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(
            metric_card("Forecast Horizon", f"{horizon} days", "var(--accent-blue)"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card("Best Model", best_model, "var(--accent-purple)"),
            unsafe_allow_html=True,
        )
    with c4:
        rmse_str = f"{best_rmse:.2f}" if best_rmse < float("inf") else "N/A"
        st.markdown(
            metric_card("Best RMSE", rmse_str, "var(--accent-amber)"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs 
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [" Forecast", " Model Comparison", " Decomposition", " Stationarity", " Features"]
    )

    # Tab 1: Forecast 
    with tab1:
        st.plotly_chart(
            forecast_plot(series, results, title=f"Forecast — {data_name}"),
            use_container_width=True,
        )
        with st.expander(" Forecast Details"):
            for name, res in results.items():
                st.markdown(f"**{res['model_name']}**")
                metrics = res["metrics"]
                cols = st.columns(3)
                cols[0].metric("MAE", f"{metrics['MAE']:.4f}")
                cols[1].metric("RMSE", f"{metrics['RMSE']:.4f}")
                cols[2].metric("MAPE", f"{metrics['MAPE']:.2f}%")
                st.markdown("---")

    # Tab 2: Model Comparison 
    with tab2:
        metrics_dict = {name: res["metrics"] for name, res in results.items()}
        st.plotly_chart(
            model_comparison_bar(metrics_dict),
            use_container_width=True,
        )

        # Comparison table
        st.markdown("#### Detailed Metrics")
        comparison_df = pd.DataFrame(metrics_dict).T
        comparison_df.index.name = "Model"
        st.dataframe(
            comparison_df.style.highlight_min(axis=0, color="#1a3a1a"),
            use_container_width=True,
        )

    # Tab 3: Decomposition 
    with tab3:
        if use_decomposition:
            try:
                period = detect_seasonality(series)
                trend, seasonal, residual = get_decomposition(series, period=period)
                st.plotly_chart(
                    decomposition_plot(trend, seasonal, residual),
                    use_container_width=True,
                )
                st.info(f" Detected seasonal period: **{period}** time steps")
            except Exception as e:
                st.error(f"Decomposition failed: {e}")
        else:
            st.info("Enable **Seasonal Decomposition** in the sidebar to view this analysis.")

    # Tab 4: Stationarity 
    with tab4:
        adf = stationarity_test(series)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown("#### Augmented Dickey-Fuller Test")
            if adf["test_statistic"] is not None:
                st.metric("Test Statistic", f"{adf['test_statistic']:.4f}")
                st.metric("P-Value", f"{adf['p_value']:.6f}")
                st.metric("Lags Used", adf["n_lags_used"])
                st.metric("Observations", adf["n_observations"])

                if adf["is_stationary"]:
                    st.success(" Series is **stationary** (p < 0.05)")
                else:
                    st.warning(" Series is **non-stationary** (p ≥ 0.05)")

                st.markdown("**Critical Values:**")
                for k, v in adf["critical_values"].items():
                    st.text(f" {k}: {v}")
            else:
                st.error(f"ADF test failed: {adf.get('error', 'unknown error')}")

        with col_b:
            st.plotly_chart(acf_plot(series), use_container_width=True)

    # Tab 5: Feature Importance 
    with tab5:
        st.markdown("#### Engineered Features")

        features_df = create_features(series)

        # Show feature table
        st.dataframe(features_df.head(50), use_container_width=True)

        # Correlation with target
        st.markdown("#### Feature Correlations with Target")
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        if "value" in numeric_cols:
            corr = features_df[numeric_cols].corr()["value"].drop("value").sort_values(
                ascending=False
            )

            import plotly.graph_objects as go

            fig = go.Figure()
            colors = ["#00ff88" if v > 0 else "#ef4444" for v in corr.values]
            fig.add_trace(
                go.Bar(
                    x=corr.values,
                    y=corr.index,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in corr.values],
                    textposition="auto",
                )
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                font=dict(
                    family="JetBrains Mono, monospace", color="#e0e0e0", size=11
                ),
                title="Feature Correlation with Target",
                height=500,
                margin=dict(l=150, r=30, t=50, b=30),
                xaxis=dict(title="Correlation", gridcolor="#1a1a2e"),
                yaxis=dict(gridcolor="#1a1a2e"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature toggles summary
        with st.expander(" Active Feature Engineering"):
            st.markdown(f"- **Lag Features:** {' Enabled' if use_lag_features else ' Disabled'}")
            st.markdown(
                f"- **Rolling Statistics:** {' Enabled' if use_rolling_stats else ' Disabled'}"
            )
            st.markdown(
                f"- **Decomposition:** {' Enabled' if use_decomposition else ' Disabled'}"
            )

else:
    # No data loaded 
    st.markdown(
        """
        <div style="text-align:center; padding:80px 20px;">
            <div style="font-size:4rem;"></div>
            <h2 style="color:#00ff88;">Welcome to TemporalNet</h2>
            <p style="color:#888; max-width:500px; margin:0 auto;">
                Select a demo dataset or upload your own CSV from the sidebar
                to begin forecasting. No API keys needed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer 
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#444;font-size:0.75rem;">'
    "TemporalNet &copy; 2026 Yogesh Kuchimanchi &bull; Built with Streamlit &bull; MIT License"
    "</p>",
    unsafe_allow_html=True,
)
