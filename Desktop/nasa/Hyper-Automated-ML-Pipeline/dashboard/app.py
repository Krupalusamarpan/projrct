import streamlit as st
import pandas as pd
import joblib
import os
import mlflow
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/current.csv"
REFERENCE_PATH = "data/reference.csv"

st.set_page_config(page_title="ML Monitoring Dashboard", layout="wide")

st.title("🚀 Hyper-Automated ML Dashboard")

# Model status
col1, col2 = st.columns([1, 3])
with col1:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        st.success("✅ Model Loaded")
        st.info(f"📏 Input features: {model.n_features_in_}")
    else:
        st.error("❌ Model not found - run `python src/train.py`")
        model = None

# MLflow metrics
st.subheader("📈 Model Performance History")
try:
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(order_by=["start_time DESC"], max_results=10)
    if runs and len(runs) > 0:
        df_runs = pd.DataFrame(runs)
        available_metrics = [col for col in df_runs.columns if 'metrics' in col]
        if 'metrics.rmse' in df_runs.columns and 'metrics.mae' in df_runs.columns:
            df_runs = df_runs[['start_time', 'metrics.rmse', 'metrics.mae']].copy()
        else:
            st.warning(f"Metrics columns missing. Available: {available_metrics}")
            df_runs = pd.DataFrame()
        df_runs['start_time'] = pd.to_datetime(df_runs['start_time'], unit='ms')
        df_runs = df_runs.set_index('start_time')
        
        # Performance trend bar
        fig_perf = px.bar(df_runs, x=df_runs.index.strftime('%m-%d %H:%M'), 
                         y=['rmse', 'mae'], barmode='group',
                         title="RMSE & MAE Over Time",
                         color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        st.plotly_chart(fig_perf, use_container_width=True)
        
        latest_rmse = df_runs['rmse'].iloc[0]
        latest_mae = df_runs['mae'].iloc[0]
        col1.metric("Latest RMSE", f"{latest_rmse:.3f}", delta=f"{df_runs['rmse'].diff().iloc[-1]:.3f}")
        col2.metric("Latest MAE", f"{latest_mae:.3f}", delta=f"{df_runs['mae'].diff().iloc[-1]:.3f}")
    else:
        st.warning("⚠️ No MLflow runs found")
except Exception as e:
    st.error(f"MLflow error: {e}")

# Data preview
if os.path.exists(DATA_PATH):
    current_data = pd.read_csv(DATA_PATH)
    st.subheader("📊 Current Data (Last 100 rows)")
    st.dataframe(current_data.tail(100), use_container_width=True)

# Batch predictions bar graph
if model and len(current_data) > 0:
    st.subheader("🔮 Batch RUL Predictions")
    try:
        # Sample for demo
        sample_data = current_data[['op1','op2','op3'] + [f'sensor_{i}' for i in range(1,22)]].tail(50).values
        predictions = model.predict(sample_data)
        
        fig_rul = go.Figure()
        fig_rul.add_trace(go.Bar(x=list(range(len(predictions))), y=predictions, name='Predicted RUL', 
                                marker_color='#4ECDC4'))
        fig_rul.update_layout(title="RUL Predictions (Last 50 Samples)", xaxis_title="Sample", yaxis_title="RUL")
        st.plotly_chart(fig_rul, use_container_width=True)
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

# Drift analysis
st.subheader("🔍 Data Drift Analysis")
try:
    if os.path.exists(REFERENCE_PATH):
        reference = pd.read_csv(REFERENCE_PATH).tail(100)
        current_sample = pd.read_csv(DATA_PATH).tail(100)
        
        # Feature drift bar (simplified % change)
        drift_scores = []
        for col in reference.columns[3:]:  # Skip ID, cycle, op settings
            ref_mean = reference[col].mean()
            curr_mean = current_sample[col].mean()
            drift = abs((curr_mean - ref_mean) / ref_mean) if ref_mean > 0 else 0
            drift_scores.append(drift)
        
        fig_drift = px.bar(x=reference.columns[3:], y=drift_scores, 
                          title="Feature Drift (% Change vs Reference)",
                          color=drift_scores, color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig_drift, use_container_width=True)
        
        high_drift = sum(1 for score in drift_scores if score > 0.1)
        st.metric("High Drift Features (>10%)", high_drift)
    else:
        st.warning("Reference data missing")
except Exception as e:
    st.error(f"Drift analysis failed: {e}")

# Single prediction
st.subheader("🎯 Single Prediction")
if model:
    st.info("Enter 25 sensor values (op1,op2,op3,sensor_1-21)")
    input_data = st.text_input("Comma-separated values:")
    if st.button("🔮 Predict RUL"):
        try:
            values = [float(x.strip()) for x in input_data.split(",") if x.strip()]
            if len(values) == 25:
                prediction = model.predict([values])[0]
                st.success(f"**Predicted RUL: {prediction:.2f}**")
            else:
                st.error("Exactly 25 values required")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.warning("Train model first")

# Automation status
st.subheader("🤖 Automation Status")
col1, col2, col3 = st.columns(3)
col1.success("✅ Drift Monitor")
col2.success("✅ Metric Tracking") 
col3.info("🔄 Run `python src/retrain_trigger.py`")

st.markdown("---")
st.info("💡 **Next:** `streamlit run dashboard/app.py`")
