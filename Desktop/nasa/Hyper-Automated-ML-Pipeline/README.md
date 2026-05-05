# Hyper-Automated ML Pipeline for NASA Turbofan RUL Prediction

## Overview
End-to-end automated ML pipeline for NASA C-MAPSS Turbofan engine Remaining Useful Life (RUL) prediction.

**Key Features:**
- Data preprocessing & feature engineering for time-series sensor data
- Model training (RandomForest + LSTM) with MLflow tracking
- Production monitoring: Data drift (Evidently), performance metrics
- Auto-retraining triggers
- Streamlit dashboard for predictions, metrics, drift visualization
- FastAPI serving
- Dockerized deployment + Airflow orchestration + CI/CD workflows

**Performance:** RMSE/MAE ≈ 0.37 on FD001 dataset

## Quick Demo
```bash
cd Hyper-Automated-ML-Pipeline
pip install -r requirements.txt
```

**Services:**
- MLflow UI: `mlflow ui --port 5000`
- Dashboard: `streamlit run dashboard/app.py --server.port 8501`
- API: `uvicorn api.main:app --reload --port 8000`

**URLs:**
- [Dashboard](http://localhost:8501)
- [API Docs](http://localhost:8000/docs)
- [MLflow](http://localhost:5000)

**Test API:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [0.6,0.4,0.3,42.5,0.1,0.2,0.3,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0]}'
```

## Architecture
```
NASA Data (train_FD001.txt) 
  ↓
Preprocessing → Feature Engineering (RUL calc) → Train (RF/LSTM, MLflow)
  ↓
Evidently Drift Detect ← Current Data → Retraining Trigger (if drift/poor perf)
  ↓
Streamlit Dashboard + FastAPI Predict Endpoint
  ↓
Docker Compose + GitHub Actions + Airflow Pipeline
```

## Dataset
- **NASA CMAPSS FD001**: 20,631 cycles across 100 engines
- Columns: unit_nr, cycle, op1-3, sensor1-21 (26 cols)
- **Target**: RUL (clipped at 125 cycles)

## Installation & Usage
1. `git clone https://github.com/Krupalusamarpan/projrct.git`
2. `pip install -r requirements.txt`
3. `python src/main.py` (train & log to MLflow)
4. Start services (see Quick Demo)
5. See TODO_RUN.md for full steps

## Components
| Directory | Purpose |
|-----------|---------|
| `src/` | Core: data_loader, preprocessing, model training, retrain |
| `monitoring/` | Drift detection & triggers (Evidently) |
| `dashboard/` | Streamlit app (preds, charts, drift) |
| `api/` | FastAPI prediction service |
| `models/` | Saved model.pkl, scaler.pkl |
| `data/` | train_FD001.txt, reference.csv, current.csv |

## Monitoring & Automation
- **Drift**: `monitoring/drift_trigger.py` compares reference vs current data
- **Retraining**: `src/retrain_trigger.py` checks MAE degradation
- Auto-triggers full retrain pipeline

## Deployment
- `docker-compose up`
- `.github/workflows/deploy.yml` for CI/CD
- Airflow: `pipeline/airflow_pipeline.py`

## Next Steps
- Unit tests (TODO_IMPROVEMENTS.md)
- LSTM tuning
- More CMAPSS subsets (FD002-004)

**Generated with BLACKBOXAI assistance** 🎯

