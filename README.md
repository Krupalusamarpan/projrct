
Hyper-Automation of ML Pipelines

**Project Idea:** 

- The project aims to build a **fully automated ML pipeline** that can train, deploy, and monitor models with minimal human intervention.
- The system continuously **monitors incoming data and model performance** to detect data drift or performance degradation.
- When drift is detected, the pipeline **automatically retrains the model** using updated data.
- The newly trained model is **automatically evaluated and deployed** using a CI/CD  or docker pipeline.
- A **self-healing mechanism** enables automatic rollback to a stable model if performance drops.
- This approach ensures **adaptive, reliable, and production-ready ML systems**.

![Screenshot 2026-03-31 at 12.35.33.png](attachment:e1a341f4-8e78-4399-8556-8af101d7ae3d:Screenshot_2026-03-31_at_12.35.33.png)

PROBLEM STATEMENT

Design and implement an automated machine learning pipeline that can automatically preprocess data, select appropriate models, perform hyperparameter tuning, and evaluate performance with minimal manual intervention

The pipeline is evaluated on NASA’s CMAPSS dataset

# COMPLETE WORKFLOW

```
NASA CMAPSS Dataset
        ↓
Data Preprocessing
        ↓
Model Training
        ↓
MLflow Experiment Tracking
        ↓
Saved Trained Model
        ↓
FastAPI Server (API)
        ↓
User/Application sends sensor data
        ↓
Real-time RUL Prediction
```

DATASET

NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) Dataset 

The dataset used is the **CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** turbofan engine dataset from the **NASA Open Data Portal**.

https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data/resource/5224bcd1-ad61-490b-93b9-2817288accb8

### Dataset Type

- **Multivariate Time-Series Dataset**
- **Regression Problem**
- **Predictive Maintenance Dataset**

---

### What the Dataset Contains

Each record represents the condition of an aircraft engine during one operational cycle.


## Setup and Imports

**Purpose:** Initialize all required libraries, configurations, and system parameters for building the automated ML pipeline.

**Includes:**

- Import libraries:
    - Pandas, NumPy → data processing
    - Scikit-learn → machine learning models
    - MLflow → experiment tracking
    - FastAPI → deployment API
    - Uvicorn → API server
    - Joblib/Pickle → model saving & loading
- Define constants:
    - DATA_PATH = dataset location
    - MODEL_PATH = saved model directory
    - RANDOM_STATE = 42
    - TEST_SIZE = 0.2

**Outcome:**

Environment prepared for automated ML workflow execution.

---

## Dataset Loading and Understanding

**Purpose:** Load and inspect the NASA CMAPSS turbofan engine dataset.

**Includes:**

- Load dataset from NASA Open Data Portal provided by **NASA**
- Dataset structure:
    - unit_number (engine ID)
    - cycle (time step)
    - operational settings (3 columns)
    - sensor_1 to sensor_21
- Understand degradation behavior across engine life cycles.

**Outcome:**

Raw time-series sensor dataset ready for processing.

---

## Automatic Data Validation

**Purpose:** Ensure data quality before training begins.

**Includes:**

- Check missing values
- Verify numeric formats
- Detect abnormal sensor ranges
- Remove corrupted records
- Validate engine cycle sequences

**Outcome:**

Clean and reliable dataset enters the ML pipeline.

---

## Data Preprocessing Module

**Purpose:** Convert raw industrial sensor data into model-ready format.

**Includes:**

- Sorting data by engine and cycle
- Removing constant or irrelevant sensors
- Scaling/normalizing sensor values
- Handling noise and inconsistencies

**Outcome:**

Structured and normalized time-series data.

---

## Feature Engineering & RUL Creation

**Purpose:** Create meaningful learning features and prediction target.

**Includes:**

- Calculate **Remaining Useful Life (RUL)**:

```
RUL = max_cycle − current_cycle
```

- Generate degradation indicators
- Select informative sensor features
- Prepare feature matrix (X) and target (y)

**Outcome:**

Dataset transformed into supervised learning format.

---

## Automatic Model Training

**Purpose:** Train machine learning model to learn engine degradation patterns.

**Includes:**

- Split dataset:
    - Training set
    - Testing set
- Train regression model (e.g., Random Forest / Gradient Boosting)
- Learn relationship:

```
Sensor Behaviour → Engine Health → Remaining Useful Life
```

**Outcome:**

Initial predictive model created automatically.

---

## Hyperparameter Optimization

**Purpose:** Improve model performance automatically.

**Includes:**

- Tune parameters such as:
    - number of estimators
    - tree depth
    - learning parameters
- Use automated search methods (GridSearchCV / RandomSearch)

**Outcome:**

Optimized model configuration with improved accuracy.

---

## Performance Evaluation & Experiment Tracking

**Purpose:** Evaluate model performance and record experiments systematically.

**Includes:**

- Calculate evaluation metrics:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
- Log experiments using MLflow:
    - parameters
    - metrics
    - training runs

**Outcome:**

Reproducible experiment history and performance comparison dashboard.

---

## Model Versioning and Best Model Selection

**Purpose:** Automatically identify and store the best performing model.

**Includes:**

- Compare multiple experiment runs
- Select best model based on evaluation metric
- Save model artifact in Model Registry

**Outcome:**

Production-ready model selected automatically.

---

## Deployment using FastAPI

**Purpose:** Make the trained model usable in real-world applications.

**Includes:**

- Create FastAPI server
- Load saved trained model
- Create prediction endpoint:

```
/predict
```

- Accept sensor values as input JSON.

Example Input:

```json
{
  "sensor_1": 520,
  "sensor_2": 642,
  "sensor_3": 1589
}
```

**Outcome:**

Model accessible via API instead of notebook execution.

---

## Real-Time Prediction System

**Purpose:** Generate live Remaining Useful Life predictions.

**Includes:**

- External system sends sensor data
- API processes input
- Model predicts engine RUL
- Response returned instantly

Example Output:

```json
{
  "Predicted_RUL": 85
}
```

**Outcome:**

Real-time predictive maintenance capability.

---

## Hyper-Automation Capability (Core Contribution)

**Purpose:** Automate the full ML lifecycle.

**Includes:**

- Automated training pipeline
- Experiment tracking
- Model version control
- Monitoring readiness
- Continuous prediction service

**Outcome:**

Self-maintaining ML system similar to industry MLOps solutions.Workflow Summary Overview


<img width="373" height="461" alt="Screenshot 2026-05-05 at 15 24 09" src="https://github.com/user-attachments/assets/764d2028-0b29-4754-8ff4-dad11ffc9049" />

