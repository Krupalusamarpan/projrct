import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_loader import load_dataset
from feature_engineering import create_rul
from preprocessing import scale_data
from sklearn.preprocessing import StandardScaler
from config import *

mlflow.set_experiment("RUL_prediction")

def train_model():
    try:
        df = load_dataset(DATA_PATH)
        df = create_rul(df)
        X = df.drop(["engine_id","cycle","RUL"], axis=1)
        y = df["RUL"]
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        import optuna
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 8, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
            }
            model = RandomForestRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return mean_absolute_error(y_val, preds)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        print("Best MAE:", study.best_value)
        print("Best params:", best_params)
        
        model = RandomForestRegressor(**best_params, random_state=42)
        with mlflow.start_run():
            model.fit(X_train,y_train)
            preds_val = model.predict(X_val)
            preds_test = model.predict(X_test)
            rmse_val = mean_squared_error(y_val,preds_val) ** 0.5
            mae_val = mean_absolute_error(y_val,preds_val)
            rmse_test = mean_squared_error(y_test,preds_test) ** 0.5
            mae_test = mean_absolute_error(y_test,preds_test)
            mlflow.log_param("n_estimators",200)
            mlflow.log_param("max_depth",12)
            mlflow.log_metric("rmse_val",rmse_val)
            mlflow.log_metric("mae_val",mae_val)
            mlflow.log_metric("rmse_test",rmse_test)
            mlflow.log_metric("mae_test",mae_test)
            mlflow.sklearn.log_model(model,"model")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump(model,MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
            print("Model saved successfully to", MODEL_PATH)
            print("Val RMSE/MAE:", rmse_val, mae_val)
            print("Test RMSE/MAE:", rmse_test, mae_test)
    except Exception as e:
        print("Training failed:", str(e))

if __name__ == "__main__":
    train_model()

