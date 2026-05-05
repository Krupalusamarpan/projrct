from sklearn.metrics import mean_absolute_error
import joblib
from config import *

def evaluate(X_test,y_test):
    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test,preds)
    print("MAE:",mae)

