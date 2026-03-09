import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load the dataset
data = pd.read_csv("X:\Economics\IIT Kanpur\Sem 2\Computational Methods\Project\\US_Presidential_Election_Data_June.csv")
final_test_data = data[13:16]
data = data.iloc[0:13]
def preprocess_data(data):
    X = data[["Gallup june"]][:16]  # Features
    y = data["votes/total in next election incumbent party"][:16]  # Target (continuous variable)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test,scaler

def train_svr(X_train, X_test, y_train, y_test, C, alpha, a=None):
    svr_model = SVR(C=C, kernel='rbf')  # SVR does not use alpha and a directly
    svr_model.fit(X_train, y_train)
    y_pred = svr_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    y_final_pred = svr_model.predict(X_final_test)
    final_rmse = np.sqrt(mean_squared_error(y_final_test, y_final_pred))
    final_mape = mean_absolute_percentage_error(y_final_test, y_final_pred)
    print(f"SVR Model (C={C}, alpha={alpha}, a={a})")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}\n")
    print(f"Final Test Set - RMSE: {final_rmse:.4f}, MAPE: {final_mape:.4f}\n")

# Process dataset once
X_train, X_test, y_train, y_test,scaler = preprocess_data(data)
X_final_test = scaler.transform(final_test_data[["Gallup june"]][0:3])
y_final_test = final_test_data["votes/total in next election incumbent party"][0:3].values
# Train and evaluate models
train_svr(X_train, X_test, y_train, y_test, C=0.25, alpha=5, a=0.07)
