from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# === Load Target Variable ===
data = pd.read_csv("X:\\Economics\\IIT Kanpur\\Sem 2\\Computational Methods\\Project\\US_Presidential_Election_Data.csv")


vote_of_incumbent_next_election = np.array([
    0.35868935, 0.46875558, 0.40252173, 0.5, 0.33794732, 0.49760394,
    0.38941565, 0.33014223, 0.48072585, 0.43494272, 0.3, 0.39985607,
    0.39265723, 0.41235037, 0.36949773, 0.41536832, 0.39096109, 0.37976175,
    0.38935569
])
target  = vote_of_incumbent_next_election.copy()
approval_rating = np.array([
    0.36315789, 0.49390582, 0.47174515, 0.5, 0.41689751, 0.42188366,
    0.3, 0.36371191, 0.40526316, 0.46066482, 0.4501385, 0.38642659,
    0.44958449, 0.45623269, 0.31828255, 0.37700831, 0.40637119, 0.33933518,
    0.34542936
])
personal_income = np.array([
    0.3, 0.30049297, 0.30144253, 0.30259064, 0.30510462, 0.30837028,
    0.31390816, 0.32304178, 0.33383741, 0.34547807, 0.35720585, 0.36920637,
    0.38452039, 0.3962888, 0.41754508, 0.42326918, 0.44022674, 0.45915483,
    0.5
])
unemployment = np.array([
    0.3, 0.35555556, 0.44444444, 0.38518519, 0.32592593, 0.39259259,
    0.48888889, 0.46296296, 0.47777778, 0.4037037, 0.47777778, 0.4,
    0.34814815, 0.4037037, 0.41481481, 0.5, 0.38148148, 0.5, 0.34814815
])
consecutive_terms=np.array([
    0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.3, 0.4,
    0.5, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.3
])
# === Combine All Predictors into DataFrame ===
X = pd.DataFrame({
    "personal_income":personal_income,
    "unemployment_rate": unemployment,
    "gallup_june":approval_rating,
    "consecutive_terms":consecutive_terms
})

y = vote_of_incumbent_next_election

model = LinearRegression()
sfs = SFS(model,
          k_features='best',
          forward=True,
          floating=False,
          scoring='r2',
          cv=0)

sfs.fit(X, y)
print("Selected features:", list(sfs.k_feature_names_))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load the dataset
data = pd.read_csv("X:\Economics\IIT Kanpur\Sem 2\Computational Methods\Project\\US_Presidential_Election_Data_June.csv")
final_test_data = data.tail(4)
data = data.iloc[:-4]
def preprocess_data(data):
    X = pd.DataFrame({
        "personal_income": personal_income[:16],
        "unemployment_rate": unemployment[:16],
        "gallup_june": approval_rating[:16],
        "consecutive_terms":consecutive_terms[:16]
    })                              # Features
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
X_final_test = X[16:19]
y_final_test = final_test_data["votes/total in next election incumbent party"][0:3].values
# Train and evaluate models
train_svr(X_train, X_test, y_train, y_test, C=0.25, alpha=2)