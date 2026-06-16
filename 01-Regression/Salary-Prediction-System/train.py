import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("data/wage_predict.csv")

# Features and Target
X = df.drop("monthly_salary", axis=1)
y = np.log1p(df["monthly_salary"])

# One Hot Encoding
X = pd.get_dummies(
    X,
    columns=[
        "industry",
        "occupation",
        "highest_qual",
        "area_of_study"
    ],
    drop_first=True
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Training
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))

# Save Model
with open("models/salary_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save Columns
with open("models/model_columns.pkl", "wb") as file:
    pickle.dump(X.columns.tolist(), file)

print("Model Saved Successfully")

with open("models/model_columns.pkl", "rb") as file:
    model_columns = pickle.load(file)

print(len(model_columns))
print(model_columns[:20])