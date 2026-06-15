import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load dataset
studata = pd.read_csv("data/student_data.csv")

# Features
X = studata[[
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "higher",
    "internet",
    "freetime",
    "goout",
    "health",
    "absences",
    "G1",
    "G2"
]]

# Target
y = studata["G3"]

# Encoding
X["higher"] = X["higher"].map({
    "yes": 1,
    "no": 0
})

X["internet"] = X["internet"].map({
    "yes": 1,
    "no": 0
})

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=100,random_state=42)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 2))

# Save model
with open("models/student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")