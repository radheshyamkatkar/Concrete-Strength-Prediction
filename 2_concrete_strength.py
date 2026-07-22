"""
PROJECT 2: CONCRETE STRENGTH PREDICTION
Goal: Predict the compressive strength of concrete (in MPa) based on the
proportions of its ingredients, using Linear Regression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Creation of synthetic concrete mix dataset
#Real concrete strength depends on: cement, water, aggregate, age (days), etc.
#We simulate realistic ranges based on typical civil engineering datasets

np.random.seed(7)
n_samples = 300

cement = np.random.uniform(100, 500, n_samples)        # kg per m^3
water = np.random.uniform(120, 250, n_samples)          # kg per m^3
coarse_aggregate = np.random.uniform(800, 1100, n_samples)  # kg per m^3
fine_aggregate = np.random.uniform(600, 950, n_samples)     # kg per m^3
age_days = np.random.choice([1, 3, 7, 14, 28, 56, 90], n_samples)  # curing age

#Water-cement ratio strongly affects strength (lower ratio = stronger concrete)
water_cement_ratio = water / cement

#Simulate compressive strength (target) using a realistic formula + noise
strength = (0.25 * cement - 40 * water_cement_ratio + 0.01 * coarse_aggregate + 0.015 * fine_aggregate + 0.4 * np.sqrt(age_days) * 5 + np.random.normal(0, 4, n_samples))
strength = np.clip(strength, 5, None)  # strength can't be negative

df = pd.DataFrame({"cement": cement,"water": water,"coarse_aggregate": coarse_aggregate,"fine_aggregate": fine_aggregate,"age_days": age_days,"strength_mpa": strength})

print("First 5 rows of dataset:")
print(df.head(), "\n")

#EDA
print("Summary statistics:")
print(df.describe(), "\n")
print("Correlation with strength:")
print(df.corr()["strength_mpa"].sort_values(ascending=False), "\n")

#Define features (X) and target (y)
X = df[["cement", "water", "coarse_aggregate", "fine_aggregate", "age_days"]]
y = df["strength_mpa"]

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

#Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Training the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Model coefficients (on scaled features):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.3f}")
print(f"  Intercept: {model.intercept_:.3f}\n")

#Making predictions
y_pred = model.predict(X_test_scaled)

#Evaluatation model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation on Test Set:")
print(f"  MAE: {mae:.2f} MPa")
print(f"  RMSE: {rmse:.2f} MPa")
print(f"  R-squared: {r2:.3f}\n")

#Visualization actual vs predicted strength
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="seagreen", edgecolor="k")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Perfect Prediction")
plt.xlabel("Actual Strength (MPa)")
plt.ylabel("Predicted Strength (MPa)")
plt.title("Concrete Strength: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()
