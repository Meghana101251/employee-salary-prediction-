# Employee Salary Prediction using Linear Regression

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (you can replace with your own CSV path)
data = pd.read_csv("employee_data.csv")

# Display first few rows
print("\nSample data:")
print(data.head())

# Check for null values
print("\nMissing values:")
print(data.isnull().sum())

# Drop rows with missing values (optional, based on data quality)
data = data.dropna()

# Encode categorical variables (e.g., education, job_title, industry)
data_encoded = pd.get_dummies(data, drop_first=True)

# Features and target
X = data_encoded.drop("salary", axis=1)
y = data_encoded["salary"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Plot actual vs predicted salaries
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.grid(True)
plt.show()

# Save model using joblib (optional for deployment)
import joblib
joblib.dump(model, "salary_model.pkl")

print("\nModel saved as 'salary_model.pkl'")
