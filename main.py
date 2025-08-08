# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv("dataset/house_prices.csv")

# 2. Clean column names (remove spaces, lowercase)
df.columns = df.columns.str.strip().str.lower()

# 3. Explore basic info
print("✅ First 5 rows:\n", df.head())
print("\nℹ️ Dataset Summary:\n", df.describe())
print("\n❓ Missing values:\n", df.isnull().sum())
print("📌 Columns in DataFrame:", df.columns.tolist())

# 4. Drop missing values (if any)
df.dropna(inplace=True)

# 5. Feature selection (make sure names match cleaned columns)
X = df[["area"]]  # Independent variable
y = df["price"]   # Dependent variable

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Predict
y_pred = model.predict(X_test)

# 9. Evaluation
print("\n📊 Model Evaluation:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# 10. Coefficients
print("\n📈 Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")

# 11. Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (₹)")
plt.title("Linear Regression - Area vs Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("images/regression_plot.png")  # Make sure 'images/' folder exists
plt.show()
