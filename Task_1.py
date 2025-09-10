import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# Select meaningful features (similar to "square_feet, bedrooms, bathrooms")
features = ['median_income', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'households', 'latitude', 'longitude']
X = df[features]
y = df['median_house_value']

# Handle missing values
X = X.fillna(X.mean())

# -------------------------------
# Create Pipeline (Scaling + Model)
# -------------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
])

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"Root Mean Squared Error: {rmse:,.2f}")
print(f"R² Score: {r2:.3f}")

# Cross-validation
cv_scores = cross_val_score(pipeline, X, y, scoring='r2', cv=5)
print(f"\n🔄 Cross-Validation R² Scores: {cv_scores}")
print(f"Average CV R²: {cv_scores.mean():.3f}")

# -------------------------------
# Feature Importance
# -------------------------------
# Extract coefficients from model
model = pipeline.named_steps['lin_reg']
scaler = pipeline.named_steps['scaler']

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\n📌 Feature Importance (Coefficients):")
print(coefficients)

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(14, 4))

# Actual vs Predicted
plt.subplot(1, 3, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted")

# Residuals
plt.subplot(1, 3, 2)
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, bins=30)
plt.title("Residuals Distribution")
plt.xlabel("Error (Actual - Predicted)")

# Predicted vs Actual Distribution
plt.subplot(1, 3, 3)
sns.kdeplot(y_test, label="Actual", fill=True)
sns.kdeplot(y_pred, label="Predicted", fill=True)
plt.title("Price Distribution")
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# Example Prediction
# -------------------------------
new_house = pd.DataFrame([[4.0, 20, 3000, 4, 2, 34.19, -118.35]],
                         columns=features)
predicted_price = pipeline.predict(new_house)[0]

print(f"\n🏠 Predicted price for the new house: ${predicted_price:,.2f}")
