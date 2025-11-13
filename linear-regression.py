import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 1️⃣ Load your engineered data ---
df = pd.read_csv("fe_matches.csv")

# --- 2️⃣ Separate features (X) and target (y) ---
X = df.drop(columns=["target"])
y = df["target"]

# --- 3️⃣ Identify categorical vs numeric columns ---
cat_cols = ["player1_hand", "player2_hand", "match_type"]
num_cols = [c for c in X.columns if c not in cat_cols]

# --- 4️⃣ Preprocess data ---
# One-hot encode categorical columns, keep numeric as is
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ]
)

# --- 5️⃣ Define Linear Regression model ---
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# --- 6️⃣ Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 7️⃣ Train model ---
model.fit(X_train, y_train)

# --- 8️⃣ Predict and evaluate ---
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression trained!")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")

# Optional: show a few predictions
comparison = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
})
print("\nSample predictions:")
print(comparison)