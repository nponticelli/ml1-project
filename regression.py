import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path="phaseII_pca_reduced.csv"):  # <-- CRITICAL CHANGE: Load the final PCA file

    df = pd.read_csv(file_path)

    # 7 Numerical PCs + 3 Categorical features = 10 total features (if 7 PC)
    # If you reduced to 6 PCs, this list should be PC1 to PC6 + the 3 OHE features.
    # Let's assume 6 PCs for the reduced model:
    FEATURES_ALL = [
        "PC1", "PC2", "PC3", "PC4", "PC5", "PC6",
        "rusty_diff_0.0", "rusty_diff_1.0", "best_of_5"
    ]
    TARGET = "playerA_points_won_pct"  # Assuming this is your binary target (0/1)

    # ---------------------------------------
    # 1. Select features
    # ---------------------------------------
    # Features are already scaled and encoded from Phase I pipeline
    X = df[FEATURES_ALL].copy()
    Y = df[TARGET].copy()

    # ---------------------------------------
    # 2. Split the data
    # ---------------------------------------
    cutoff = int(len(X) * 0.8)
    x_train, x_test = X[:cutoff], X[cutoff:]  # Use final names directly
    y_train, y_test = Y[:cutoff], Y[cutoff:]

    print("Train/Test Split Complete:")
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

    full_feature_list = FEATURES_ALL

    print("\nFinal Processed Data Shapes:")
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
    print(f"Total features: {len(full_feature_list)}")

    return x_train.values, x_test.values, y_train, y_test, full_feature_list
    # Note: Returning .values for x_train/x_test to match np.hstack output format

# --- Helper Function for Stepwise Regression ---
def stepwise_regression(X_train, y_train, feature_names, verbose=False):
    """Performs backward stepwise selection based on Adjusted R-squared."""

    selected_features = feature_names[:]
    best_score = -np.inf
    best_model = None

    if verbose:
        print("\n--- Stepwise Regression (Backward Elimination by Adj. R²) ---")

    while len(selected_features) > 1:
        scores = {}
        models = {}

        # Try removing each feature one by one
        for feature_to_remove in selected_features:
            current_features = [f for f in selected_features if f != feature_to_remove]
            X_temp = X_train[:, [feature_names.index(f) for f in current_features]]

            # Add constant and fit OLS
            X_temp_const = sm.add_constant(X_temp, has_constant='add')
            model = sm.OLS(y_train, X_temp_const).fit()

            scores[feature_to_remove] = model.rsquared_adj
            models[feature_to_remove] = model

        # Find the feature whose removal yields the highest Adjusted R²
        best_removal = max(scores, key=scores.get)
        current_best_score = scores[best_removal]

        if current_best_score > best_score:
            best_score = current_best_score
            selected_features.remove(best_removal)
            best_model = models[best_removal]
            if verbose:
                print(
                    f"Removed: {best_removal}. New Adj. R²: {best_score:.4f}. Features remaining: {len(selected_features)}")
        else:
            # Stopping criterion: If removing any feature decreases Adj. R², stop.
            if verbose:
                print(f"Stopping. Max Adj. R² was achieved with current features.")
            break

    # Return the final set of features and the model before the final step failed to improve
    return selected_features, best_model

def run_regression(x_train, x_test, y_train, y_test, feature_names):
    # ---------------------------------------
    # 1. Full Model Training (using statsmodels)
    # ---------------------------------------
    # Statsmodels requires an explicit constant (intercept) term
    X_train_sm = sm.add_constant(x_train)
    X_test_sm = sm.add_constant(x_test)

    # Fit the Ordinary Least Squares (OLS) model
    model = sm.OLS(y_train, X_train_sm).fit()

    print("\n" + "=" * 50)
    print("      STATISTICAL SUMMARY (FULL MODEL)")
    print("=" * 50)
    print(model.summary())

    # ---------------------------------------
    # 2. Stepwise Regression (Backward)
    # ---------------------------------------
    # Perform backward elimination based on Adj. R²
    selected_features_indices = [i for i, f in enumerate(feature_names)]

    # Use the helper function (note: it uses all features initially)
    final_features_stepwise, best_model_stepwise = stepwise_regression(
        x_train, y_train, feature_names, verbose=True
    )

    # Get the X_test for the stepwise model
    X_test_stepwise = x_test[:, [feature_names.index(f) for f in final_features_stepwise]]
    X_test_stepwise_sm = sm.add_constant(X_test_stepwise)

    # Prediction using the stepwise model
    y_pred_stepwise = best_model_stepwise.predict(X_test_stepwise_sm)

    print("\n" + "=" * 50)
    print(f"   FINAL STEPWISE MODEL SUMMARY ({len(final_features_stepwise)} FEATURES)")
    print("=" * 50)
    print(best_model_stepwise.summary())

    # Use the stepwise model as the final recommended model
    final_model = best_model_stepwise
    y_pred = y_pred_stepwise

    # ---------------------------------------
    # 3. Required Metrics Table (Using Stepwise Model)
    # ---------------------------------------
    aic = final_model.aic
    bic = final_model.bic
    r_squared = final_model.rsquared
    adj_r_squared = final_model.rsquared_adj
    mse = mean_squared_error(y_test, y_pred)

    print("\n" + "=" * 50)
    print("      FINAL MODEL PERFORMANCE METRICS")
    print("=" * 50)
    metrics_data = {
        'Metric': ['R-squared', 'Adjusted R-squared', 'MSE (Test Set)', 'AIC', 'BIC'],
        'Value': [
            f"{r_squared:.4f}",
            f"{adj_r_squared:.4f}",
            f"{mse:.4f}",
            f"{aic:.2f}",
            f"{bic:.2f}"
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df.to_markdown(index=False))

    # ---------------------------------------
    # 4. Confidence Interval Analysis
    # ---------------------------------------
    print("\n" + "=" * 50)
    print("      95% CONFIDENCE INTERVALS (Stepwise Model)")
    print("=" * 50)
    # CI provided by statsmodels summary
    ci_table = final_model.conf_int(alpha=0.05)
    ci_table.columns = ['Lower CI (2.5%)', 'Upper CI (97.5%)']
    print(ci_table.to_markdown())

    # ---------------------------------------
    # 5. T-test and F-test Analysis
    # ---------------------------------------
    # T-test: Provided in the model summary (P>|t| column).
    # F-test: Provided in the model summary (Prob(F-statistic)).
    print(
        "\n*T-test Analysis:* Refer to the 'P>|t|' column in the summary table. A P-value < 0.05 indicates the coefficient is statistically significant.")
    print(
        "\n*F-test Analysis:* Refer to the 'Prob (F-statistic)' in the summary table. A P-value < 0.05 indicates the overall model is statistically significant.")

    # ---------------------------------------
    # 6. Plotting Results
    # ---------------------------------------
    # Limit the number of test points for a readable plot
    n_plot = 500

    plt.figure(figsize=(10, 6))

    plt.plot(range(n_plot), y_test[:n_plot], label='Actual Test Value (y_test)', color='blue', marker='.',
             linestyle='None')

    # Fix 2: REMOVE .values from y_pred (it is a NumPy array from the predict() method)
    plt.plot(range(n_plot), y_pred[:n_plot], label='Predicted Value (y_pred)', color='red', marker='x',
             linestyle='None')

    # Fix 3: Remove .values from y_train (it is a Pandas Series, but can be treated like a numpy array)
    plt.plot(range(n_plot, n_plot + n_plot), y_train[-n_plot:], label='Training Data (y_train)', color='gray',
             alpha=0.5)

    plt.title("Actual vs. Predicted Percentage of Points Won (Test Set)")
    plt.xlabel("Data Point Index (Chronological)")
    plt.ylabel("Player A Points Won Percentage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# You would integrate this function into your main execution block:
# if __name__ == '__main__':
#     xTrain, xTest, yTrain, yTest, feature_names = load_and_prepare_data()
#     run_regression(xTrain, xTest, yTrain, yTest, feature_names)

if __name__ == '__main__':
    # Load and preprocess
    xTrain, xTest, yTrain, yTest, feature_names = load_and_prepare_data()

    run_regression(xTrain, xTest, yTrain, yTest, feature_names)
