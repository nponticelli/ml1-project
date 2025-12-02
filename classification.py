import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_Classifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def load_and_prepare_data(file_path="phaseII.csv"):
    # Load dataset
    df = pd.read_csv(file_path)
    FEATURES_NUM = [
        "combined_elo_diff",
        "last_minutes_diff",
        "fatigue_10d_diff",
        "year_fatigue_diff",
        "raw_age_diff",
        "prime_age_diff",
        "raw_height_diff",
        "prime_height_diff",
        "service_advantage_diff",
    ]
    FEATURES_CAT = ["rusty_diff", "best_of"]
    TARGET = "log_target"

    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    Y = df[TARGET].copy()
    # Split the data
    cutoff = int(len(X) * 0.8)
    x_train_raw, x_test_raw = X[:cutoff], X[cutoff:]
    y_train, y_test = Y[:cutoff], Y[cutoff:]

    print("Train/Test Split Complete:")
    print(f"x_train_raw shape: {x_train_raw.shape}, x_test_raw shape: {x_test_raw.shape}")

    scaler = StandardScaler()

    # FIT the scaler ONLY on the training data
    x_train_num_scaled = scaler.fit_transform(x_train_raw[FEATURES_NUM])
    # TRANSFORM the test data using the TR AINING fit
    x_test_num_scaled = scaler.transform(x_test_raw[FEATURES_NUM])

    # Encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop='first')

    # FIT the encoder ONLY on the training data
    x_train_cat = encoder.fit_transform(x_train_raw[FEATURES_CAT])
    # TRANSFORM the test data using the TRAINING fit
    x_test_cat = encoder.transform(x_test_raw[FEATURES_CAT])

    # Get feature names for final list
    cat_feature_names = encoder.get_feature_names_out(FEATURES_CAT)

    # ---------------------------------------
    # 4. Combine into final feature matrices
    # ---------------------------------------
    # Stack the scaled numerical features and the encoded categorical features
    x_train = np.hstack([x_train_num_scaled, x_train_cat])
    x_test = np.hstack([x_test_num_scaled, x_test_cat])

    # Create the full list of feature names
    full_feature_list = FEATURES_NUM + list(cat_feature_names)

    print("\nFinal Processed Data Shapes:")
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
    print(f"Total features: {len(full_feature_list)}")

    return x_train, x_test, y_train, y_test, full_feature_list

def run_lda(x_train, x_test, y_train, y_test, feature_names):
    # Assuming x_train, x_test, y_train, y_test are defined from your previous steps

    # ---------------------------------------
    # 1. Define the LDA Classifier
    # ---------------------------------------
    lda_clf = LDA_Classifier()

    # ---------------------------------------
    # 2. Define the Hyperparameter Grid
    # ---------------------------------------
    # LDA generally has very few parameters to tune.
    # - solver: 'svd' (default, no shrinkage), 'lsqr', or 'eigen'.
    # - shrinkage: used with 'lsqr' or 'eigen' solvers.
    param_grid = [
        {
            'solver': ['svd'],  # SVD does not support shrinkage
            'shrinkage': [None]
        },
        {
            'solver': ['lsqr'],  # LSQR supports shrinkage
            'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]  # Test no shrinkage, auto, and various fixed values
        }
    ]

    # ---------------------------------------
    # 3. Perform Grid Search
    # ---------------------------------------
    print("Starting LDA Grid Search...")
    start_time = time.time()

    # Use cross-validation (cv=5) and n_jobs=-1 to speed up search
    grid_search_lda = GridSearchCV(
        estimator=lda_clf,
        param_grid=param_grid,
        scoring='accuracy',  # Maximize classification accuracy
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid_search_lda.fit(x_train, y_train)

    end_time = time.time()
    print(f"LDA Grid Search finished in {end_time - start_time:.2f} seconds.")

    # ---------------------------------------
    # 4. Evaluate Best Model
    # ---------------------------------------
    best_lda = grid_search_lda.best_estimator_

    # Predict on the test set
    y_pred_lda = best_lda.predict(x_test)

    print("\n--- LDA Classification Results ---")
    print(f"Best LDA Parameters: {grid_search_lda.best_params_}")
    print(f"Best CV Score (Accuracy): {grid_search_lda.best_score_:.4f}")
    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred_lda):.4f}")
    print("\nClassification Report (Test Set):\n", classification_report(y_test, y_pred_lda))

def run_logistic_regression(x_train, x_test, y_train, y_test, feature_names):
    # ---------------------------------------
    # 1. Define the Logistic Regression Classifier
    # ---------------------------------------
    # Set max_iter to a higher value to ensure convergence, especially with LBFGS/Sag solvers
    # random_state is set for reproducibility
    log_reg_clf = LogisticRegression(solver='liblinear', random_state=42, max_iter=5000)

    # ---------------------------------------
    # 2. Define the Hyperparameter Grid
    # ---------------------------------------
    # C: Inverse of regularization strength (smaller C means stronger regularization)
    # penalty: 'l1' or 'l2' (L1 for feature selection, L2 for general stability)

    param_grid = [
        {
            'penalty': ['l1'],
            # C values are tested on a logarithmic scale
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear']  # 'liblinear' supports both l1 and l2
        },
        {
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear']  # 'lbfgs' is generally good for l2
        }
    ]

    # ---------------------------------------
    # 3. Perform Grid Search (Cross-Validation)
    # ---------------------------------------
    print("\nStarting Logistic Regression Grid Search...")
    start_time = time.time()

    # Use 5-fold cross-validation (cv=5)
    grid_search_log_reg = GridSearchCV(
        estimator=log_reg_clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid_search_log_reg.fit(x_train, y_train)

    end_time = time.time()
    print(f"Logistic Regression Grid Search finished in {end_time - start_time:.2f} seconds.")

    # ---------------------------------------
    # 4. Evaluate Best Model
    # ---------------------------------------
    best_log_reg = grid_search_log_reg.best_estimator_

    # Predict on the test set
    y_pred_log_reg = best_log_reg.predict(x_test)

    print("\n--- Logistic Regression Classification Results ---")
    print(f"Best LR Parameters: {grid_search_log_reg.best_params_}")
    print(f"Best CV Score (Accuracy): {grid_search_log_reg.best_score_:.4f}")
    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
    print("\nClassification Report (Test Set):\n", classification_report(y_test, y_pred_log_reg))

def run_random_forest(x_train, x_test, y_train, y_test, feature_names):
    # ---------------------------------------
    # 1. Define the Random Forest Classifier
    # ---------------------------------------
    # Default is n_estimators=100. Setting class_weight='balanced' addresses potential imbalance.
    rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

    # ---------------------------------------
    # 2. Define the Hyperparameter Grid
    # ---------------------------------------
    # This grid focuses on parameters that control model complexity and diversity.
    param_grid = {
        # n_estimators: Number of trees in the forest.
        'n_estimators': [100, 200, 300],
        # max_depth: Maximum depth of the tree (controls individual tree complexity/pre-pruning).
        'max_depth': [5, 10, None], # None means nodes are expanded until all leaves are pure
        # min_samples_split: Minimum number of samples required to split an internal node.
        'min_samples_split': [5, 10],
        # max_features: Number of features to consider when looking for the best split (key to Bagging).
        'max_features': ['sqrt', 0.5]
    }

    # ---------------------------------------
    # 3. Perform Grid Search (Cross-Validation)
    # ---------------------------------------
    print("\nStarting Random Forest Grid Search (Bagging)...")
    start_time = time.time()

    # Use 5-fold cross-validation (cv=5) and n_jobs=-1 for parallel processing
    grid_search_rf = GridSearchCV(
        estimator=rf_clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid_search_rf.fit(x_train, y_train)

    end_time = time.time()
    print(f"Random Forest Grid Search finished in {end_time - start_time:.2f} seconds.")

    # ---------------------------------------
    # 4. Evaluate Best Model
    # ---------------------------------------
    best_rf = grid_search_rf.best_estimator_

    # Predict on the test set
    y_pred_rf = best_rf.predict(x_test)

    print("\n--- Random Forest Classification Results (Bagging) ---")
    print(f"Best RF Parameters: {grid_search_rf.best_params_}")
    print(f"Best CV Score (Accuracy): {grid_search_rf.best_score_:.4f}")
    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print("\nClassification Report (Test Set):\n", classification_report(y_test, y_pred_rf))

    return best_rf

if __name__ == '__main__':
    # Load and preprocess
    xTrain, xTest, yTrain, yTest, feature_names = load_and_prepare_data()

    run_lda(xTrain, xTest, yTrain, yTest, feature_names)

    run_logistic_regression(xTrain, xTest, yTrain, yTest, feature_names)

    run_random_forest(xTrain, xTest, yTrain, yTest, feature_names)
