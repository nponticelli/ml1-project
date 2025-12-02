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

def run_regression(xTrain, xTest, yTrain, yTest, feature_names):
    return

if __name__ == '__main__':
    # Load and preprocess
    xTrain, xTest, yTrain, yTest, feature_names = load_and_prepare_data()

    run_regression(xTrain, xTest, yTrain, yTest, feature_names)
