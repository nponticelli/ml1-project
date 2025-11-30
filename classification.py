import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def load_and_prepare_data(file_path="fe_simplified_modular.csv"):
    # Load dataset
    df = pd.read_csv(file_path)
    df = df.sort_values("pseudo_date").reset_index(drop=True)

    # Warm-up drop
    df = df.iloc[3000:].reset_index(drop=True)
    print("After warm-up drop:", df.shape)

    # Define features and target
    FEATURES_NUM = [
        "lower_combined_elo", "combined_elo_diff", "fatigue_diff",
        "age_diff_z", "lower_service_advantage", "service_advantage_diff",
        "first_serve_pct_diff", "height_diff"
    ]
    FEATURES_CAT = ["age_advantage", "is_grand_slam", "surface"]
    TARGET = "log_target"

    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[TARGET].copy().values.ravel()



    # Handle outliers with IQR Winsorization
    for col in FEATURES_NUM:
        Q1, Q3 = X[col].quantile([0.25,0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        X[col] = X[col].clip(lower, upper)

    # Encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X[FEATURES_CAT])
    cat_feature_names = encoder.get_feature_names_out(FEATURES_CAT)

    # Scale numerical features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X[FEATURES_NUM])

    # Combine numerical and categorical
    X_full = np.hstack([X_num_scaled, X_cat])
    feature_names = FEATURES_NUM + list(cat_feature_names)

    return X_full, y, feature_names

def train_test_split_chronological(X, y, train_ratio=0.8):
    # 1. Chronological train/test split
    cutoff = int(len(X) * train_ratio)
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    print("Before SMOTE -> Train class distribution:\n",
          pd.Series(y_train).value_counts(normalize=True))

    # 2. Apply SMOTE to the training set only
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    (unique, counts) = np.unique(y_train_bal, return_counts=True)
    print("After SMOTE -> Train class distribution:\n", dict(zip(unique, counts / counts.sum())))

    print("Train shape:", X_train_bal.shape, "Test shape:", X_test.shape)
    return X_train_bal, y_train_bal, X_test, y_test

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    return y_pred
# -----------------------------
# Helper function: plot confusion matrix
# -----------------------------
def plot_cm(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

# -----------------------------
# 1. Linear Discriminant Analysis
# -----------------------------
def run_lda(X_train, X_test, y_train, y_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    print("=== LDA ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    if hasattr(lda, "coef_"):
        print("Coefficients:", lda.coef_)
    plot_cm(y_test, y_pred, "LDA Confusion Matrix")

# -----------------------------
# 2. Decision Tree
# -----------------------------
def run_decision_tree(X_train, X_test, y_train, y_test, **kwargs):
    dt = DecisionTreeClassifier(**kwargs)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("=== Decision Tree ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    if hasattr(dt, "feature_importances_"):
        print("Feature importances:", dt.feature_importances_)
    plot_cm(y_test, y_pred, "Decision Tree Confusion Matrix")

# -----------------------------
# 3. Logistic Regression
# -----------------------------
def run_logistic_regression(X_train, X_test, y_train, y_test, **kwargs):
    lr = LogisticRegression(**kwargs)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("=== Logistic Regression ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    if hasattr(lr, "coef_"):
        print("Coefficients:", lr.coef_)
    plot_cm(y_test, y_pred, "Logistic Regression Confusion Matrix")

# -----------------------------
# 4. K-Nearest Neighbors
# -----------------------------
def run_knn(X_train, X_test, y_train, y_test, **kwargs):
    knn = KNeighborsClassifier(**kwargs)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("=== KNN ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_cm(y_test, y_pred, "KNN Confusion Matrix")

# -----------------------------
# 5. SVM
# -----------------------------
def run_svm(X_train, X_test, y_train, y_test, kernel="linear", **kwargs):
    svm = SVC(kernel=kernel, **kwargs)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print(f"=== SVM ({kernel}) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_cm(y_test, y_pred, f"SVM ({kernel}) Confusion Matrix")

# -----------------------------
# 6. Naive Bayes
# -----------------------------
def run_naive_bayes(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    print("=== Naive Bayes ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_cm(y_test, y_pred, "Naive Bayes Confusion Matrix")

# -----------------------------
# 7. Random Forest
# -----------------------------
def run_random_forest(X_train, X_test, y_train, y_test, **kwargs):
    rf = RandomForestClassifier(**kwargs)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("=== Random Forest ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    if hasattr(rf, "feature_importances_"):
        print("Feature importances:", rf.feature_importances_)
    plot_cm(y_test, y_pred, "Random Forest Confusion Matrix")

# -----------------------------
# 8. Neural Network (MLP)
# -----------------------------
def run_neural_network(X_train, X_test, y_train, y_test, **kwargs):
    mlp = MLPClassifier(**kwargs)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    print("=== Neural Network (MLP) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    plot_cm(y_test, y_pred, "MLP Confusion Matrix")


if __name__ == '__main__':
    # Load and preprocess
    X, y, feature_names = load_and_prepare_data()

    # Chronological split + SMOTE
    X_train, y_train, X_test, y_test = train_test_split_chronological(X, y)

    # Run models
    run_lda(X_train, X_test, y_train, y_test)
    run_decision_tree(X_train, X_test, y_train, y_test, max_depth=5)
    run_logistic_regression(X_train, X_test, y_train, y_test, max_iter=500)
    run_random_forest(X_train, X_test, y_train, y_test, n_estimators=200)
    run_neural_network(X_train, X_test, y_train, y_test, hidden_layer_sizes=(100, 50), max_iter=500)
    run_knn(X_train, X_test, y_train, y_test, n_neighbors=7)
    run_svm(X_train, X_test, y_train, y_test, kernel="rbf")
    run_naive_bayes(X_train, X_test, y_train, y_test)


