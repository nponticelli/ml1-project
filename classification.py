import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    classification_report,
    ConfusionMatrixDisplay
)
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree


def load_and_prepare_data(file_path="phaseII.csv"):
    # Load dataset
    df = pd.read_csv(file_path)
    FEATURES_NUM = [
        "combined_elo_diff",
        "ace_pct_diff",
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
    """
    Runs Linear Discriminant Analysis with Grid Search, Evaluation, and Visualization.
    Returns a dictionary of metrics for the comparison table.
    """

    # ---------------------------------------
    # 1. Setup Stratified K-Fold & Hyperparams
    # ---------------------------------------
    # Explicitly using StratifiedKFold to satisfy requirements
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lda = LinearDiscriminantAnalysis()

    # Solver 'svd' is the default and does not support shrinkage.
    # 'lsqr' and 'eigen' support shrinkage.
    param_grid = [
        {'solver': ['svd']},
        {'solver': ['lsqr'], 'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]}
    ]

    # ---------------------------------------
    # 2. Perform Grid Search
    # ---------------------------------------
    print("Starting LDA Grid Search...")
    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=lda,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    end_time = time.time()

    best_model = grid_search.best_estimator_
    print(f"LDA Grid Search finished in {end_time - start_time:.2f} seconds.")
    print(f"Best Parameters: {grid_search.best_params_}")

    # ---------------------------------------
    # 3. Predictions and Probabilities
    # ---------------------------------------
    y_pred = best_model.predict(x_test)

    # Get probabilities for ROC curve (Class 1)
    # Note: LDA can only predict proba if the assumptions are met or using specific solvers,
    # but sklearn handles this for standard LDA.
    try:
        y_prob = best_model.predict_proba(x_test)[:, 1]
    except AttributeError:
        # Fallback if solver doesn't support probabilities (rare in sklearn LDA)
        y_prob = best_model.decision_function(x_test)

    # ---------------------------------------
    # 4. Calculate Metrics
    # ---------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Also known as Sensitivity
    f1 = f1_score(y_test, y_pred)

    # Specificity Calculation: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC / AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Print Text Report
    print("\n--- LDA Performance Metrics ---")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Sensitivity: {recall:.4f} (Recall)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F-Score:     {f1:.4f}")
    print(f"AUC:         {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---------------------------------------
    # 5. Visualizations
    # ---------------------------------------

    # Create an SVD-based LDA model specifically for projection visualization,
    # as the best_model (potentially 'lsqr') doesn't support .transform().
    lda_visualizer = LinearDiscriminantAnalysis(solver='svd')
    lda_visualizer.fit(x_train, y_train)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Confusion Matrix (Uses best_model)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Plot 2: ROC Curve (Uses y_prob from best_model)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[1].set_ylabel('True Positive Rate (Sensitivity)')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc="lower right")

    # Plot 3: LDA Separation (Fisher Criterion Visualization)
    # Project data using the lda_visualizer (SVD solver)
    x_test_lda = lda_visualizer.transform(x_test)

    # Create a DataFrame for plotting
    lda_df = pd.DataFrame({'LDA Component': x_test_lda.flatten(), 'Target': y_test})

    sns.histplot(data=lda_df, x='LDA Component', hue='Target', element="step", stat="density", common_norm=False,
                 ax=axes[2])
    axes[2].set_title("LDA Separation (SVD Projection)")
    axes[2].set_xlabel("Discriminant Value")

    plt.tight_layout()
    plt.show()
    # ---------------------------------------
    # 6. Return Data for Comparison Table
    # ---------------------------------------
    return {
        "Classifier": "LDA",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F-Score": f1,
        "AUC": roc_auc,
        "Best Params": grid_search.best_params_
    }


def run_logistic_regression(x_train, x_test, y_train, y_test, feature_names):
    """
    Runs Logistic Regression with Grid Search, Evaluation, and Visualization.
    Returns a dictionary of metrics for the comparison table.
    """

    # ---------------------------------------
    # 1. Setup Stratified K-Fold & Hyperparams
    # ---------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logreg = LogisticRegression(solver='liblinear', random_state=42)

    # Define the Hyperparameter Grid for C (Inverse of regularization strength)
    param_grid = [
        # L1 regularization (Lasso)
        {'penalty': ['l1'], 'C': np.logspace(-3, 1, 5)},
        # L2 regularization (Ridge)
        {'penalty': ['l2'], 'C': np.logspace(-3, 1, 5)}
    ]
    # np.logspace(-3, 1, 5) -> [0.001, 0.01, 0.1, 1.0, 10.0]

    # ---------------------------------------
    # 2. Perform Grid Search
    # ---------------------------------------
    print("Starting Logistic Regression Grid Search...")
    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=logreg,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    end_time = time.time()

    best_model = grid_search.best_estimator_
    print(f"Logistic Regression Grid Search finished in {end_time - start_time:.2f} seconds.")
    print(f"Best Parameters: {grid_search.best_params_}")

    # ---------------------------------------
    # 3. Predictions and Probabilities
    # ---------------------------------------
    y_pred = best_model.predict(x_test)

    # Get probabilities for ROC curve (Probability of Class 1)
    y_prob = best_model.predict_proba(x_test)[:, 1]

    # ---------------------------------------
    # 4. Calculate Metrics
    # ---------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Standard Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Sensitivity
    f1 = f1_score(y_test, y_pred)

    # Specificity Calculation: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC / AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Print Text Report
    print("\n--- Logistic Regression Performance Metrics ---")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Sensitivity: {recall:.4f} (Recall)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F-Score:     {f1:.4f}")
    print(f"AUC:         {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---------------------------------------
    # 5. Visualizations
    # ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_).plot(ax=axes[0], cmap='Blues')
    axes[0].set_title("Confusion Matrix")

    # Plot 2: ROC Curve
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[1].set_ylabel('True Positive Rate (Sensitivity)')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # 6. Return Data for Comparison Table
    # ---------------------------------------
    return {
        "Classifier": "Logistic Regression",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F-Score": f1,
        "AUC": roc_auc,
        "Best Params": grid_search.best_params_
    }


def run_decision_tree(x_train, x_test, y_train, y_test, feature_names):
    """
    Runs Decision Tree with Grid Search, Evaluation, and Visualization.
    Returns a dictionary of metrics for the comparison table.
    """

    # ---------------------------------------
    # 1. Setup Stratified K-Fold & Hyperparams
    # ---------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    dt_clf = DecisionTreeClassifier(random_state=42)

    # Define the Hyperparameter Grid
    param_grid = {
        'criterion': ['gini', 'entropy'],  # Measure for split quality
        'max_depth': [3, 5, 7, 10, None],  # Max depth of the tree (None means unlimited)
        'min_samples_leaf': [1, 5, 10],  # Minimum samples required to be at a leaf node
        'min_samples_split': [2, 5, 10]  # Minimum samples required to split an internal node
    }

    # ---------------------------------------
    # 2. Perform Grid Search
    # ---------------------------------------
    print("Starting Decision Tree Grid Search...")
    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=dt_clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    end_time = time.time()

    best_model = grid_search.best_estimator_
    print(f"Decision Tree Grid Search finished in {end_time - start_time:.2f} seconds.")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (Accuracy): {grid_search.best_score_:.4f}")

    # ---------------------------------------
    # 3. Predictions and Probabilities
    # ---------------------------------------
    y_pred = best_model.predict(x_test)

    # Get probabilities for ROC curve (Probability of Class 1)
    y_prob = best_model.predict_proba(x_test)[:, 1]

    # ---------------------------------------
    # 4. Calculate Metrics
    # ---------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Standard Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Sensitivity
    f1 = f1_score(y_test, y_pred)

    # Specificity Calculation: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC / AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Print Text Report
    print("\n--- Decision Tree Performance Metrics (Optimized Model) ---")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Sensitivity: {recall:.4f} (Recall)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F-Score:     {f1:.4f}")
    print(f"AUC:         {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---------------------------------------
    # 5. Visualizations
    # ---------------------------------------

    # Create the figure with 3 subplots
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3)

    # Plot 1: Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_).plot(ax=ax1, cmap='Reds')
    ax1.set_title("Optimized Decision Tree CM")

    # Plot 2: ROC Curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate (1 - Specificity)')
    ax2.set_ylabel('True Positive Rate (Sensitivity)')
    ax2.set_title('ROC Curve')
    ax2.legend(loc="lower right")

    # Plot 3: Best Decision Tree Structure

    ax3 = fig.add_subplot(gs[0, 2])
    plot_tree(
        best_model,
        feature_names=feature_names,
        class_names=[str(c) for c in best_model.classes_],
        filled=True,
        rounded=True,
        fontsize=6,
        ax=ax3
    )
    ax3.set_title(f"Optimized Decision Tree (Depth: {best_model.max_depth})")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # 6. Return Data for Comparison Table
    # ---------------------------------------
    return {
        "Classifier": "Decision Tree",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F-Score": f1,
        "AUC": roc_auc,
        "Best Params": grid_search.best_params_
    }


def run_pre_pruned_tree(x_train, x_test, y_train, y_test, feature_names):
    """
    Runs Pre-Pruned Decision Tree with Grid Search, Evaluation, and Visualization.
    Returns a dictionary of metrics for the comparison table.
    """

    # ---------------------------------------
    # 1. Setup Stratified K-Fold & Hyperparams
    # ---------------------------------------
    # Explicitly define StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    dt_clf = DecisionTreeClassifier(random_state=42)

    # The Grid enforces pre-pruning by setting limits on depth and minimum samples
    tuned_parameters = [
        {'max_depth': [1, 2, 3, 4, 5, 7, 10],
         'min_samples_split': [20, 30, 40, 60],  # Requires more samples to split
         'min_samples_leaf': [10, 20, 30, 50],  # Requires more samples per leaf
         'criterion': ['gini', 'entropy'],  # log_loss is equivalent to entropy, using both is redundant
         'splitter': ['best'],  # Using 'best' is usually sufficient
         'max_features': [None, 'sqrt', 'log2']}]  # None: use all features

    # ---------------------------------------
    # 2. Perform Grid Search
    # ---------------------------------------
    print("Starting Pre-Pruned Decision Tree Grid Search...")
    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=dt_clf,
        param_grid=tuned_parameters,
        scoring='accuracy',
        cv=cv,  # Using Stratified K-Fold
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    end_time = time.time()

    best_model = grid_search.best_estimator_
    print(f"Pre-Pruned DT Grid Search finished in {end_time - start_time:.2f} seconds.")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (Accuracy): {grid_search.best_score_:.4f}")

    # ---------------------------------------
    # 3. Predictions and Probabilities
    # ---------------------------------------
    y_pred = best_model.predict(x_test)

    # Get probabilities for ROC curve (Probability of Class 1)
    y_prob = best_model.predict_proba(x_test)[:, 1]

    # ---------------------------------------
    # 4. Calculate Metrics
    # ---------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Standard Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Sensitivity
    f1 = f1_score(y_test, y_pred)

    # Specificity Calculation: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC / AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Print Text Report
    print("\n--- Pre-Pruned DT Performance Metrics ---")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Sensitivity: {recall:.4f} (Recall)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F-Score:     {f1:.4f}")
    print(f"AUC:         {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---------------------------------------
    # 5. Visualizations
    # ---------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Plot 1: Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_).plot(ax=axes[0], cmap='Greens')
    axes[0].set_title("Pre-Pruned DT Confusion Matrix")

    # Plot 2: ROC Curve
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[1].set_ylabel('True Positive Rate (Sensitivity)')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc="lower right")

    # Plot 3: Optimized Tree Structure
    plot_tree(
        best_model,
        feature_names=feature_names,
        class_names=[str(c) for c in best_model.classes_],
        filled=True,
        rounded=True,
        fontsize=7,
        ax=axes[2]
    )
    axes[2].set_title(f"Optimized Pre-Pruned Tree (Depth: {best_model.max_depth})")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # 6. Return Data for Comparison Table
    # ---------------------------------------
    return {
        "Classifier": "Pre-Pruned Decision Tree",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F-Score": f1,
        "AUC": roc_auc,
        "Best Params": grid_search.best_params_
    }


def run_post_prune_tree(x_train, x_test, y_train, y_test, feature_names):
    """
    Runs Post-Pruned Decision Tree using CCP and generates the Accuracy vs. Alpha plot
    based on Cross-Validation scores, ensuring computational efficiency via alpha sampling.
    """

    # ---------------------------------------
    # 1. Determine Pruning Path (Alpha values) and Sample
    # ---------------------------------------
    unpruned_dt = DecisionTreeClassifier(random_state=42)
    unpruned_dt.fit(x_train, y_train)

    prune_path = unpruned_dt.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas = prune_path.ccp_alphas
    ccp_alphas = ccp_alphas[:-1]  # Exclude alpha that prunes to the root

    # --- Re-integrated Alpha Sampling for performance safeguard ---
    if len(ccp_alphas) > 100:
        print(f"Sampling {len(ccp_alphas)} alphas down to 100 for efficiency...")
        alpha_indices = np.linspace(0, len(ccp_alphas) - 1, 100).astype(int)
        sampled_alphas = ccp_alphas[alpha_indices]
    else:
        sampled_alphas = ccp_alphas

    # ---------------------------------------
    # 2. Evaluate Alpha Values using Stratified K-Fold CV
    # ---------------------------------------
    print(f"Starting cross-validation to find optimal alpha over {len(sampled_alphas)} samples...")
    start_time = time.time()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    alpha_mean_scores = []

    for alpha in sampled_alphas:  # <-- Using sampled_alphas
        dt = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        scores = cross_val_score(dt, x_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        alpha_mean_scores.append(scores.mean())

    end_time = time.time()
    print(f"Alpha CV Evaluation finished in {end_time - start_time:.2f} seconds.")

    # Find the alpha that gave the best CV accuracy
    best_alpha_index = np.argmax(alpha_mean_scores)
    best_alpha = sampled_alphas[best_alpha_index]  # <-- Indexing into sampled_alphas

    # ---------------------------------------
    # 3. Plot Accuracy vs. Alpha Value 📈
    # ---------------------------------------
    plt.figure(figsize=(10, 6))
    # Note: We plot the mean scores against the sampled_alphas
    plt.plot(sampled_alphas, alpha_mean_scores, marker='o', drawstyle="steps-post", label='5-Fold CV Accuracy')
    plt.axvline(best_alpha, color='r', linestyle='--', label=f'Optimal Alpha: {best_alpha:.6f}')
    plt.xlabel("Cost Complexity Pruning Alpha Value ($c_p$)")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.title("Post-Pruning: CV Accuracy vs. Alpha Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    print(f"\nOptimal Alpha based on 5-Fold CV: {best_alpha:.6f}")

    # ---------------------------------------
    # 4. Final Model Training and Evaluation
    # ---------------------------------------

    # Train the final model using the optimal alpha found via CV
    best_model = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
    best_model.fit(x_train, y_train)

    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)[:, 1]

    # ... (rest of metric and visualization code remains the same) ...
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print("\n--- Final Post-Pruned DT Performance Metrics ---")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Sensitivity: {recall:.4f} (Recall)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F-Score:     {f1:.4f}")
    print(f"AUC:         {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---------------------------------------
    # 5. Visualizations
    # ---------------------------------------
    target_classes = best_model.classes_

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_classes).plot(ax=axes[0], cmap='Purples')
    axes[0].set_title("Post-Pruned DT Confusion Matrix")

    # Plot 2: ROC Curve
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[1].set_ylabel('True Positive Rate (Sensitivity)')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # 6. Return Data for Comparison Table
    # ---------------------------------------
    return {
        "Classifier": "Post-Pruned Decision Tree (CCP)",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F-Score": f1,
        "AUC": roc_auc,
        "Best Params": {'ccp_alpha': best_alpha}
    }


def run_knn(x_train, x_test, y_train, y_test, feature_names):
    """
    Runs K-Nearest Neighbors (KNN) with hyperparameter optimization (K),
    Elbow Method visualization, and full evaluation.
    Returns a dictionary of metrics for the comparison table.
    """

    # ---------------------------------------
    # 1. Setup Stratified K-Fold & Hyperparams
    # ---------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_clf = KNeighborsClassifier()

    # Range of K values to search for the Elbow Method
    k_range = list(range(1, 22, 2))

    # Define the Hyperparameter Grid for GridSearchCV (Formal Optimization)
    param_grid = {
        'n_neighbors': k_range,
        'weights': ['uniform', 'distance'],  # Weighting strategies
        'metric': ['euclidean', 'manhattan']  # Distance metrics
    }

    # ---------------------------------------
    # 2. Perform Grid Search (Formal Optimization)
    # ---------------------------------------
    print("Starting KNN Grid Search...")
    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=knn_clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    end_time = time.time()

    best_model = grid_search.best_estimator_
    best_k = grid_search.best_params_['n_neighbors']

    print(f"KNN Grid Search finished in {end_time - start_time:.2f} seconds.")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Optimal K found via Grid Search: {best_k}")

    # ---------------------------------------
    # 3. Elbow Method Visualization (Finding Optimum K) 📈
    # ---------------------------------------

    # Extract mean cross-validation scores for each K value (across all other parameters)
    # The elbow method typically plots accuracy vs K, often ignoring other tuning parameters.

    # Calculate Mean CV Score for each K (using the best non-K parameters found by GridSearchCV)
    mean_scores = grid_search.cv_results_['mean_test_score']

    # Restructure scores to average over the other parameters for a clean K vs Accuracy plot
    # The structure depends on param_grid order, so let's simplify by running a dedicated loop:

    k_scores = []
    for k in k_range:
        knn_temp = KNeighborsClassifier(n_neighbors=k,
                                        weights=grid_search.best_params_['weights'],
                                        metric=grid_search.best_params_['metric'])
        scores = cross_val_score(knn_temp, x_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        k_scores.append(scores.mean())

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, k_scores, marker='o', linestyle='-', color='teal')
    plt.axvline(best_k, color='red', linestyle='--', label=f'Optimal K = {best_k}')
    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Mean 5-Fold Cross-Validation Accuracy")
    plt.title("KNN Elbow Method: Optimizing K")
    plt.xticks(k_range[::2])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()  #

    # ---------------------------------------
    # 4. Final Predictions and Metrics
    # ---------------------------------------
    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Standard Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Sensitivity
    f1 = f1_score(y_test, y_pred)

    # Specificity Calculation: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC / AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Print Text Report
    print("\n--- KNN Performance Metrics (Optimized Model) ---")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Sensitivity: {recall:.4f} (Recall)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F-Score:     {f1:.4f}")
    print(f"AUC:         {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---------------------------------------
    # 5. Visualizations
    # ---------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_).plot(ax=axes[0], cmap='Blues')
    axes[0].set_title("Optimized KNN Confusion Matrix")

    # Plot 2: ROC Curve
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[1].set_ylabel('True Positive Rate (Sensitivity)')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # 6. Return Data for Comparison Table
    # ---------------------------------------
    return {
        "Classifier": "K-Nearest Neighbors",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F-Score": f1,
        "AUC": roc_auc,
        "Best Params": grid_search.best_params_
    }

if __name__ == '__main__':
    # Load and preprocess
    xTrain, xTest, yTrain, yTest, feature_names = load_and_prepare_data()

    #run_lda(xTrain, xTest, yTrain, yTest, feature_names)

    #run_logistic_regression(xTrain, xTest, yTrain, yTest, feature_names)

    #run_decision_tree(xTrain, xTest, yTrain, yTest, feature_names)

    #run_pre_pruned_tree(xTrain, xTest, yTrain, yTest, feature_names)

    #run_post_prune_tree(xTrain, xTest, yTrain, yTest, feature_names)

    run_knn(xTrain, xTest, yTrain, yTest, feature_names)

    #run_random_forest(xTrain, xTest, yTrain, yTest, feature_names)


