import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, RandomizedSearchCV, StratifiedKFold
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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


def load_and_prepare_data(file_path="phaseII_pca_reduced.csv"):  # <-- CRITICAL CHANGE: Load the final PCA file

    df = pd.read_csv(file_path)

    # 7 Numerical PCs + 3 Categorical features = 10 total features (if 7 PC)
    # If you reduced to 6 PCs, this list should be PC1 to PC6 + the 3 OHE features.
    # Let's assume 6 PCs for the reduced model:
    FEATURES_ALL = [
        "PC1", "PC2", "PC3", "PC4", "PC5", "PC6",
        "rusty_diff_0.0", "rusty_diff_1.0", "best_of_5"
    ]
    TARGET = "log_target"  # Assuming this is your binary target (0/1)

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

    # Create an SVD-based LDA model specifically for projection visualization
    lda_visualizer = LinearDiscriminantAnalysis(solver='svd')
    lda_visualizer.fit(x_train, y_train)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Plot 2: ROC Curve
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[1].set_ylabel('True Positive Rate (Sensitivity)')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc="lower right")

    # Plot 3: LDA Separation
    x_test_lda = lda_visualizer.transform(x_test)
    lda_df = pd.DataFrame({'LDA Component': x_test_lda.flatten(), 'Target': y_test})

    sns.histplot(data=lda_df, x='LDA Component', hue='Target', element="step", stat="density", common_norm=False,
                 ax=axes[2])
    axes[2].set_title("LDA Separation (SVD Projection)")
    axes[2].set_xlabel("Discriminant Value")

    plt.tight_layout()

    # 🚨 ADDED: Automatically save the combined figure
    plt.savefig('lda_analysis_visuals.png')

    # Keep plt.show() if you still want the plot to pop up
    # plt.show()

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
        "Best Params": grid_search.best_params_,
        "Best Model": best_model,
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

    # 🚨 ADDED: Automatically save the combined figure
    plt.savefig('logistic_regression_visuals.png')

    #plt.show()

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
        "Best Params": grid_search.best_params_,
        "Best Model": best_model,
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
    # 🚨 CHANGE: Save the plot instead of showing it
    plt.savefig('decision_tree_analysis_visuals.png')

    # plt.show() # Removed to only save the plot

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
        "Best Params": grid_search.best_params_,
        # 💡 ADDED: Return the fitted model object for combined ROC plotting
        "Best Model": best_model
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
    #

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
    # 🚨 CHANGE: Save the plot instead of showing it
    plt.savefig('pre_pruned_dt_analysis_visuals.png')

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
        "Best Params": grid_search.best_params_,
        "Best Model": best_model
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

    # 🚨 CHANGE: Save the Alpha vs. Accuracy plot
    plt.savefig('post_pruned_dt_alpha_vs_accuracy.png')
    plt.close()  # Close the figure to free memory

    print(f"\nOptimal Alpha based on 5-Fold CV: {best_alpha:.6f}")

    # ---------------------------------------
    # 4. Final Model Training and Evaluation
    # ---------------------------------------

    # Train the final model using the optimal alpha found via CV
    best_model = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
    best_model.fit(x_train, y_train)

    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)[:, 1]

    # ... (rest of metric calculation code) ...
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
    plt.savefig('post_pruned_dt_cm_roc_visuals.png')
    # plt.show() # Removed

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
        "Best Params": {'ccp_alpha': best_alpha},
        "Best Model": best_model
    }

def run_knn(x_train, x_test, y_train, y_test, feature_names):
    """
    Runs K-Nearest Neighbors (KNN) with hyperparameter optimization (K),
    Elbow Method visualization, and full evaluation.
    Returns a dictionary of metrics for the comparison table, including the "Best Model".
    """

    # ---------------------------------------
    # 1. Setup Stratified K-Fold & Hyperparams
    # ---------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_clf = KNeighborsClassifier()

    # Range of K values to search for the Elbow Method
    k_range = list(range(1, 26, 2))

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

    # Calculate Mean CV Score for each K (using the best non-K parameters found by GridSearchCV)
    k_scores = []
    for k in k_range:
        # Use the best non-K parameters found by GS for a fair comparison across K
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
    plt.xticks(k_range)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 🚨 CHANGE: Save the Elbow Method plot
    plt.savefig('knn_elbow_method.png')
    plt.close()  # Close the figure to free memory

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
    # 🚨 CHANGE: Save the CM/ROC plot
    plt.savefig('knn_cm_roc_visuals.png')
    # plt.show() # Removed

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
        "Best Params": grid_search.best_params_,
        # 🚨 CHANGE: Return the fitted model object using the key "Best Model"
        "Best Model": best_model
    }


def run_svm(x_train, x_test, y_train, y_test, feature_names):
    """
    Runs Support Vector Machine (SVM) with Grid Search over Linear, Poly, and RBF kernels,
    and performs full evaluation.
    Returns a dictionary of metrics for the comparison table, including the "Best Model".
    """

    # ---------------------------------------
    # 1. Setup Stratified K-Fold & Hyperparams
    # ---------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # SVC with probability=True is needed for ROC/AUC calculation
    svm_clf = SVC(random_state=42, probability=True)

    # Define the Hyperparameter Grid for all kernels
    param_grid = [
        # Linear Kernel
        {'kernel': ['linear'], 'C': [0.1, 1, 10]},

        # RBF Kernel (Radial Basis Function - Gaussian)
        {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01, 'scale']},

        # Polynomial Kernel
        {'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'gamma': ['scale']}
    ]

    # ---------------------------------------
    # 2. Perform Grid Search
    # ---------------------------------------
    print("Starting SVM Grid Search...")
    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=svm_clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    end_time = time.time()

    best_model = grid_search.best_estimator_

    print(f"\nSVM Grid Search finished in {end_time - start_time:.2f} seconds. ⏱️")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (Accuracy): {grid_search.best_score_:.4f}")

    # ---------------------------------------
    # 3. Final Predictions and Metrics
    # ---------------------------------------
    y_pred = best_model.predict(x_test)

    # Get probabilities for ROC curve (Probability of Class 1)
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
    print("\n--- SVM Performance Metrics (Optimized Model) ---")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Sensitivity: {recall:.4f} (Recall)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F-Score:     {f1:.4f}")
    print(f"AUC:         {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---------------------------------------
    # 4. Visualizations
    # ---------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_).plot(ax=axes[0], cmap='Purples')
    axes[0].set_title(f"Optimized SVM Confusion Matrix ({best_model.kernel.upper()} Kernel)")

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
    # 🚨 CHANGE: Save the plot instead of showing it
    plt.savefig('svm_cm_roc_visuals.png')
    # plt.show() # Removed

    # ---------------------------------------
    # 5. Return Data for Comparison Table
    # ---------------------------------------
    return {
        "Classifier": "Support Vector Machine",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F-Score": f1,
        "AUC": roc_auc,
        "Best Params": grid_search.best_params_,
        # 🚨 CHANGE: Return the fitted model object using the key "Best Model"
        "Best Model": best_model
    }


def run_mlp_neural_network(x_train, x_test, y_train, y_test, feature_names):
    """
    Runs Multi-Layered Perceptron (MLP) with Randomized Search, evaluation,
    and saves visual plots.
    Returns a dictionary of metrics for the comparison table, including the "Best Model".
    """

    # Define filenames for saving plots
    classifier_name = "MLP_Neural_Network"
    output_filename = f"{classifier_name}_Visuals.png"

    # ---------------------------------------
    # 1. Setup Stratified K-Fold & Hyperparams
    # ---------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # max_iter is often increased for MLPs; setting it to 500 gives time to converge.
    mlp_clf = MLPClassifier(max_iter=500, random_state=42)

    # Define the Hyperparameter Grid for Randomized Search
    param_grid = {
        # Architecture: Testing different sizes (small, medium, large)
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],

        # Non-linear activation functions
        'activation': ['tanh', 'relu'],

        # Regularization (L2 penalty) - alpha is regularization strength
        'alpha': np.logspace(-5, -1, 5),  # [0.00001, 0.0001, 0.001, 0.01, 0.1]

        # Solver for weight optimization
        'solver': ['adam'],  # 'adam' is generally faster and highly effective

        # Learning rate
        'learning_rate_init': [0.001, 0.01]
    }

    # ---------------------------------------
    # 2. Perform Randomized Search (Faster Optimization)
    # ---------------------------------------
    print("Starting MLP Neural Network Randomized Search...")
    start_time = time.time()

    # Use RandomizedSearchCV due to the large search space
    random_search = RandomizedSearchCV(
        estimator=mlp_clf,
        param_distributions=param_grid,
        n_iter=50,  # Test 50 random combinations for efficiency
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    random_search.fit(x_train, y_train)
    end_time = time.time()

    best_model = random_search.best_estimator_

    print(f"\nMLP Randomized Search finished in {end_time - start_time:.2f} seconds. ⏱️")
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best CV Score (Accuracy): {random_search.best_score_:.4f}")

    # ---------------------------------------
    # 3. Final Predictions and Metrics
    # ---------------------------------------
    y_pred = best_model.predict(x_test)

    # Get probabilities for ROC curve (Probability of Class 1)
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
    print("\n--- MLP Neural Network Performance Metrics (Optimized Model) ---")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Sensitivity: {recall:.4f} (Recall)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F-Score:     {f1:.4f}")
    print(f"AUC:         {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---------------------------------------
    # 4. Visualizations and Saving Plots
    # ---------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Confusion Matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    cm_display.plot(ax=axes[0], cmap='cividis')
    axes[0].set_title("Optimized MLP Confusion Matrix")

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
    # 🚨 CHANGE: Save the combined figure and remove plt.show()
    plt.savefig(output_filename)
    print(f"Saved Confusion Matrix and ROC Curve to: {output_filename}")
    # plt.show() # Removed

    # ---------------------------------------
    # 5. Return Data for Comparison Table
    # ---------------------------------------
    return {
        "Classifier": "Multi-Layered Perceptron",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F-Score": f1,
        "AUC": roc_auc,
        "Best Params": random_search.best_params_,
        # 🚨 CHANGE: Return the fitted model object using the key "Best Model"
        "Best Model": best_model
    }

def run_naive_bayes(x_train, x_test, y_train, y_test, feature_names):
    """
    Runs Gaussian Naïve Bayes (GNB) with hyperparameter tuning, evaluation,
    and saves visual plots.
    Returns a dictionary of metrics for the comparison table.
    """

    # Define filenames for saving plots
    classifier_name = "Naive_Bayes"
    cm_filename = f"{classifier_name}_Confusion_Matrix.png"
    roc_filename = f"{classifier_name}_ROC_Curve.png"

    # ---------------------------------------
    # 1. Setup Stratified K-Fold & Hyperparams
    # ---------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gnb_clf = GaussianNB()

    # The 'var_smoothing' parameter is the most common hyperparameter for GNB.
    # It adds a small value to the variances to ensure stability (prevent division by zero).
    param_grid = {
        'var_smoothing': np.logspace(0, -9, num=100)  # Search a wide range (100 values)
    }

    # ---------------------------------------
    # 2. Perform Grid Search
    # ---------------------------------------
    print("Starting Naïve Bayes Grid Search...")
    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=gnb_clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    end_time = time.time()

    best_model = grid_search.best_estimator_

    print(f"\nNaïve Bayes Grid Search finished in {end_time - start_time:.2f} seconds. ⏱️")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (Accuracy): {grid_search.best_score_:.4f}")

    # ---------------------------------------
    # 3. Final Predictions and Metrics
    # ---------------------------------------
    y_pred = best_model.predict(x_test)

    # Get probabilities for ROC curve (Probability of Class 1)
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
    print("\n--- Naïve Bayes Performance Metrics (Optimized Model) ---")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Sensitivity: {recall:.4f} (Recall)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F-Score:     {f1:.4f}")
    print(f"AUC:         {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---------------------------------------
    # 4. Visualizations and Saving Plots
    # ---------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Confusion Matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    cm_display.plot(ax=axes[0], cmap='Oranges')
    axes[0].set_title("Optimized Naïve Bayes Confusion Matrix")

    # Save Confusion Matrix
    cm_display.figure_.savefig(cm_filename, bbox_inches='tight')
    print(f"Saved Confusion Matrix to: {cm_filename}")

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
    # Save ROC Curve
    plt.savefig(roc_filename, bbox_inches='tight')
    print(f"Saved ROC Curve to: {roc_filename}")
    #plt.show()

    # ---------------------------------------
    # 5. Return Data for Comparison Table
    # ---------------------------------------
    return {
        "Classifier": "Naïve Bayes",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F-Score": f1,
        "AUC": roc_auc,
        "Best Params": grid_search.best_params_,
        "Best Model": best_model,
    }

def run_random_forest(x_train, x_test, y_train, y_test, feature_names):
    """
    Runs Random Forest (Bagging) with Randomized Search, evaluation,
    and saves visual plots.
    Returns a dictionary of metrics for the comparison table, including the "Best Model".
    """

    # Define filenames for saving plots
    classifier_name = "Random_Forest"
    output_filename = f"{classifier_name}_Visuals.png"

    # ---------------------------------------
    # 1. Setup Stratified K-Fold & Hyperparams
    # ---------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_clf = RandomForestClassifier(random_state=42)

    # Define the Hyperparameter Grid for Randomized Search
    param_grid = {
        'n_estimators': [100, 200, 300, 500],  # Number of trees in the forest
        'max_features': ['sqrt', 'log2'],  # Number of features to consider for best split
        'max_depth': [10, 20, 30, None],  # Maximum number of levels in tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at a leaf node
        'criterion': ['gini', 'entropy']
    }

    # ---------------------------------------
    # 2. Perform Randomized Search (for Efficiency)
    # ---------------------------------------
    print("Starting Random Forest Randomized Search...")
    start_time = time.time()

    random_search = RandomizedSearchCV(
        estimator=rf_clf,
        param_distributions=param_grid,
        n_iter=50,  # Test 50 random combinations
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    random_search.fit(x_train, y_train)
    end_time = time.time()

    best_model = random_search.best_estimator_

    print(f"\nRandom Forest Randomized Search finished in {end_time - start_time:.2f} seconds. ⏱️")
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best CV Score (Accuracy): {random_search.best_score_:.4f}")

    # ---------------------------------------
    # 3. Final Predictions and Metrics
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
    print("\n--- Random Forest Performance Metrics (Optimized Model) ---")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Sensitivity: {recall:.4f} (Recall)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F-Score:     {f1:.4f}")
    print(f"AUC:         {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---------------------------------------
    # 4. Visualizations and Saving Plots
    # ---------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Confusion Matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    cm_display.plot(ax=axes[0], cmap='YlGnBu')
    axes[0].set_title("Random Forest Confusion Matrix")

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
    # 🚨 CHANGE: Save the combined figure once and remove plt.show()
    plt.savefig(output_filename)
    print(f"Saved Confusion Matrix and ROC Curve to: {output_filename}")
    # plt.show() # Removed

    # ---------------------------------------
    # 5. Return Data for Comparison Table
    # ---------------------------------------
    return {
        "Classifier": "Random Forest (Bagging)",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F-Score": f1,
        "AUC": roc_auc,
        "Best Params": random_search.best_params_,
        # 🚨 CHANGE: Return the fitted model object using the key "Best Model"
        "Best Model": best_model
    }
def plot_all_roc_curves(models_and_names, X_test, y_test):
    """
    Generates and saves a single plot showing the ROC curves for multiple models.

    Args:
        models_and_names (dict): A dictionary mapping model names to fitted model objects.
        X_test (np.array): The test feature matrix.
        y_test (np.array): The test target vector.
    """
    plt.figure(figsize=(10, 8))

    # Plot the baseline random curve
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')

    for name, model in models_and_names.items():
        # Get prediction probabilities for the positive class (Class 1)
        try:
            # Check if the model has predict_proba (most classifiers do)
            y_prob = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            # Fallback for models like some LDA solvers
            y_prob = model.decision_function(X_test)

        # Calculate ROC curve metrics
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Plot the curve
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

    # Final plot styling and saving
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('all_models_roc_comparison.png')
    plt.show()
    print("Saved combined ROC curve plot to 'all_models_roc_comparison.png'.")

if __name__ == '__main__':
    # Load and preprocess
    xTrain, xTest, yTrain, yTest, feature_names = load_and_prepare_data()

    lda_results = run_lda(xTrain, xTest, yTrain, yTest, feature_names)
    print("LDA Results: ", lda_results)

    log_reg_results = run_logistic_regression(xTrain, xTest, yTrain, yTest, feature_names)
    print("Logistic Regression Results: ", log_reg_results)

    decision_tree_results = run_decision_tree(xTrain, xTest, yTrain, yTest, feature_names)
    print("Decision Tree Results: ", decision_tree_results)

    pre_pruned_tree_results = run_pre_pruned_tree(xTrain, xTest, yTrain, yTest, feature_names)
    print("Pre-pruned Tree Results: ", pre_pruned_tree_results)

    post_prune_tree_results = run_post_prune_tree(xTrain, xTest, yTrain, yTest, feature_names)
    print("Post-pruned Tree Results: ", post_prune_tree_results)

    knn_results = run_knn(xTrain, xTest, yTrain, yTest, feature_names)
    print("KNN Results: ", knn_results)

    random_forest_results = run_random_forest(xTrain, xTest, yTrain, yTest, feature_names)
    print("Random Forest Results: ", random_forest_results)

    svm_results = run_svm(xTrain, xTest, yTrain, yTest, feature_names)
    print("SVM Results: ", svm_results)

    naive_bayes_results = run_naive_bayes(xTrain, xTest, yTrain, yTest, feature_names)
    print("Naive Bayes Results: ", naive_bayes_results)

    neural_net_results = run_mlp_neural_network(xTrain, xTest, yTrain, yTest, feature_names)
    print("MLP Neural Network Results: ", neural_net_results)

    models_to_plot = {
        "LDA": lda_results["Best Model"],
        "Logistic Regression": log_reg_results["Best Model"],
        "Decision Tree": decision_tree_results["Best Model"],
        "Pre-pruned Tree": pre_pruned_tree_results["Best Model"],
        "Post-pruned Tree": post_prune_tree_results["Best Model"],
        "KNN": knn_results["Best Model"],
        "Random Forest": random_forest_results["Best Model"],
        "SVM": svm_results["Best Model"],
        "Naive Bayes": naive_bayes_results["Best Model"],
        "MLP Neural Network": neural_net_results["Best Model"],


    }

    plot_all_roc_curves(models_to_plot, xTest, yTest)