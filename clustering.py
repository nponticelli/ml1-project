
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, \
    mutual_info_score
import warnings
warnings.filterwarnings("ignore") # Suppress warnings related to K-Means convergence

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


def run_clustering(x_train, x_test, y_train, y_test, feature_names):
    """
    Performs K-Means clustering analysis for K=2 to 15, including all requested metrics, and saves the plots.
    """

    # --- Configuration ---
    max_k = 25
    K_range = range(2, max_k + 1)
    SAMPLE_SIZE = 10000

    # --- Data Sampling for Metrics (Crucial for performance) ---
    if x_train.shape[0] > SAMPLE_SIZE:
        np.random.seed(42)
        sample_indices = np.random.choice(x_train.shape[0], size=SAMPLE_SIZE, replace=False)
        x_sample = x_train[sample_indices]
        y_sample = y_train[sample_indices]  # Sample the target too
        print(f"\nSampling data for metrics (N={SAMPLE_SIZE}) to ensure reasonable execution time.")
    else:
        x_sample = x_train
        y_sample = y_train
        SAMPLE_SIZE = x_train.shape[0]

    # --- Metric Storage ---
    inertia_values = []
    silhouette_scores = []
    db_scores = []
    ch_scores = []
    mis_scores = []
    ars_scores = []

    print(f"\n--- Running K-Means and Metrics Analysis (K=2 to {max_k}) ---")

    for k in K_range:
        # 1. Apply K-Means (Fit always on full training data)
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(x_train)

        # 2. Store Within-Cluster Sum of Squares (Inertia)
        inertia_values.append(kmeans.inertia_)

        # 3. Predict labels on the sampled data for metric calculation
        sample_labels = kmeans.predict(x_sample)

        # 4. Calculate Clustering Metrics (using sampled data)

        # Intrinsic Metrics (Unsupervised)
        silhouette_avg = silhouette_score(x_sample, sample_labels)
        db_index = davies_bouldin_score(x_sample, sample_labels)
        ch_index = calinski_harabasz_score(x_sample, sample_labels)

        # Extrinsic Metrics (Supervised - requires ground truth y_sample)
        mi_score = mutual_info_score(y_sample, sample_labels)
        ar_score = adjusted_rand_score(y_sample, sample_labels)

        silhouette_scores.append(silhouette_avg)
        db_scores.append(db_index)
        ch_scores.append(ch_index)
        mis_scores.append(mi_score)
        ars_scores.append(ar_score)

        print(
            f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_avg:.4f}, DBI={db_index:.4f}, CHI={ch_index:.2f}, MIS={mi_score:.4f}, ARS={ar_score:.4f}")

    # ----------------------------------------------------
    # Plotting 1: Within-Cluster Variation (Elbow Method)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(K_range, inertia_values, marker='o', linestyle='--', color='blue')
    plt.title('Within-Cluster Variation (Inertia) - Elbow Method', fontsize=16)
    plt.xlabel('Number of Clusters (K)', fontsize=14)
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=14)
    plt.xticks(K_range)
    plt.grid(True, alpha=0.5)
    plt.savefig('k_means_elbow_plot.png')
    plt.close()

    # ----------------------------------------------------
    # Plotting 2: Silhouette Analysis (Average Score)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(K_range, silhouette_scores, marker='o', linestyle='-', color='red')
    plt.title('Average Silhouette Score Analysis', fontsize=16)
    plt.xlabel('Number of Clusters (K)', fontsize=14)
    plt.ylabel('Average Silhouette Score', fontsize=14)
    plt.xticks(K_range)
    plt.grid(True, alpha=0.5)
    plt.savefig('k_means_silhouette_plot.png')
    plt.close()

    # ----------------------------------------------------
    # Plotting 3: All Metrics Analysis (Combined)
    # ----------------------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(10, 25), sharex=True)

    # Intrinsic Metrics (Unsupervised)
    axes[0].plot(K_range, silhouette_scores, marker='o', linestyle='-', color='red')
    axes[0].set_title('Silhouette Score (Higher is Better)', fontsize=14)
    axes[0].set_ylabel('Avg. Silhouette Score')
    axes[0].grid(True, alpha=0.5)

    axes[1].plot(K_range, db_scores, marker='o', linestyle='-', color='green')
    axes[1].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14)
    axes[1].set_ylabel('DBI')
    axes[1].grid(True, alpha=0.5)

    axes[2].plot(K_range, ch_scores, marker='o', linestyle='-', color='purple')
    axes[2].set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=14)
    axes[2].set_ylabel('CHI')
    axes[2].grid(True, alpha=0.5)

    # Extrinsic Metrics (Supervised - uses y_train as ground truth)
    axes[3].plot(K_range, mis_scores, marker='o', linestyle='-', color='darkorange')
    axes[3].set_title('Mutual Information Score (Higher is Better)', fontsize=14)
    axes[3].set_ylabel('MIS')
    axes[3].grid(True, alpha=0.5)

    axes[4].plot(K_range, ars_scores, marker='o', linestyle='-', color='darkblue')
    axes[4].set_title('Adjusted Rand Score (Higher is Better)', fontsize=14)
    axes[4].set_ylabel('ARS')
    axes[4].set_xlabel('Number of Clusters (K)', fontsize=14)
    axes[4].grid(True, alpha=0.5)

    plt.suptitle(f'All Clustering Metrics Analysis (N={SAMPLE_SIZE} Sample)', fontsize=16, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('k_means_all_metrics_plots.png')
    plt.close()

    # --- Summary Output ---
    summary_df = pd.DataFrame({
        'K': K_range,
        'Inertia': inertia_values,
        'Silhouette Score (Sampled)': silhouette_scores,
        'Davies-Bouldin Index (DBI)': db_scores,
        'Calinski-Harabasz Index (CHI)': ch_scores,
        'Mutual Info Score (MIS)': mis_scores,
        'Adjusted Rand Score (ARS)': ars_scores
    })

    print("\n" + "=" * 80)
    print("                      K-MEANS METRICS SUMMARY (K=2 to 15)")
    print("=" * 80)
    print(summary_df.to_markdown(index=False, floatfmt=".4f"))

    print("\n**Clustering Analysis Complete:**")
    print("1. Elbow Plot saved as `k_means_elbow_plot.png`")
    print("2. Average Silhouette Plot saved as `k_means_silhouette_plot.png`")
    print("3. All Metrics Plot saved as `k_means_all_metrics_plots.png` (Includes all metrics)")
if __name__ == '__main__':
    # Load and preprocess
    xTrain, xTest, yTrain, yTest, feature_names = load_and_prepare_data()

    run_clustering(xTrain, xTest, yTrain, yTest, feature_names)

