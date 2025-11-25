import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import matplotlib.pyplot as plt
from numpy.linalg import svd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

def data_cleaning():

    df = pd.read_csv("atp_matches.csv")

    print("This is thet start, and the head of the dataset")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nWe are now cleaning the dataset")

    num_duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {num_duplicates}")

    print("Convert rankings to numeric, and fill blanks with max value (unranked players should be ranked lowly)")
    for col in ["winner_rank", "loser_rank"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["winner_rank", "loser_rank"]:
        max_rank = df[col].max(skipna=True)
        df[col] = df[col].fillna(max_rank)

    print("\nMissing surface will be the mode")
    df['surface'] = df['surface'].fillna(df['surface'].mode()[0])
    for col in ["winner_hand", "loser_hand"]:
        # Compute mode ignoring NaNs
        mode_value = df[col].mode()[0]
        # Replace anything not 'R' or 'L' with mode
        df[col] = df[col].apply(lambda x: x if x in ['R', 'L'] else mode_value)

    print("We want to clean the score column to calculate games difference")

    print("\nSeeding, entry, rank, and points will be N/A if missing")
    seed_entry_rank_cols = ['winner_seed', 'loser_seed',
                            'winner_entry', 'loser_entry',
                            'winner_rank_points',
                            'loser_rank_points']
    df[seed_entry_rank_cols] = df[seed_entry_rank_cols].fillna('N/A')

    missing_after = df.isnull().sum()
    print(missing_after)

    num_duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {num_duplicates}")

    print(df.info())

    print(df.describe())

    # Now we are going to feature engineer columns
    # --- Basic cleaning ---
    df = df.dropna(subset=["winner_id", "loser_id"])
    #New Code

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.sort_values(["tourney_date", "match_num"])

    # Convert rankings
    for col in ["winner_rank", "loser_rank"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].max())

    # Clean score -> game difference target
    def clean_score(s):
        if pd.isna(s): return ""
        s = s.upper()
        s = re.sub(r"\([^)]*\)", "", s)
        s = re.sub(r"[A-Z]+", "", s)
        return s.strip()

    def compute_game_diff(s):
        if not s: return np.nan
        p1 = p2 = 0
        for setscore in s.split():
            try:
                a, b = map(int, setscore.split("-"))
                p1 += a; p2 += b
            except:
                pass
        return p1 - p2

    df["clean_score"] = df["score"].apply(clean_score)
    df["game_diff"] = df["clean_score"].apply(compute_game_diff)

    print("\nAll other columns (game statistical measures) will be the mean of the respective feature")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols_to_fill = [col for col in numeric_cols if col not in seed_entry_rank_cols]
    df[numeric_cols_to_fill] = df[numeric_cols_to_fill].fillna(df[numeric_cols_to_fill].mean())

    # Reorder players by ranking
    def reorder(row):
        if row["winner_rank"] < row["loser_rank"]:
            return pd.Series({
                "higher_id": row["winner_id"],
                "higher_rank": row["winner_rank"],
                "higher_name": row["winner_name"],
                "lower_id": row["loser_id"],
                "lower_rank": row["loser_rank"],
                "lower_name": row["loser_name"],
                "higher_age": row["winner_age"],
                "lower_age": row["loser_age"],
                "log_target": 1
            })
        else:
            return pd.Series({
                "higher_id": row["loser_id"],
                "higher_rank": row["loser_rank"],
                "higher_name": row["loser_name"],
                "lower_id": row["winner_id"],
                "lower_rank": row["winner_rank"],
                "lower_name": row["winner_name"],
                "higher_age": row["loser_age"],
                "lower_age": row["winner_age"],
                "log_target": 0
            })

    df = pd.concat([df, df.apply(reorder, axis=1)], axis=1)

    # ---------------------------
    # 1) ELO RATING SYSTEM
    # ---------------------------

    BASE_ELO = 1500
    K = 32
    elo = defaultdict(lambda: BASE_ELO)

    df["higher_elo"] = np.nan
    df["lower_elo"] = np.nan

    def expected(h, l):
        return 1 / (1 + 10 ** ((l - h) / 400))

    for idx, row in df.iterrows():
        hi, lo = row["higher_id"], row["lower_id"]
        out = row["log_target"]

        h_elo, l_elo = elo[hi], elo[lo]

        df.at[idx, "higher_elo"] = h_elo
        df.at[idx, "lower_elo"] = l_elo

        exp_h = expected(h_elo, l_elo)

        elo[hi] = h_elo + K * (out - exp_h)
        elo[lo] = l_elo + K * ((1 - out) - (1 - exp_h))

    df["elo_difference"] = df["higher_elo"] - df["lower_elo"]

    # -------------------------------
    # 2) SURFACE EWMA
    # -------------------------------

    alpha_value = 0.2
    min_matches = 3
    neutral = 0.5

    surf_stats = defaultdict(lambda: {"score": 0.0, "weight": 0.0, "count": 0})

    df["higher_surface_ewma"] = np.nan
    df["lower_surface_ewma"] = np.nan

    for idx, row in df.iterrows():
        surf = row["surface"]
        hi, lo = row["higher_id"], row["lower_id"]
        winner = row["winner_id"]

        H = surf_stats[(hi, surf)]
        L = surf_stats[(lo, surf)]

        df.at[idx, "higher_surface_ewma"] = H["score"] / H["weight"] if H["count"] >= min_matches else neutral
        df.at[idx, "lower_surface_ewma"] = L["score"] / L["weight"] if L["count"] >= min_matches else neutral

        # update AFTER
        H["score"] += (1 if winner == hi else 0) * alpha_value
        H["weight"] += alpha_value
        H["count"] += 1

        L["score"] += (1 if winner == lo else 0) * alpha_value
        L["weight"] += alpha_value
        L["count"] += 1

    # -------------------------------
    # 3) SHORT-TERM FATIGUE (< 10 days)
    # -------------------------------

    df["pseudo_date"] = df["tourney_date"]
    df = df.sort_values("pseudo_date")

    player_hist = defaultdict(list)  # (date, minutes)
    df["minutes"] = df.get("minutes", 60)

    df["higher_short_fatigue"] = 0.0
    df["lower_short_fatigue"]  = 0.0

    window = pd.Timedelta(days=10)

    for idx, row in df.iterrows():
        date = row["pseudo_date"]
        hi, lo = row["higher_id"], row["lower_id"]

        def fatigue(p):
            return sum(dur for d, dur in player_hist[p] if d >= date - window)

        df.at[idx, "higher_short_fatigue"] = fatigue(hi)
        df.at[idx, "lower_short_fatigue"]  = fatigue(lo)

        # update
        player_hist[hi].append((date, row["minutes"]))
        player_hist[lo].append((date, row["minutes"]))

    all_ages = pd.concat([df["winner_age"], df["loser_age"]])
    mean_age = all_ages.mean()
    std_age = all_ages.std()
    print("Mean age:", mean_age)
    print("Standard deviation of age:", std_age)

    df["higher_dev"] = abs(df["higher_age"] - mean_age)
    df["lower_dev"] = abs(df["lower_age"] - mean_age)

    df["age_dev_diff"] = df["lower_dev"] - df["higher_dev"]

    def compute_age_adv(row):
        if row["age_dev_diff"] > std_age:
            return "higher_seed_age_advantage"
        elif row["age_dev_diff"] < -std_age:
            return "lower_seed_age_advantage"
        else:
            return "no_age_advantage"

    df["age_advantage"] = df.apply(compute_age_adv, axis=1)

    # Final dataset
    fe = df[[
        "pseudo_date",
        "higher_id",
        "higher_name",
        "higher_rank",
        "higher_age",
        "lower_id",
        "lower_rank",
        "lower_name",
        "lower_age",
        "age_advantage",
        "higher_elo",
        "lower_elo",
        "elo_difference",
        "higher_surface_ewma",
        "lower_surface_ewma",
        "higher_short_fatigue",
        "lower_short_fatigue",
        "game_diff",
        "log_target"
    ]]

    fe.to_csv("fe_simplified.csv", index=False)
    print("Saved fe_simplified.csv", fe.shape)

    return fe



def feature_engineering():
    # --- 1. Load dataset for LDA and PCA ---
    df = pd.read_csv("fe_matches.csv")  # assuming you've saved it with all features
    df = df.sort_values("pseudo_date").reset_index(drop=True)
    # --- 3. Select features ---
    features = [
        "ranking_difference",
        "height_difference",
        "age_difference",
        "higher_hand",
        "lower_hand",
        "higher_h2h_win_pct",
        "match_type",
        "higher_surface_ewma",
        "lower_surface_ewma",
        "elo_difference",
        "higher_short_fatigue",
        "lower_short_fatigue",
        "higher_long_fatigue",
        "lower_long_fatigue",
    ]

    cat_features = ["match_type", "higher_hand", "lower_hand"]

    num_features = [
        "ranking_difference",
        "height_difference",
        "age_difference",
        "higher_h2h_win_pct",
        "higher_surface_ewma",
        "lower_surface_ewma",
        "elo_difference",
        "higher_short_fatigue",
        "lower_short_fatigue",
        "higher_long_fatigue",
        "lower_long_fatigue",
    ]

    X = df[features]
    y = df["log_target"]

    iqr_features = ["ranking_difference", "height_difference", "age_difference", "elo_difference"]

    for col in iqr_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # Optional: check the result
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df[iqr_features])
    plt.title("After IQR Capping: Boxplot of Selected Features")
    plt.show()

    # Plot boxplots after capping
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df[num_features])
    plt.title("After Outlier Capping: Boxplot of Numeric Features")
    plt.show()

    print("Outlier capping complete. Numeric features have been Winsorized.")

    # Build Random Forest model
    cutoff_idx = int(len(df) * 0.8)

    # Chronological split
    X_train = X.iloc[:cutoff_idx]
    y_train = y.iloc[:cutoff_idx]

    X_test = X.iloc[cutoff_idx:]
    y_test = y.iloc[cutoff_idx:]

    scaler = StandardScaler()

    x_num_train_scaled = scaler.fit_transform(X_train[num_features])

    x_train_scaled = np.hstack([x_num_train_scaled, X_train[cat_features].values])



    cov_matrix = np.cov(x_num_train_scaled, rowvar=False)
    print("This is the covariance matrix")
    print(cov_matrix)

    x_train_scaled_with_labels = pd.DataFrame(x_train_scaled, columns=num_features)




    # Pearson correlation matrix
    corr_matrix = pd.DataFrame(x_train_scaled_with_labels).corr(method="pearson")

    # Heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Pearson Correlation Heatmap")
    plt.show()

    # --- 6. Apply PCA ---
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(x_train_scaled_with_labels)

    print("Explained variance by PCA components:", pca.explained_variance_ratio_)

    # --- 7. Apply LDA ---
    print("Starting LDA")
    # Create LDA object
    lda = LinearDiscriminantAnalysis(n_components=1)  # For binary target, max n_components = 1
    # Fit LDA on training data
    X_lda = lda.fit_transform(x_train_scaled_with_labels, y)

    print("Shape of LDA-transformed training set:", X_lda.shape)
    print("Explained variance ratio (discriminative power):", lda.explained_variance_ratio_)
    plt.figure(figsize=(8,4))
    plt.hist(X_lda[y==1], alpha=0.5, label='Higher Seed Wins (1)')
    plt.hist(X_lda[y==0], alpha=0.5, label='Lower Seed Wins (0)')
    plt.title("LDA Projection")
    plt.xlabel("LD1")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    #Single Value Decomposition
    print("Single Value Decomposition")

    U, S, Vt = svd(x_num_train_scaled, full_matrices=False)

    print("Singular values:")
    print(S)

    # Compute explained variance ratio from singular values
    variance_explained = (S**2) / np.sum(S**2)
    print("\nVariance explained by each singular value (component):")
    print(variance_explained)

    print("\nCumulative variance explained:")
    print(np.cumsum(variance_explained))

    condition_number = S[0] / S[-1]
    print("\nCondition number:", condition_number)

    #VIF
    print("Variance Inflation Factor (VIF)")

    # Add a constant column for statsmodels
    X_const = pd.DataFrame(x_num_train_scaled.copy())
    X_const["intercept"] = 1

    vif_df = pd.DataFrame()
    vif_df["feature"] = X_const.columns
    vif_df["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

    # Drop intercept row for readability
    vif_df = vif_df[vif_df.feature != "intercept"]

    print("\n===== VARIANCE INFLATION FACTOR (VIF) RESULTS =====")
    print(vif_df)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight="balanced",   # handles imbalance without resampling
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    # Predictions
    y_pred_rf = rf.predict(X_test)

    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

    # Feature importance ranking
    importances = rf.feature_importances_

    feature_names_rf = num_features + cat_features
    print(len(importances))
    print(importances)
    print(len(feature_names_rf))
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names_rf,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\n===== RANDOM FOREST FEATURE IMPORTANCE =====")
    print(feature_importance_df)

    #Use PCA and then do logistic regression


if __name__ == '__main__':
    data_cleaning()
    #feature_engineering()