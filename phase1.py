import pandas as pd
import numpy as np
import re

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
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

    #Calculate the exact date of the tournament to better calculate fatigue

    # --- pseudo_date: add 1 day per extra match (2 days for GS) ---
    grand_slam_names = {"Wimbledon", "Roland Garros", "Australian Open", "US Open"}

    # ensure tourney_date is datetime already
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')

    # initialize pseudo_date and tournament-count tracker
    df['pseudo_date'] = df['tourney_date']
    tourney_dict = defaultdict(lambda: defaultdict(int))  # tourney_id -> (player_id -> count)

    # iterate in tournament / date order so counts are prior-appearance counts
    for idx, row in df.sort_values(['tourney_date', 'match_num']).iterrows():
        tid = row['tourney_id']
        winner = row['winner_id']
        loser = row['loser_id']

        # defensively get tourney_name (empty string if missing)
        tourney_name = row.get('tourney_name', "") if isinstance(row, dict) else row.get('tourney_name',
                                                                                         row.get('tourney_name', ""))

        # determine per-prior-match increment (2 days for GS, else 1 day)
        is_gs = str(tourney_name) in grand_slam_names
        day_increment = 2 if is_gs else 1

        # how many times each player has appeared earlier in this tournament
        count_w = tourney_dict[tid].get(winner, 0)
        count_l = tourney_dict[tid].get(loser, 0)
        max_prior = max(count_w, count_l)

        # assign pseudo_date using prior-appearance count
        df.at[idx, 'pseudo_date'] = row['tourney_date'] + pd.Timedelta(days=day_increment * max_prior)

        # now increment the appearance counts (so next match for the same player is considered "prior")
        tourney_dict[tid][winner] = max_prior + 1
        tourney_dict[tid][loser] = max_prior + 1

    # finalize pseudo_date dtype and re-sort dataset chronologically by pseudo_date
    df['pseudo_date'] = pd.to_datetime(df['pseudo_date'], errors='coerce')
    df = df.sort_values('pseudo_date').reset_index(drop=True)

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

    df['tourney_prefix'] = df['tourney_id'].astype(str).str[:4] + "_" + df['tourney_name']

    # Final dataset
    fe = df[[
        "tourney_prefix",
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
def feature_engineering_option_c():

    # ---------------------------------------
    # 1. Load dataset
    # ---------------------------------------
    df = pd.read_csv("fe_simplified.csv")
    df = df.sort_values("pseudo_date").reset_index(drop=True)

    # Drop first 3000 rows for warm-up period
    df = df.iloc[3000:].reset_index(drop=True)
    print("After warm-up drop:", df.shape)

    # ---------------------------------------
    # 2. Select Features
    # ---------------------------------------
    # Keep only the features you defined
    FEATURES_NUM = [
        "higher_elo",
        "lower_elo",
        "elo_difference",
        "higher_surface_ewma",
        "lower_surface_ewma",
        "higher_short_fatigue",
        "lower_short_fatigue"
    ]

    FEATURES_CAT = ["age_advantage"]  # 3-category feature

    TARGET = "log_target"

    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[TARGET].copy()

    print("Selected feature matrix shape:", X.shape)

    # ---------------------------------------
    # 3. Handle Outliers (IQR Winsorization)
    # ---------------------------------------
    IQR_cap = ["higher_elo", "lower_elo", "elo_difference"]

    for col in IQR_cap:
        Q1, Q3 = X[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X[col] = X[col].clip(lower, upper)

    print("Finished outlier winsorization.")

    # ---------------------------------------
    # 4. Encode categorical features
    # ---------------------------------------
    # Three outcomes → OneHotEncode to avoid ordinality
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X[FEATURES_CAT])
    cat_feature_names = encoder.get_feature_names_out(FEATURES_CAT)

    # ---------------------------------------
    # 5. Scale numerical features
    # ---------------------------------------
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X[FEATURES_NUM])

    # ---------------------------------------
    # 6. Combine into full feature matrix
    # ---------------------------------------
    X_full = np.hstack([X_num_scaled, X_cat])
    full_feature_list = FEATURES_NUM + list(cat_feature_names)
    print("Final feature matrix:", X_full.shape)

    # ---------------------------------------
    # 7. Covariance matrix (numerical only)
    # ---------------------------------------
    cov_matrix = np.cov(X_num_scaled, rowvar=False)
    print("\nCovariance Matrix (Numerical Features Only):")
    print(pd.DataFrame(cov_matrix, index=FEATURES_NUM, columns=FEATURES_NUM))

    # ---------------------------------------
    # 8. Pearson correlation matrix
    # ---------------------------------------
    corr_matrix = pd.DataFrame(X_num_scaled, columns=FEATURES_NUM).corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Correlation — Numerical Features")
    plt.show()

    # ---------------------------------------
    # 9. Chronological train/test split
    # ---------------------------------------
    cutoff = int(len(X_full) * 0.8)
    X_train, X_test = X_full[:cutoff], X_full[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    print("Train:", X_train.shape, " Test:", X_test.shape)

    # ---------------------------------------
    # 10. PCA (numerical only)
    # ---------------------------------------
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_num_scaled)
    print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)

    # ---------------------------------------
    # 11. LDA (full feature set)
    # ---------------------------------------
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X_full, y)

    print("LDA explained variance:", lda.explained_variance_ratio_)

    plt.figure(figsize=(8,4))
    plt.hist(X_lda[y == 1], alpha=.5, label="Higher seed wins (1)")
    plt.hist(X_lda[y == 0], alpha=.5, label="Lower seed wins (0)")
    plt.legend()
    plt.title("LDA projection (LD1)")
    plt.show()

    # ---------------------------------------
    # 12. Random Forest classifier
    # ---------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # ---------------------------------------
    # 13. Feature importance
    # ---------------------------------------
    importance = rf.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": full_feature_list,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    print("\nRandom Forest Feature Importances:")
    print(fi_df)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "rf": rf,
        "scaler": scaler,
        "encoder": encoder,
        "feature_names": full_feature_list
    }



if __name__ == '__main__':
    #data_cleaning()
    feature_engineering_option_c()