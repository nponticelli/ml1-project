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

    # --- Hyperparams (tune as needed) ---
    BASE_ELO = 1500
    K_global = 32  # how fast global ELO moves
    K_surface = 24  # how fast surface ELO moves (usually <= K_global)
    min_surface_matches = 3  # fall back to global ELO until this many surface matches exist
    surface_weight = 0.6  # combined = surface_weight * surface_elo + (1-surface_weight) * global_elo

    # helpers
    def expected(h, l):
        return 1.0 / (1.0 + 10 ** ((l - h) / 400.0))

    # initialize dictionaries
    global_elo = defaultdict(lambda: BASE_ELO)  # player -> global elo
    surface_elo = defaultdict(lambda: BASE_ELO)  # (player, surface) -> surface elo
    surface_count = defaultdict(int)  # (player, surface) -> count of prior matches on surface

    # create columns to store pre-match ratings and diffs
    df["higher_global_elo"] = np.nan
    df["lower_global_elo"] = np.nan
    df["global_elo_diff"] = np.nan

    df["higher_surface_elo"] = np.nan
    df["lower_surface_elo"] = np.nan
    df["surface_elo_diff"] = np.nan

    df["higher_combined_elo"] = np.nan
    df["lower_combined_elo"] = np.nan
    df["combined_elo_diff"] = np.nan

    # IMPORTANT: iterate in chronological order (use pseudo_date if that's your chronological column)
    # df must be sorted prior to this loop:
    # df = df.sort_values("pseudo_date").reset_index(drop=True)

    for idx, row in df.iterrows():
        hi = row["higher_id"]
        lo = row["lower_id"]
        surf = row.get("surface", None)  # surface string, e.g. 'Hard', 'Clay', etc.
        outcome = row["log_target"]  # 1 if higher won, 0 otherwise

        # --- retrieve pre-match global ELOs (these are what go into features) ---
        h_global = global_elo[hi]
        l_global = global_elo[lo]

        # --- retrieve pre-match surface ELOs ---
        # Use surface_count to decide whether to trust a dedicated surface ELO.
        # If not enough surface matches, fallback to the player's current global ELO.
        if surf is None or surface_count[(hi, surf)] < min_surface_matches:
            h_surf = h_global
        else:
            h_surf = surface_elo[(hi, surf)]

        if surf is None or surface_count[(lo, surf)] < min_surface_matches:
            l_surf = l_global
        else:
            l_surf = surface_elo[(lo, surf)]

        # --- store the pre-match ratings into dataframe (features) ---
        df.at[idx, "higher_global_elo"] = h_global
        df.at[idx, "lower_global_elo"] = l_global
        df.at[idx, "global_elo_diff"] = h_global - l_global

        df.at[idx, "higher_surface_elo"] = h_surf
        df.at[idx, "lower_surface_elo"] = l_surf
        df.at[idx, "surface_elo_diff"] = h_surf - l_surf

        # combined (weighted) version
        h_combined = surface_weight * h_surf + (1.0 - surface_weight) * h_global
        l_combined = surface_weight * l_surf + (1.0 - surface_weight) * l_global
        df.at[idx, "higher_combined_elo"] = h_combined
        df.at[idx, "lower_combined_elo"] = l_combined
        df.at[idx, "combined_elo_diff"] = h_combined - l_combined

        # --- now update the ratings (use outcome and expected probabilities) ---
        # update global ELO
        exp_h_global = expected(h_global, l_global)
        new_h_global = h_global + K_global * (outcome - exp_h_global)
        new_l_global = l_global + K_global * ((1 - outcome) - (1 - exp_h_global))

        global_elo[hi] = new_h_global
        global_elo[lo] = new_l_global

        # update surface ELOs (use separate expected on surface ratings)
        # If the (player,surface) entry wasn't present, surface_elo initialized to BASE_ELO but we
        # intentionally used global as pre-match value; still update surface_elo dict so it starts tracking.
        exp_h_surf = expected(h_surf, l_surf)
        new_h_surf = h_surf + K_surface * (outcome - exp_h_surf)
        new_l_surf = l_surf + K_surface * ((1 - outcome) - (1 - exp_h_surf))

        surface_elo[(hi, surf)] = new_h_surf
        surface_elo[(lo, surf)] = new_l_surf

        # increment surface counts AFTER using them for pre-match decision
        surface_count[(hi, surf)] += 1
        surface_count[(lo, surf)] += 1

    #Now to we have calculated elo, we can calculate volatility


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

    #compute fatigue difference
    df["fatigue_diff"] = df["higher_short_fatigue"] - df["lower_short_fatigue"]

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

    # ------------------------------------------------------------
    # 2. GRAND SLAM INDICATOR
    # ------------------------------------------------------------
    grand_slams = ["Australian Open", "Roland Garros", "Wimbledon", "US Open"]
    df["is_grand_slam"] = df["tourney_name"].isin(grand_slams).astype(int)

    # Final dataset
    fe = df[[
        "tourney_prefix",
        "is_grand_slam",
        "surface",
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
        "higher_global_elo",
        "lower_global_elo",
        "global_elo_diff",
        "higher_surface_elo",
        "lower_surface_elo",
        "surface_elo_diff",
        "higher_short_fatigue",
        "lower_short_fatigue",
        "fatigue_diff",
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
        "lower_surface_elo",
        "surface_elo_diff",
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
    IQR_cap = ["surface_elo_diff", "higher_short_fatigue", "lower_short_fatigue"]

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
    pca = PCA(n_components=len(FEATURES_NUM))
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