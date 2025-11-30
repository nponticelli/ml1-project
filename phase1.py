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
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------
# Master Cleaner
# ---------------------------------------
def load_raw_data():
    df = pd.read_csv("atp_matches.csv")

    # convert ranking to numeric early
    for col in ["winner_rank", "loser_rank"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].max())

    # fill surfaces and handedness
    df["surface"] = df["surface"].fillna(df["surface"].mode()[0])
    for col in ["winner_hand", "loser_hand"]:
        mode_val = df[col].mode()[0]
        df[col] = df[col].apply(lambda x: x if x in ["R", "L"] else mode_val)

    # seed/entry/points missing → "N/A"
    seed_cols = [
        'winner_seed', 'loser_seed',
        'winner_entry', 'loser_entry',
        'winner_rank_points', 'loser_rank_points'
    ]
    df[seed_cols] = df[seed_cols].fillna("N/A")

    # date format
    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"], format="%Y%m%d", errors="coerce"
    )

    return df

def clean_basic_fields(df):
    """Standardizes core columns: converts dtypes, cleans hand, surface,
    seed formats, ranking fields, match times, and score strings."""

    # --- Date and numeric casting ---
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df["tourney_id"] = df["tourney_id"].astype(str)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["winner_ht"] = df["winner_ht"].fillna(df["winner_ht"].median())
    df["loser_ht"] = df["loser_ht"].fillna(df["loser_ht"].median())

    # --- Handedness normalization: R, L, U (Unknown), N (Unknown?) ---
    df["winner_hand"] = df["winner_hand"].fillna("U").replace("N", "U")
    df["loser_hand"] = df["loser_hand"].fillna("U").replace("N", "U")

    # --- Surface cleaning ---
    df["surface"] = df["surface"].fillna("Unknown")

    # --- Seed and ranking normalizations ---
    df["winner_seed"] = df["winner_seed"].fillna(-1)
    df["loser_seed"] = df["loser_seed"].fillna(-1)
    df["winner_rank"] = df["winner_rank"].fillna(df["winner_rank"].max())
    df["loser_rank"] = df["loser_rank"].fillna(df["loser_rank"].max())
    df["loser_age"] = df["loser_age"].fillna(df["loser_age"].median())
    df["winner_age"] = df["winner_age"].fillna(df["winner_age"].median())


    # --- Clean awkward formats like "34/45" or "45, 3" ---
    for col in ["player_height", "winner_ht", "loser_ht"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col]
                .astype(str)
                .str.replace(",", "")
                .str.replace("/", "")
                .str.extract(r"(\d+)", expand=False),
                errors="coerce",
            )

    return df

def clean_score_fields(df):

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

    return df

def create_rank_order_features(df):

    def reorder(row):
        if row["winner_rank"] < row["loser_rank"]:
            return pd.Series({
                "higher_id": row["winner_id"],
                "higher_rank": row["winner_rank"],
                "higher_name": row["winner_name"],
                "higher_height": row["winner_ht"],
                "higher_ace": row["w_ace"],
                "higher_df": row["w_df"],
                "higher_svpt": row["w_svpt"],
                "higher_1stIn": row["w_1stIn"],
                "higher_1stWon": row["w_1stWon"],
                "higher_2ndWon": row["w_2ndWon"],
                "higher_SvGms": row["w_SvGms"],
                "higher_bpSaved": row["w_bpSaved"],
                "higher_bpFaced": row["w_bpFaced"],
                "lower_id": row["loser_id"],
                "lower_rank": row["loser_rank"],
                "lower_name": row["loser_name"],
                "lower_height": row["loser_ht"],
                "lower_ace": row["l_ace"],
                "lower_df": row["l_df"],
                "lower_svpt": row["l_svpt"],
                "lower_1stIn": row["l_1stIn"],
                "lower_1stWon": row["l_1stWon"],
                "lower_2ndWon": row["l_2ndWon"],
                "lower_SvGms": row["l_SvGms"],
                "lower_bpSaved": row["l_bpSaved"],
                "lower_bpFaced": row["l_bpFaced"],
                "higher_age": row["winner_age"],
                "lower_age": row["loser_age"],
                "log_target": 1
                })
        else:
            return pd.Series({
                "higher_id": row["loser_id"],
                "higher_rank": row["loser_rank"],
                "higher_name": row["loser_name"],
                "higher_height": row["loser_ht"],
                "higher_ace": row["l_ace"],
                "higher_df": row["l_df"],
                "higher_svpt": row["l_svpt"],
                "higher_1stIn": row["l_1stIn"],
                "higher_1stWon": row["l_1stWon"],
                "higher_2ndWon": row["l_2ndWon"],
                "higher_SvGms": row["l_SvGms"],
                "higher_bpSaved": row["l_bpSaved"],
                "higher_bpFaced": row["l_bpFaced"],
                "lower_id": row["winner_id"],
                "lower_rank": row["winner_rank"],
                "lower_name": row["winner_name"],
                "lower_height": row["winner_ht"],
                "lower_ace": row["w_ace"],
                "lower_df": row["w_df"],
                "lower_svpt": row["w_svpt"],
                "lower_1stIn": row["w_1stIn"],
                "lower_1stWon": row["w_1stWon"],
                "lower_2ndWon": row["w_2ndWon"],
                "lower_SvGms": row["w_SvGms"],
                "lower_bpSaved": row["w_bpSaved"],
                "lower_bpFaced": row["w_bpFaced"],
                "higher_age": row["loser_age"],
                "lower_age": row["winner_age"],
                "log_target": 0
            })

    df = pd.concat([df, df.apply(reorder, axis=1)], axis=1)
    return df

def compute_elo_features(df):

    BASE_ELO = 1500
    K_global = 32
    K_surface = 24
    min_surface_matches = 3
    surface_weight = 0.7

    df = df.sort_values(["tourney_date", "match_num"])

    global_elo = defaultdict(lambda: BASE_ELO)
    surface_elo = defaultdict(lambda: BASE_ELO)
    surface_count = defaultdict(int)

    # preallocate columns
    for col in [
        "higher_global_elo", "lower_global_elo", "global_elo_diff",
        "higher_surface_elo", "lower_surface_elo", "surface_elo_diff",
        "higher_combined_elo", "lower_combined_elo", "combined_elo_diff"
    ]:
        df[col] = np.nan

    def expected(h, l):
        return 1 / (1 + 10 ** ((l - h) / 400))

    for idx, row in df.iterrows():

        hi, lo, surf = row["higher_id"], row["lower_id"], row["surface"]
        outcome = row["log_target"]

        h_global, l_global = global_elo[hi], global_elo[lo]

        # surface-level ELO (fallback logic)
        h_surf = surface_elo[(hi, surf)] if surface_count[(hi, surf)] >= min_surface_matches else h_global
        l_surf = surface_elo[(lo, surf)] if surface_count[(lo, surf)] >= min_surface_matches else l_global

        # store pre-match values
        df.at[idx, "higher_global_elo"] = h_global
        df.at[idx, "lower_global_elo"] = l_global
        df.at[idx, "global_elo_diff"] = h_global - l_global

        df.at[idx, "higher_surface_elo"] = h_surf
        df.at[idx, "lower_surface_elo"] = l_surf
        df.at[idx, "surface_elo_diff"] = h_surf - l_surf

        # combined
        h_comb = surface_weight * h_surf + (1 - surface_weight) * h_global
        l_comb = surface_weight * l_surf + (1 - surface_weight) * l_global
        df.at[idx, "higher_combined_elo"] = h_comb
        df.at[idx, "lower_combined_elo"] = l_comb
        df.at[idx, "combined_elo_diff"] = h_comb - l_comb

        # update ELOs
        exp_global = expected(h_global, l_global)
        global_elo[hi] += K_global * (outcome - exp_global)
        global_elo[lo] += K_global * ((1 - outcome) - (1 - exp_global))

        exp_surf = expected(h_surf, l_surf)
        surface_elo[(hi, surf)] = h_surf + K_surface * (outcome - exp_surf)
        surface_elo[(lo, surf)] = l_surf + K_surface * ((1 - outcome) - (1 - exp_surf))

        surface_count[(hi, surf)] += 1
        surface_count[(lo, surf)] += 1

    return df

def compute_pseudo_dates(df):
    grand_slams = {"Wimbledon", "Roland Garros", "Australian Open", "US Open"}

    df['pseudo_date'] = df['tourney_date']
    tourney_dict = defaultdict(lambda: defaultdict(int))

    for idx, row in df.sort_values(['tourney_date', 'match_num']).iterrows():

        tid = row['tourney_id']
        winner, loser = row['winner_id'], row['loser_id']

        is_gs = str(row['tourney_name']) in grand_slams
        inc = 2 if is_gs else 1

        c_w = tourney_dict[tid].get(winner, 0)
        c_l = tourney_dict[tid].get(loser, 0)
        prior = max(c_w, c_l)

        df.at[idx, 'pseudo_date'] = row['tourney_date'] + pd.Timedelta(days=inc * prior)

        tourney_dict[tid][winner] = prior + 1
        tourney_dict[tid][loser] = prior + 1

    df = df.sort_values('pseudo_date').reset_index(drop=True)
    return df

def compute_short_term_fatigue(df):
    window = pd.Timedelta(days=10)
    df["minutes"] = df.get("minutes", 60)

    df["higher_short_fatigue"] = 0.0
    df["lower_short_fatigue"] = 0.0

    hist = defaultdict(list)

    for idx, row in df.iterrows():
        date = row["pseudo_date"]

        def fatigue(p):
            return sum(m for d, m in hist[p] if d >= date - window)

        hi, lo = row["higher_id"], row["lower_id"]
        df.at[idx, "higher_short_fatigue"] = fatigue(hi)
        df.at[idx, "lower_short_fatigue"] = fatigue(lo)

        hist[hi].append((date, row["minutes"]))
        hist[lo].append((date, row["minutes"]))

    df["higher_short_fatigue"] = df["higher_short_fatigue"].fillna(0)
    df["lower_short_fatigue"] = df["lower_short_fatigue"].fillna(0)
    df["fatigue_diff"] = df["higher_short_fatigue"] - df["lower_short_fatigue"]
    return df

def compute_age_features(df):
    all_ages = pd.concat([df["winner_age"], df["loser_age"]])
    mean_age, std_age = all_ages.mean(), all_ages.std()

    df["higher_dev"] = abs(df["higher_age"] - mean_age)
    df["lower_dev"] = abs(df["lower_age"] - mean_age)

    df["age_dev_diff"] = df["lower_dev"] - df["higher_dev"]

    df["age_diff_z"] = (df["age_dev_diff"] - df["age_dev_diff"].mean()) / df["age_dev_diff"].std()

    def age_adv(row):
        if row["age_dev_diff"] > std_age:
            return "higher_seed_age_advantage"
        elif row["age_dev_diff"] < -std_age:
            return "lower_seed_age_advantage"
        else:
            return "no_age_advantage"

    df["age_advantage"] = df.apply(age_adv, axis=1)
    return df

def compute_height_features(df):
    df['height_diff'] = df['higher_height'] - df['lower_height']
    return df

def create_tournament_features(df):

    df['tourney_prefix'] = df['tourney_id'].astype(str).str[:4] + "_" + df['tourney_name']

    grand_slams = ["Australian Open", "Roland Garros", "Wimbledon", "US Open"]
    df["is_grand_slam"] = df["tourney_name"].isin(grand_slams).astype(int)

    return df

def compute_service_stats(df, window=5):
    """
    Compute higher_service_advantage, lower_service_advantage,
    and weighted first serve percentage over last N matches for each player.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
            - higher_id, lower_id
            - higher_svpt, higher_1stIn, higher_1stWon, higher_2ndWon
            - lower_svpt, lower_1stIn, lower_1stWon, lower_2ndWon
            - pseudo_date (datetime)
    window : int
        Number of past matches to use for rolling averages.

    Returns
    -------
    df : pd.DataFrame
        Adds the following columns:
            - higher_service_advantage
            - lower_service_advantage
            - service_advantage_diff
            - higher_first_serve_pct
            - lower_first_serve_pct
    """

    service_hist = defaultdict(list)
    return_hist = defaultdict(list)
    first_serve_hist = defaultdict(list)

    higher_adv = []
    lower_adv = []
    higher_first_pct = []
    lower_first_pct = []

    # Ensure chronological order
    df = df.sort_values('pseudo_date').reset_index(drop=True)

    def weighted_pct(hist, player_id):
        """Weighted rolling percentage with linear weights."""
        last_matches = hist[player_id][-window:]
        if not last_matches:
            return 0.5
        total_won = 0
        total_pts = 0
        for i, (won, pts) in enumerate(reversed(last_matches), 1):
            weight = i
            total_won += won * weight
            total_pts += pts * weight
        return total_won / total_pts if total_pts > 0 else 0.5

    def weighted_first_serve_pct(hist, player_id):
        """Weighted rolling first serve %."""
        last_matches = hist[player_id][-window:]
        if not last_matches:
            return 0.65  # neutral default first serve %
        total_first_in = 0
        total_svpt = 0
        for i, (first_in, svpt) in enumerate(reversed(last_matches), 1):
            weight = i
            total_first_in += first_in * weight
            total_svpt += svpt * weight
        return total_first_in / total_svpt if total_svpt > 0 else 0.65

    for idx, row in df.iterrows():
        hi, lo = row['higher_id'], row['lower_id']

        # Service points
        hi_service_won = row['higher_1stWon'] + row['higher_2ndWon']
        hi_service_total = row['higher_svpt']
        lo_service_won = row['lower_1stWon'] + row['lower_2ndWon']
        lo_service_total = row['lower_svpt']

        # Return points
        hi_return_won = lo_service_total - lo_service_won
        lo_return_won = hi_service_total - hi_service_won

        # First serve
        hi_first_in = row['higher_1stIn']
        lo_first_in = row['lower_1stIn']

        # Compute advantages
        higher_advantage = weighted_pct(service_hist, hi) - weighted_pct(return_hist, lo)
        lower_advantage = weighted_pct(service_hist, lo) - weighted_pct(return_hist, hi)

        higher_adv.append(higher_advantage)
        lower_adv.append(lower_advantage)

        higher_first_pct.append(weighted_first_serve_pct(first_serve_hist, hi))
        lower_first_pct.append(weighted_first_serve_pct(first_serve_hist, lo))

        # Update histories after current match
        service_hist[hi].append((hi_service_won, hi_service_total))
        service_hist[lo].append((lo_service_won, lo_service_total))

        return_hist[hi].append((hi_return_won, lo_service_total))
        return_hist[lo].append((lo_return_won, hi_service_total))

        first_serve_hist[hi].append((hi_first_in, hi_service_total))
        first_serve_hist[lo].append((lo_first_in, lo_service_total))

    df['higher_service_advantage'] = higher_adv
    df['lower_service_advantage'] = lower_adv
    df['service_advantage_diff'] = df['higher_service_advantage'] - df['lower_service_advantage']
    df['higher_first_serve_pct'] = higher_first_pct
    df['lower_first_serve_pct'] = lower_first_pct
    df['first_serve_pct_diff'] = df['higher_first_serve_pct'] - df['lower_first_serve_pct']


    return df

def feature_interactions(df):

    return df

def export_final_dataset(df):

    cols = [
        "tourney_prefix",
        "is_grand_slam",
        "surface",
        "pseudo_date",
        "higher_id", "higher_name", "higher_rank", "higher_age",
        "lower_id", "lower_name", "lower_rank", "lower_age",
        "height_diff",
        "age_advantage",
        "age_diff_z",
        "higher_global_elo", "lower_global_elo", "global_elo_diff",
        "higher_surface_elo", "lower_surface_elo", "surface_elo_diff",
        "higher_combined_elo", "lower_combined_elo", "combined_elo_diff",
        "higher_short_fatigue", "lower_short_fatigue", "fatigue_diff",
        "higher_service_advantage", "higher_first_serve_pct",
        "higher_1stWon", "higher_2ndWon", "higher_svpt",
        "lower_service_advantage", "lower_first_serve_pct",
        "lower_1stWon", "lower_2ndWon","lower_svpt",
        "service_advantage_diff",
        "first_serve_pct_diff",
        "game_diff",
        "log_target"
    ]

    fe = df[cols].copy()
    fe.to_csv("fe_simplified_modular.csv", index=False)

    print("Saved fe_simplified.csv with shape:", fe.shape)
    return fe

def data_cleaning():

    df = load_raw_data()
    df = clean_basic_fields(df)
    df = clean_score_fields(df)
    df = create_rank_order_features(df)
    df = compute_elo_features(df)
    df = compute_pseudo_dates(df)
    df = compute_short_term_fatigue(df)
    df = compute_age_features(df)
    df  = compute_height_features(df)
    df = create_tournament_features(df)
    df = compute_service_stats(df)
    df = feature_interactions(df)
    df = export_final_dataset(df)

    return df




def feature_engineering():

    # ---------------------------------------
    # 1. Load dataset
    # ---------------------------------------
    df = pd.read_csv("fe_simplified_modular.csv")
    df = df.sort_values("pseudo_date").reset_index(drop=True)

    # Drop first 3000 rows for warm-up period
    df = df.iloc[3000:].reset_index(drop=True)
    print("After warm-up drop:", df.shape)

    # ---------------------------------------
    # 2. Select Features
    # ---------------------------------------
    # Keep only the features you defined
    FEATURES_NUM = [
        "lower_combined_elo",
        "combined_elo_diff",
        "fatigue_diff",
        "age_diff_z",
        "lower_service_advantage",
        "service_advantage_diff",

        "first_serve_pct_diff",''
        "height_diff",

    ]

    FEATURES_CAT = ["age_advantage", "is_grand_slam", "surface"]  # 3-category feature

    TARGET = "log_target"

    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[TARGET].copy()

    print("Selected feature matrix shape:", X.shape)

    # ---------------------------------------
    # 3. Handle Outliers (IQR Winsorization)
    # ---------------------------------------
    IQR_cap = ["lower_combined_elo","combined_elo_diff", "fatigue_diff",
        "lower_service_advantage",
        "service_advantage_diff", "age_diff_z", "first_serve_pct_diff", "height_diff",
        ]

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
    # 5a. Singular Value Decomposition (SVD)
    # ---------------------------------------
    U, S, VT = np.linalg.svd(X_num_scaled, full_matrices=False)
    print("\nSVD singular values:", S)
    print("Explained variance ratio by SVD (normalized):", S**2 / np.sum(S**2))

    # ---------------------------------------
    # 5b. Variance Inflation Factor (VIF)
    # ---------------------------------------
    vif_data = pd.DataFrame()
    vif_data["Feature"] = FEATURES_NUM
    vif_data["VIF"] = [variance_inflation_factor(X_num_scaled, i) for i in range(X_num_scaled.shape[1])]
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data)

    # ---------------------------------------
    # 6. Combine into full feature matrix
    # ---------------------------------------
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X[FEATURES_CAT])
    cat_feature_names = encoder.get_feature_names_out(FEATURES_CAT)

    X_full = np.hstack([X_num_scaled, X_cat])
    full_feature_list = FEATURES_NUM + list(cat_feature_names)
    print("Final feature matrix:", X_full.shape)

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
        bootstrap=True,
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
    data_cleaning()
    feature_engineering()