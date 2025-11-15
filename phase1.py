import pandas as pd
import numpy as np
import re
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt

df = pd.read_csv('atp_matches.csv')

print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nWe are now cleaning the dataset")

print("Sort dataset by match date")
df = df.sort_values(["tourney_date", "match_num"])

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
def clean_score(s):
    if pd.isna(s):
        return ""
    s = s.upper().strip()
    s = re.sub(r"\([^)]*\)", "", s)        # remove parentheses
    s = re.sub(r"[A-Z]+", "", s)           # remove letters like RET, W/O
    s = re.sub(r"\s+", " ", s).strip()     # normalize whitespace
    return s

df["clean_score"] = df["score"].apply(clean_score)

print("Now based off of the clean score we want to compute games difference")
def compute_game_diff(clean_score2):
    if not clean_score2:
        return np.nan
    sets = clean_score2.split()
    p1_games = 0
    p2_games = 0
    for s in sets:
        try:
            g1, g2 = map(int, s.split("-"))
            p1_games += g1
            p2_games += g2
        except ValueError:
            continue
    return p1_games - p2_games

df["game_diff"] = df["clean_score"].apply(compute_game_diff)

print("\nSeeding, entry, rank, and points will be N/A if missing")
seed_entry_rank_cols = ['winner_seed', 'loser_seed',
                        'winner_entry', 'loser_entry',
                         'winner_rank_points',
                         'loser_rank_points']
df[seed_entry_rank_cols] = df[seed_entry_rank_cols].fillna('N/A')

print("\nAll other columns (game statistical measures) will be the mean of the respective feature")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
numeric_cols_to_fill = [col for col in numeric_cols if col not in seed_entry_rank_cols]
df[numeric_cols_to_fill] = df[numeric_cols_to_fill].fillna(df[numeric_cols_to_fill].mean())

missing_after = df.isnull().sum()
print(missing_after)

num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

print(df.info())

print(df.describe())

#Now we are going to feature engineer columns
# --- Basic cleaning ---
df = df.dropna(subset=["winner_id", "loser_id"])

def reorder_by_seed(row2):
    # Determine which player is higher seeded (lower rank number)
    w_rank = row2["winner_rank"]
    l_rank = row2["loser_rank"]

    # Winner is higher-seeded
    if w_rank < l_rank:
        higher = {
            "id": row2["winner_id"],
            "name": row2["winner_name"],
            "hand": row2["winner_hand"],
            "ht": row2["winner_ht"],
            "age": row2["winner_age"],
            "rank": row2["winner_rank"]
        }
        lower = {
            "id": row2["loser_id"],
            "name": row2["loser_name"],
            "hand": row2["loser_hand"],
            "ht": row2["loser_ht"],
            "age": row2["loser_age"],
            "rank": row2["loser_rank"]
        }
        higher_won = 1  # winner is higher seed

    # Loser is higher-seeded
    else:
        higher = {
            "id": row2["loser_id"],
            "name": row2["loser_name"],
            "hand": row2["loser_hand"],
            "ht": row2["loser_ht"],
            "age": row2["loser_age"],
            "rank": row2["loser_rank"]
        }
        lower = {
            "id": row2["winner_id"],
            "name": row2["winner_name"],
            "hand": row2["winner_hand"],
            "ht": row2["winner_ht"],
            "age": row2["winner_age"],
            "rank": row2["winner_rank"]
        }
        higher_won = 0  # lower seed (winner) upset the match

    return pd.Series({
        "higher_id": higher["id"],
        "lower_id": lower["id"],
        "higher_name": higher["name"],
        "lower_name": lower["name"],
        "higher_hand": higher["hand"],
        "lower_hand": lower["hand"],
        "higher_ht": higher["ht"],
        "lower_ht": lower["ht"],
        "higher_age": higher["age"],
        "lower_age": lower["age"],
        "higher_rank": higher["rank"],
        "lower_rank": lower["rank"],
        "log_target": higher_won
    })

# Apply this to entire dataset
reordered = df.apply(reorder_by_seed, axis=1)
df = pd.concat([df, reordered], axis=1)


# --- Ranking difference ---
df["ranking_difference"] =  df["lower_rank"] - df["higher_rank"]
df["height_difference"] =  round(df["lower_ht"] - df["higher_ht"],2)
df["age_difference"] = round(df["lower_age"] - df["higher_age"],2)

hand_map = {"R": 1, "L": 0}
df["higher_hand"] = df["higher_hand"].map(hand_map)
df["lower_hand"] = df["lower_hand"].map(hand_map)

# Your existing H2H and recent-win code can remain,
# but rename player1 → higher_id, player2 → lower_id.


# --- Match type (3-set vs 5-set) ---
df["match_type"] = np.where(df["best_of"] == 5, 1, 0)



# Dictionary to store cumulative head-to-head counts
# Initialize column
df["higher_h2h_win_pct"] = np.nan
h2h_dict = {}

# Loop through matches in order
for idx, row in df.iterrows():
    p_high = row["higher_id"]
    p_low = row["lower_id"]
    p_min, p_max = sorted([p_high, p_low])

    # Use tuple in consistent order
    key = (p_min, p_max)

    # If previous H2H exists → compute percentage BEFORE this match
    if key in h2h_dict:
        wins_min, total = h2h_dict[key]

        if total < 3:
            win_pct_min = .5
        else:
            win_pct_min = wins_min / total if total > 0 else 0.5

        #the higher ranked player also has the first portion of the key
        if p_high == p_min:
            df.at[idx, "higher_h2h_win_pct"] = win_pct_min
        #the higher ranked player is the second part of the key, win % must be reversed
        else:
            df.at[idx, "higher_h2h_win_pct"] = 1 - win_pct_min
    else:
        df.at[idx, "higher_h2h_win_pct"] = 0.5  # neutral before first match

    # Update after match
    if key not in h2h_dict:
        h2h_dict[key] = [0, 0]

    # Update total matches
    h2h_dict[key][1] += 1

    # Update wins for the player that actually won
    winner = row["winner_id"]
    if winner == p_min:
        h2h_dict[key][0] += 1

df["higher_h2h_win_pct"] = df["higher_h2h_win_pct"].fillna(0.5)

df["higher_recent_win_pct"] = np.nan
df["lower_recent_win_pct"] = np.nan

# Dictionary: player_id -> deque of last 5 results (1 = win, 0 = loss)
last5 = defaultdict(lambda: deque(maxlen=5))

for idx, row in df.iterrows():
    high = row["higher_id"]
    low  = row["lower_id"]
    high_win = row["log_target"]      # 1 = higher won, 0 = lower upset

    # Fill feature BEFORE updating today's match
    df.at[idx, "higher_recent_win_pct"] = (
        np.mean(last5[high]) if len(last5[high]) > 0 else 0.5
    )
    df.at[idx, "lower_recent_win_pct"] = (
        np.mean(last5[low]) if len(last5[low]) > 0 else 0.5
    )

    # Update last 5 for both players AFTER match
    last5[high].append(high_win)
    last5[low].append(1 - high_win)

#Now we are creating a metric about a players success on a surface
# Hyperparameters
alpha = 0.2            # EWMA weight (tune between ~0.1 - 0.3)
min_matches = 3        # Minimum required matches to trust estimate
neutral_value = 0.5    # Fallback value

# Dictionary to track exponentially weighted stats
# key: (player_id, surface)   value = {'score': ew_sum, 'weight': ew_weight, 'count': match_count}
ewma_data = defaultdict(lambda: {'score': 0.0, 'weight': 0.0, 'count': 0})

# Initialize output columns
df['higher_surface_ewma'] = np.nan
df['lower_surface_ewma'] = np.nan
for idx, row in df.iterrows():
    surf = row['surface']
    high = row['higher_id']
    low  = row['lower_id']

    h_stats = ewma_data[(high, surf)]
    l_stats = ewma_data[(low, surf)]

    # Compute **prior** EWMA values (before updating with today's match)
    if h_stats['count'] >= min_matches and h_stats['weight'] > 0:
        df.at[idx, 'higher_surface_ewma'] = h_stats['score'] / h_stats['weight']
    else:
        df.at[idx, 'higher_surface_ewma'] = neutral_value

    if l_stats['count'] >= min_matches and l_stats['weight'] > 0:
        df.at[idx, 'lower_surface_ewma'] = l_stats['score'] / l_stats['weight']
    else:
        df.at[idx, 'lower_surface_ewma'] = neutral_value

    # Determine winner and update **AFTER** recording features
    winner = row['winner_id']

    # Update high player record
    h_win = 1 if winner == high else 0
    ewma_data[(high, surf)]['score'] += h_win * alpha
    ewma_data[(high, surf)]['weight'] += alpha
    ewma_data[(high, surf)]['count']  += 1

    # Update low player record
    l_win = 1 if winner == low else 0
    ewma_data[(low, surf)]['score'] += l_win * alpha
    ewma_data[(low, surf)]['weight'] += alpha
    ewma_data[(low, surf)]['count']  += 1



# --- Select simplified features ---
features = df[[
    "higher_id",
    "lower_id",
    "higher_name",
    "lower_name",
    "ranking_difference",
    "height_difference",
    "age_difference",
    "higher_hand",
    "lower_hand",
    "higher_h2h_win_pct",
    "match_type", 
    "higher_recent_win_pct",
    "lower_recent_win_pct",
    "surface",
    "higher_surface_ewma",
    "lower_surface_ewma"
]]

log_target = df["log_target"]

# --- Step 5: Make target = game difference ---
lin_target = df["game_diff"]

# --- Step 6: Combine + save ---
fe_df = features.copy()
fe_df["lin_target"] = lin_target
fe_df["log_target"] = log_target
fe_df.to_csv("fe_matches.csv", index=False)
print(f"Saved fe_matches.csv — shape: {fe_df.shape}")

# --- 1. Load dataset for LDA and PCA ---
df = pd.read_csv("fe_matches.csv")  # assuming you've saved it with all features


# --- 3. Select features ---
features = [
    "ranking_difference",
    "height_difference",
    "age_difference",
    "higher_hand",
    "lower_hand",
    "higher_h2h_win_pct",
    "match_type",
    "higher_recent_win_pct",
    "lower_recent_win_pct",
    "higher_surface_ewma",
    "lower_surface_ewma"
]

X = df[features]
y = df["log_target"]

scaler = StandardScaler()

# --- 4. Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. Standardize numeric features ---
num_features = [
    "ranking_difference",
    "height_difference",
    "age_difference",
    "higher_h2h_win_pct",
    "higher_surface_ewma",
    "lower_surface_ewma"
]

x_train_scaled = scaler.fit_transform(X_train[num_features])
X_test__scaled = scaler.transform(X_test[num_features])

cov_matrix = np.cov(x_train_scaled, rowvar=False)
print("This is the covariance matrix")
print(cov_matrix)

cat_features = ["higher_hand", "lower_hand", "match_type"]

# Concatenate categorical features without scaling
X_train_processed = np.hstack([x_train_scaled, X_train[cat_features].values])
X_test_processed = np.hstack([X_test__scaled, X_test[cat_features].values])

# --- 6. Apply PCA ---
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_processed)
X_test_pca = pca.transform(X_test_processed)

print("Explained variance by PCA components:", pca.explained_variance_ratio_)

# --- 7. Apply LDA ---
print("Starting LDA")
# Create LDA object
lda = LinearDiscriminantAnalysis(n_components=1)  # For binary target, max n_components = 1
# Fit LDA on training data
X_train_lda = lda.fit_transform(X_train_processed, y_train)

# Transform test data
X_test_lda = lda.transform(X_test_processed)

print("Shape of LDA-transformed training set:", X_train_lda.shape)
print("Explained variance ratio (discriminative power):", lda.explained_variance_ratio_)

