import pandas as pd
import numpy as np
import re
from collections import defaultdict, deque
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

df = pd.read_csv('atp_matches.csv')

print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nWe are now cleaning the dataset")

print("Sort dataset by match date")
df = df.sort_values(by=["tourney_date", "match_num"]).reset_index(drop=True)

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
print(df[["score", "clean_score"]])

print("Now based off of the clean score we want to compute games difference")
def compute_game_diff(clean_score):
    if not clean_score:
        return np.nan
    sets = clean_score.split()
    p1_games = 0
    p2_games = 0
    for s in sets:
        try:
            g1, g2 = map(int, s.split("-"))
            p1_games += g1
            p2_games += g2
        except:
            continue
    return p1_games - p2_games

df["game_diff"] = df["clean_score"].apply(compute_game_diff)

print("\nSeeding, entry, rank, and points will be N/A if missing")
seed_entry_rank_cols = ['winner_seed', 'loser_seed',
                        'winner_entry', 'loser_entry',
                        'winner_rank', 'winner_rank_points',
                        'loser_rank', 'loser_rank_points']
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

# --- Ranking difference ---
df["ranking_difference"] = df["winner_rank"] - df["loser_rank"]

# --- Height difference ---
df["height_difference"] = df["winner_ht"] - df["loser_ht"]

# --- Age difference ---
df["age_difference"] = df["winner_age"] - df["loser_age"]

# --- Handedness encoding ---
hand_map = {"R": 1, "L": 0}
df["player1_hand"] = df["winner_hand"].map(hand_map)
df["player2_hand"] = df["loser_hand"].map(hand_map)

# --- Match type (3-set vs 5-set) ---
df["match_type"] = np.where(df["best_of"] == 5, 1, 0)

# Initialize column
df["player1_h2h_win_pct"] = np.nan

# Dictionary to store cumulative head-to-head counts
# Structure: {(player1_id, player2_id): [p1_wins, total_matches]}
h2h_dict = {}

# Loop through matches in order
for idx, row in df.iterrows():
    p1 = row["winner_id"]
    p2 = row["loser_id"]

    # Use tuple in consistent order
    key = (min(p1, p2), max(p1, p2))

    # Check previous head-to-head stats
    if key in h2h_dict:
        wins, total = h2h_dict[key]
        # Determine player1 perspective
        if p1 < p2:
            h2h = wins / total if total > 0 else np.nan
        else:
            h2h = (total - wins) / total if total > 0 else np.nan
        df.at[idx, "player1_h2h_win_pct"] = h2h
    else:
        df.at[idx, "player1_h2h_win_pct"] = np.nan  # first match

    # Update cumulative counts
    if key not in h2h_dict:
        h2h_dict[key] = [0, 0]
    # Increment winner count
    if p1 < p2:
        h2h_dict[key][0] += 1  # p1 wins
    else:
        h2h_dict[key][0] += 0  # winner is "player2" from dictionary perspective
    h2h_dict[key][1] += 1  # total matches

df["player1_h2h_win_pct"] = df["player1_h2h_win_pct"].fillna(0.5)

#Recent win percentage
df["player1_recent_win_pct"] = np.nan
df["player2_recent_win_pct"] = np.nan

last5_dict = defaultdict(lambda: deque(maxlen=5))

for idx, row in df.iterrows():
    p1 = row["winner_id"]
    p2 = row["loser_id"]

    # Player 1 perspective (winner)
    if p1 in last5_dict and len(last5_dict[p1]) > 0:
        df.at[idx, "player1_recent_win_pct"] = np.mean(last5_dict[p1])
    else:
        df.at[idx, "player1_recent_win_pct"] = 0.5  # neutral if no history

    # Player 2 perspective (loser)
    if p2 in last5_dict and len(last5_dict[p2]) > 0:
        df.at[idx, "player2_recent_win_pct"] = np.mean(last5_dict[p2])
    else:
        df.at[idx, "player2_recent_win_pct"] = 0.5  # neutral if no history

    # Update last 5 results
    last5_dict[p1].append(1)  # winner won
    last5_dict[p2].append(0)  # loser lost

# --- Select simplified features ---
features = df[[
    "winner_name", 
    "loser_name",
    "ranking_difference",
    "height_difference",
    "age_difference",
    "player1_hand",
    "player2_hand",
    "player1_h2h_win_pct",
    "match_type", 
    "player1_recent_win_pct",
    "player2_recent_win_pct"
]]

# --- Step 5: Make target = game difference ---
target = df["game_diff"]

# --- Step 6: Combine + save ---
fe_df = features.copy()
fe_df["target"] = target

fe_df.to_csv("fe_matches.csv", index=False)
print(f"Saved fe_matches.csv — shape: {fe_df.shape}")

# --- 1. Load dataset for LDA and PCA ---
df = pd.read_csv("fe_matches.csv")  # assuming you've saved it with all features

# --- 2. Create target variable ---
# Binary: did player 1 win? (1 if winner_name == player1, else 0)
# Assuming player1 is the winner in your feature-engineered df:
df["target"] = 1  # or you can check if player1 is winner

# --- 3. Select features ---
features = [
    "ranking_difference",
    "height_difference",
    "age_difference",
    "player1_hand",
    "player2_hand",
    "player1_h2h_win_pct",
    "match_type",
    "player1_recent_win_pct",
    "player2_recent_win_pct"
]

X = df[features]
y = df["target"]

scaler = StandardScaler()



# --- 4. Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. Standardize numeric features ---
num_features = [
    "ranking_difference",
    "height_difference",
    "age_difference",
    "player1_h2h_win_pct",
    "player1_recent_win_pct",
    "player2_recent_win_pct"
]

x_scaled = scaler.fit_transform(X[num_features])


cov_matrix = np.cov(x_scaled, rowvar=False)
print("This is the covariance matrix")
print(cov_matrix)

cat_features = ["player1_hand", "player2_hand", "match_type"]


X_train_num = scaler.fit_transform(X_train[num_features])
X_test_num = scaler.transform(X_test[num_features])

# Concatenate categorical features without scaling
X_train_processed = np.hstack([X_train_num, X_train[cat_features].values])
X_test_processed = np.hstack([X_test_num, X_test[cat_features].values])

# --- 6. Apply PCA ---
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_processed)
X_test_pca = pca.transform(X_test_processed)

print("Explained variance by PCA components:", pca.explained_variance_ratio_)

# --- 7. Apply LDA ---
print("Starting LDA")