import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from datetime import timedelta
from thefuzz import process, fuzz

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
    df['winner_id'] = df['winner_id'].astype(str)
    df['loser_id'] = df['loser_id'].astype(str)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["winner_ht"] = df["winner_ht"].fillna(df["winner_ht"].median())
    df["loser_ht"] = df["loser_ht"].fillna(df["loser_ht"].median())

    # --- Handedness normalization: R, L, U (Unknown), N (Unknown?) ---
    df["winner_hand"] = df["winner_hand"].fillna("U").replace("N", "U")
    df["loser_hand"] = df["loser_hand"].fillna("U").replace("N", "U")

    # --- Surface cleaning ---
    df["surface"] = df["surface"].fillna(df["surface"].mode().iloc[0])

    # --- Seed and ranking normalizations ---
    df["winner_seed"] = df["winner_seed"].fillna(-1)
    df["loser_seed"] = df["loser_seed"].fillna(-1)
    df["winner_rank"] = df["winner_rank"].fillna(df["winner_rank"].max())
    df["loser_rank"] = df["loser_rank"].fillna(df["loser_rank"].max())
    df["loser_age"] = df["loser_age"].fillna(df["loser_age"].median())
    df["winner_age"] = df["winner_age"].fillna(df["winner_age"].median())

    cols_to_fill = [
        "minutes",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced"
    ]

    # Fill missing values with median for each column
    for col in cols_to_fill:
        df[col] = df[col].fillna(df[col].median())

    df["round"] = df["round"].fillna(df["round"].mode().iloc[0])

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

def add_bets(df):
    df_bet = pd.read_csv("betting_odds.csv")

    # 1. Clean and Filter
    df = df[df['tourney_date'].dt.year >= 2003].copy()
    # This tells pandas: "Try to guess each date individually, but usually assume Month/Day/Year"
    df_bet['bet_match_date'] = pd.to_datetime(df_bet['Date'], format='mixed', dayfirst=False)
    df_bet = df_bet[df_bet['bet_match_date'].dt.year >= 2003].copy()

    # 2. Aggressive Name Cleaning (Strip whitespace!)
    df['winner_name'] = df['winner_name'].str.strip()
    df['loser_name'] = df['loser_name'].str.strip()

    def normalize_betting_names(name):
        """
        Robustly handles multi-part names (De Minaur A.) and multiple initials (Tsonga J.W.).
        Converts 'De Minaur A.' -> 'A De Minaur'
        Converts 'Struff J.L.' -> 'J Struff'
        """
        if pd.isna(name): return name

        # Clean string and split by spaces
        parts = str(name).strip().split(' ')

        if len(parts) >= 2:
            # 1. The last part is ALWAYS the initial(s) in this dataset (e.g., 'A.' or 'J.W.')
            initials_part = parts[-1].replace('.', '')
            # We only need the very first initial for our strict-initial fuzzy filter
            first_init = initials_part[0] if len(initials_part) > 0 else ""

            # 2. Everything before the last part is the Last Name (e.g., ['De', 'Minaur'])
            last_name_parts = parts[:-1]
            last_name = " ".join(last_name_parts).replace('.', '')

            return f"{first_init} {last_name}".strip()

        return str(name).strip()

    df_bet['W_norm'] = df_bet['Winner'].apply(normalize_betting_names)
    df_bet['L_norm'] = df_bet['Loser'].apply(normalize_betting_names)

    # --- Pre-calculate frequency of names in the betting dataset ---
    # This combines normalized Winners and Losers to count total appearances
    all_bet_norms = pd.concat([df_bet['W_norm'], df_bet['L_norm']])
    name_counts = all_bet_norms.value_counts().to_dict()

    # 3. Improved Mapping
    std_players = pd.unique(pd.concat([df['winner_name'], df['loser_name']]))
    bet_players = pd.unique(pd.concat([df_bet['W_norm'], df_bet['L_norm']]))

    name_map = {}
    for b_name in bet_players:
        if pd.isna(b_name) or len(b_name) == 0: continue

        # Get count for the current betting name
        count = name_counts.get(b_name, 0)

        first_init = b_name[0].upper()
        potential_matches = [s for s in std_players if s.upper().startswith(first_init)]

        if potential_matches:
            match, score = process.extractOne(b_name, potential_matches, scorer=fuzz.token_set_ratio)

            if score >= 67:
                name_map[b_name] = match
            else:
                # Included the count in the print statement
                print(f"Low Score [{score}] | Count: {count:3} | Betting: {b_name:20} | Match: {match}")
        else:
            print(f"No Match   | Count: {count:3} | Name: {b_name}")

    df_bet['winner_std_name'] = df_bet['W_norm'].map(name_map)
    df_bet['loser_std_name'] = df_bet['L_norm'].map(name_map)

    # 4. The "Safe" Sort and Merge
    df = df.sort_values('tourney_date').reset_index(drop=True)
    df_bet = df_bet.sort_values('bet_match_date').reset_index(drop=True)

    # If merge_asof is still failing, it's often due to duplicate keys.
    # Let's try a standard merge first on names, then filter by date.
    # This is often more successful in tennis data.
    merged = pd.merge(
        df,
        df_bet[['bet_match_date', 'winner_std_name', 'loser_std_name', 'B365W', 'B365L']],
        left_on=['winner_name', 'loser_name'],
        right_on=['winner_std_name', 'loser_std_name'],
        how='left'
    )

    # 5. Post-Merge Filter: Only keep betting lines within 10 days of the tournament
    # This replaces the logic of merge_asof but is much more "forgiving"
    days_diff = (merged['bet_match_date'] - merged['tourney_date']).dt.days
    merged = merged[(days_diff >= -1) & (days_diff <= 12) | (merged['B365W'].isna())]

    # Remove duplicates that can occur from the broad merge
    merged = merged.sort_values('bet_match_date').drop_duplicates(
        subset=['tourney_id', 'winner_name', 'loser_name'], keep='first'
    )

    # 1. Calculate Implied Probs (No rounding yet for better precision)
    merged['raw_prob_w'] = 1 / merged['B365W']
    merged['raw_prob_l'] = 1 / merged['B365L']

    # 2. Overround (Vig)
    merged['total_book'] = merged['raw_prob_w'] + merged['raw_prob_l']
    merged['bookie_margin'] = merged['total_book'] - 1

    # 3. Normalized "True" Probabilities
    merged['prob_winner_market'] = merged['raw_prob_w'] / merged['total_book']
    merged['prob_loser_market'] = merged['raw_prob_l'] / merged['total_book']

    # 4. Binary Feature: Did the Favorite Win?
    # (In tennis, the favorite has the LOWER odds, so the HIGHER implied prob)
    merged['winner_was_favorite'] = (merged['B365W'] < merged['B365L']).astype(int)

    # 5. Final Rounding (Only at the very end)
    cols_to_round = ['prob_winner_market', 'prob_loser_market', 'bookie_margin']
    merged[cols_to_round] = merged[cols_to_round].round(4)

    return merged

def create_rank_order_features(df, seed = 42):
    def reorder(row):

        rng = np.random.RandomState(seed + row.name)
        val = rng.choice([True, False])
        new_row = {
            "tourney_id": row["tourney_id"],
            "tourney_name": row["tourney_name"],
            "tourney_date": row["tourney_date"],
            "tourney_level": row["tourney_level"],
            "best_of": row["best_of"],
            "round": row["round"],
            "surface": row["surface"],
            "match_num": row["match_num"],
            "minutes": row["minutes"],
            "clean_score": row["clean_score"],
            "bookie_margin": row.get("bookie_margin", np.nan),
            "bet_match_date": row["bet_match_date"],
        }
        if val:
            new_row.update({
                "playerA_id": row["winner_id"],
                "playerA_rank": row["winner_rank"],
                "playerA_name": row["winner_name"],
                "playerA_age": row["winner_age"],
                "playerA_height": row["winner_ht"],
                "playerA_ace": row["w_ace"],
                "playerA_df": row["w_df"],
                "playerA_svpt": row["w_svpt"],
                "playerA_1stIn": row["w_1stIn"],
                "playerA_1stWon": row["w_1stWon"],
                "playerA_2ndWon": row["w_2ndWon"],
                "playerA_SvGms": row["w_SvGms"],
                "playerA_bpSaved": row["w_bpSaved"],
                "playerA_bpFaced": row["w_bpFaced"],
                "playerA_odds": row.get("B365W", np.nan),
                "playerA_market_prob": row.get("prob_winner_market", np.nan),

                "playerB_id": row["loser_id"],
                "playerB_rank": row["loser_rank"],
                "playerB_name": row["loser_name"],
                "playerB_age": row["loser_age"],
                "playerB_height": row["loser_ht"],
                "playerB_ace": row["l_ace"],
                "playerB_df": row["l_df"],
                "playerB_svpt": row["l_svpt"],
                "playerB_1stIn": row["l_1stIn"],
                "playerB_1stWon": row["l_1stWon"],
                "playerB_2ndWon": row["l_2ndWon"],
                "playerB_SvGms": row["l_SvGms"],
                "playerB_bpSaved": row["l_bpSaved"],
                "playerB_bpFaced": row["l_bpFaced"],
                "playerB_odds": row.get("B365L", np.nan),
                "playerB_market_prob": row.get("prob_loser_market", np.nan),
                "log_target": 1,
                "game_diff": row.get("game_diff", 0)
            })

        else:
            new_row.update({
                "playerA_id": row["loser_id"],
                "playerA_rank": row["loser_rank"],
                "playerA_name": row["loser_name"],
                "playerA_age": row["loser_age"],
                "playerA_height": row["loser_ht"],
                "playerA_ace": row["l_ace"],
                "playerA_df": row["l_df"],
                "playerA_svpt": row["l_svpt"],
                "playerA_1stIn": row["l_1stIn"],
                "playerA_1stWon": row["l_1stWon"],
                "playerA_2ndWon": row["l_2ndWon"],
                "playerA_SvGms": row["l_SvGms"],
                "playerA_bpSaved": row["l_bpSaved"],
                "playerA_bpFaced": row["l_bpFaced"],
                "playerA_odds": row.get("B365L", np.nan),
                "playerA_market_prob": row.get("prob_loser_market", np.nan),

                "playerB_id": row["winner_id"],
                "playerB_rank": row["winner_rank"],
                "playerB_name": row["winner_name"],
                "playerB_age": row["winner_age"],
                "playerB_height": row["winner_ht"],
                "playerB_ace": row["w_ace"],
                "playerB_df": row["w_df"],
                "playerB_svpt": row["w_svpt"],
                "playerB_1stIn": row["w_1stIn"],
                "playerB_1stWon": row["w_1stWon"],
                "playerB_2ndWon": row["w_2ndWon"],
                "playerB_SvGms": row["w_SvGms"],
                "playerB_bpSaved": row["w_bpSaved"],
                "playerB_bpFaced": row["w_bpFaced"],
                "playerB_odds": row.get("B365W", np.nan),
                "playerB_market_prob": row.get("prob_winner_market", np.nan),
                "log_target": 0,
                "game_diff": -row.get("game_diff", 0)

            })

        new_row['playerA_points_won'] = new_row['playerA_1stWon'] + new_row['playerA_2ndWon'] + (new_row['playerB_svpt'] - new_row['playerB_1stWon'] - new_row['playerB_2ndWon'])
        new_row['total_points'] = (new_row['playerA_svpt'] + new_row['playerB_svpt'])
        new_row['playerA_points_won_pct'] = round(new_row['playerA_points_won'] / new_row['total_points'],4)
        new_row['playerB_points_won_pct'] = 1 - new_row['playerA_points_won_pct']

        return pd.Series(new_row)
    df_new = df.apply(reorder, axis=1).reset_index(drop=True)
    return df_new

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
        "playerA_global_elo", "playerB_global_elo", "global_elo_diff",
        "playerA_surface_elo", "playerB_surface_elo", "surface_elo_diff",
        "playerA_combined_elo", "playerB_combined_elo", "combined_elo_diff"
    ]:
        df[col] = np.nan

    def expected(h, l):
        return 1 / (1 + 10 ** ((l - h) / 400))

    for idx, row in df.iterrows():

        a, b, surf = row["playerA_id"], row["playerB_id"], row["surface"]
        outcome = row["log_target"]

        playerA_global, playerB_global = global_elo[a], global_elo[b]

        # surface-level ELO (fallback logic)
        playerA_surf = surface_elo[(a, surf)] if surface_count[(a, surf)] >= min_surface_matches else playerA_global
        playerB_surf = surface_elo[(b, surf)] if surface_count[(b, surf)] >= min_surface_matches else playerB_global

        # store pre-match values
        df.at[idx, "playerA_global_elo"] = playerA_global
        df.at[idx, "playerB_global_elo"] = playerB_global
        df.at[idx, "global_elo_diff"] = playerA_global - playerB_global

        df.at[idx, "playerA_surface_elo"] = playerA_surf
        df.at[idx, "playerB_surface_elo"] = playerB_surf
        df.at[idx, "surface_elo_diff"] = playerA_surf - playerB_surf

        # combined
        playerA_comb = surface_weight * playerA_surf + (1 - surface_weight) * playerA_global
        playerB_comb = surface_weight * playerB_surf + (1 - surface_weight) * playerB_global
        df.at[idx, "playerA_combined_elo"] = playerA_comb
        df.at[idx, "playerB_combined_elo"] = playerB_comb
        df.at[idx, "combined_elo_diff"] = playerA_comb - playerB_comb

        # update ELOs
        exp_global = expected(playerA_global, playerB_global)
        global_elo[a] += K_global * (outcome - exp_global)
        global_elo[b] += K_global * ((1 - outcome) - (1 - exp_global))

        exp_surf = expected(playerA_surf, playerB_surf)
        surface_elo[(a, surf)] = playerA_surf + K_surface * (outcome - exp_surf)
        surface_elo[(b, surf)] = playerB_surf + K_surface * ((1 - outcome) - (1 - exp_surf))

        surface_count[(a, surf)] += 1
        surface_count[(b, surf)] += 1

    return df

def compute_dates(df):
    # 1. Initialize with the ground truth (Anchor)
    df['pseudo_date'] = pd.to_datetime(df['bet_match_date'])

    # 2. Fill missing with Round Median (Contextual Backup)
    # This clusters missing rows with their actual peers
    df['pseudo_date'] = df['pseudo_date'].fillna(
        df.groupby(['tourney_id', 'round'])['pseudo_date'].transform('median')
    )

    # 3. Final Fallback: "Last-Played" Rest-Day Logic
    df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
    player_latest = {}

    for idx, row in df.iterrows():
        tid, pA, pB = row['tourney_id'], row['playerA_id'], row['playerB_id']

        if pd.isna(row['pseudo_date']):
            # Use best_of logic (2 days for 5-setters, 1 for 3-setters)
            inc = 2 if row['best_of'] == 5 else 1

            # Look at the most recent actual date these players competed in this tourney
            date_A = player_latest.get((tid, pA))
            date_B = player_latest.get((tid, pB))

            if date_A or date_B:
                # Use the latest of the two previous match dates as the baseline
                base = max(filter(None, [date_A, date_B]))
                new_date = base + pd.Timedelta(days=inc)
            else:
                # First round fallback: use the official tourney start date
                new_date = row['tourney_date']

            df.at[idx, 'pseudo_date'] = new_date

        # CRITICAL: Always update the tracker so the NEXT round knows when these players played
        player_latest[(tid, pA)] = df.at[idx, 'pseudo_date']
        player_latest[(tid, pB)] = df.at[idx, 'pseudo_date']

    # 4. Final Chronological Sort
    df['pseudo_date'] = pd.to_datetime(df['pseudo_date'])
    df = df.sort_values(['pseudo_date', 'match_num']).reset_index(drop=True)
    return df

def compute_fatigue(df):
    # Ensure df sorted by chronological order
    df = df.sort_values("pseudo_date").reset_index(drop=True)

    # Preallocate output columns
    for col in [
        "playerA_last_minutes", "playerB_last_minutes",
        "playerA_fatigue_10d", "playerB_fatigue_10d",
        "playerA_year_fatigue", "playerB_year_fatigue",
        "playerA_rusty", "playerB_rusty",
        "last_minutes_diff", "fatigue_10d_diff", "year_fatigue_diff",
        "rusty_diff"
    ]:
        df[col] = 0.0

    # History per player: list of (date, minutes)
    match_history = defaultdict(list)

    THREE_DAYS = pd.Timedelta(days=3)
    TEN_DAYS = pd.Timedelta(days=10)
    SIXTY_DAYS = pd.Timedelta(days=60)

    for idx, row in df.iterrows():
        date = row["pseudo_date"]
        year = str(row["tourney_id"])[:4]

        pA = row["playerA_id"]
        pB = row["playerB_id"]

        # --- LAST MATCH MINUTES (ONLY IF LAST MATCH WITHIN 3 DAYS) ---
        def get_last_minutes(player):
            hist = match_history[player]
            if not hist:
                return 0
            last_date, last_min = hist[-1]
            if last_date < date and last_date >= date - THREE_DAYS:
                return last_min
            return 0

        df.at[idx, "playerA_last_minutes"] = get_last_minutes(pA)
        df.at[idx, "playerB_last_minutes"] = get_last_minutes(pB)

        # --- FATIGUE LAST 10 DAYS ---
        def get_fatigue_10d(player):
            return sum(
                minutes for d, minutes in match_history[player]
                if (date - TEN_DAYS) <= d < date
            )

        df.at[idx, "playerA_fatigue_10d"] = get_fatigue_10d(pA)
        df.at[idx, "playerB_fatigue_10d"] = get_fatigue_10d(pB)

        # --- YEAR FATIGUE ---
        def get_year_fatigue(player):
            return sum(
                minutes for d, minutes in match_history[player]
                if str(d.year) == year and d < date
            )

        df.at[idx, "playerA_year_fatigue"] = get_year_fatigue(pA)
        df.at[idx, "playerB_year_fatigue"] = get_year_fatigue(pB)

        # --- RUSTINESS (no match in over 30 days) ---
        def get_rustiness(player):
            hist = match_history[player]
            if not hist:
                return 1  # no prior matches means very rusty
            last_date, _ = hist[-1]
            return 1 if last_date < date - SIXTY_DAYS else 0

        df.at[idx, "playerA_rusty"] = get_rustiness(pA)
        df.at[idx, "playerB_rusty"] = get_rustiness(pB)

        # --- UPDATE HISTORY AFTER calculations ---
        match_history[pA].append((date, row["minutes"]))
        match_history[pB].append((date, row["minutes"]))

    # Differences
    df["last_minutes_diff"] = df["playerA_last_minutes"] - df["playerB_last_minutes"]
    df["fatigue_10d_diff"] = df["playerA_fatigue_10d"] - df["playerB_fatigue_10d"]
    df["year_fatigue_diff"] = df["playerA_year_fatigue"] - df["playerB_year_fatigue"]
    df["rusty_diff"] = df["playerA_rusty"] - df["playerB_rusty"]

    return df

def compute_age_features(df):
    all_ages = pd.concat([df["playerA_age"], df["playerB_age"]])
    mean_age, std_age = all_ages.mean(), all_ages.std()

    df["playerA_prime_age"] = abs(df["playerA_age"] - mean_age)
    df["playerB_prime_age"] = abs(df["playerB_age"] - mean_age)
    df["prime_age_diff"] = df["playerA_prime_age"] - df["playerB_prime_age"]

    #Linear difference
    df["raw_age_diff"] = df["playerA_age"] - df["playerB_age"]

    # 3. NON-LINEAR FEATURES (The "71% Push")
    # Squared difference: Penalizes large age gaps (e.g., Veteran vs. Young Gun)
    df["raw_age_diff_sq"] = np.sign(df["raw_age_diff"]) * (df["raw_age_diff"] ** 2)

    # 4. Age Interaction with Fatigue
    # Older players (Age > 32) are more impacted by 10-day fatigue
    df["pA_age_fatigue"] = df["playerA_age"] * df["fatigue_10d_diff"]
    df["pB_age_fatigue"] = df["playerB_age"] * df["fatigue_10d_diff"]
    df["age_fatigue_diff"] = df["pA_age_fatigue"] - df["pB_age_fatigue"]

    return df

def compute_height_features(df):
    # Combine all heights to get mean and std
    all_heights = pd.concat([df["playerA_height"], df["playerB_height"]])
    mean_height, std_height = all_heights.mean(), all_heights.std()

    # Distance from mean (prime height)
    df["playerA_prime_height"] = abs(df["playerA_height"] - mean_height)
    df["playerB_prime_height"] = abs(df["playerB_height"] - mean_height)

    # Difference in prime height
    df["prime_height_diff"] = df["playerA_prime_height"] - df["playerB_prime_height"]

    # Raw height difference
    df["raw_height_diff"] = df["playerA_height"] - df["playerB_height"]

    return df

def compute_rolling_h2h(df, window=10, alpha=1):
    # Ensure chronological order
    df = df.sort_values(["pseudo_date", "match_num"]).reset_index(drop=True)

    # Key: (min_id, max_id, surface) -> Value: [list of outcomes]
    h2h_dict = defaultdict(list)

    playerA_h2h = []
    playerB_h2h = []

    for idx, row in df.iterrows():
        a_id, b_id = row["playerA_id"], row["playerB_id"]
        surface = row["surface"]

        # Consistent key regardless of who is A or B
        p_min, p_max = sorted([a_id, b_id])
        key = (p_min, p_max, surface)

        # 1. GET HISTORY (Looking only at the current surface)
        history = h2h_dict[key][-window:]
        wins_min = sum(history)
        total = len(history)

        # 2. LAPLACE SMOOTHING
        # This is vital for surface-H2H because the sample size (total)
        # will be much smaller than global H2H.
        smoothed_win_pct = (wins_min + alpha) / (total + 2 * alpha)

        # 3. ASSIGN WIN %
        if a_id == p_min:
            playerA_h2h.append(smoothed_win_pct)
            playerB_h2h.append(1 - smoothed_win_pct)
        else:
            playerA_h2h.append(1 - smoothed_win_pct)
            playerB_h2h.append(smoothed_win_pct)

        # 4. UPDATE HISTORY
        # Determine if the winner was the 'min_id' player
        # (Assuming log_target == 1 means Player A won)
        winner_is_min = (row["log_target"] == 1 and a_id == p_min) or \
                        (row["log_target"] == 0 and b_id == p_min)

        h2h_dict[key].append(1 if winner_is_min else 0)

    # Add features to dataframe
    df["playerA_surface_h2h_pct"] = playerA_h2h
    df["playerB_surface_h2h_pct"] = playerB_h2h
    df["surface_h2h_diff"] = df["playerA_surface_h2h_pct"] - df["playerB_surface_h2h_pct"]

    return df

def compute_service_stats(df, window=10):
    # 1. Initialize Histories
    service_hist = defaultdict(list)
    return_hist = defaultdict(list)
    first_serve_hist = defaultdict(list)
    ace_hist = defaultdict(list)  # <--- New Ace History

    # 2. Initialize Output Lists
    playerA_adv = []
    playerB_adv = []
    playerA_first_pct = []
    playerB_first_pct = []
    playerA_ace_pct = []  # <--- New List
    playerB_ace_pct = []  # <--- New List

    # Ensure chronological order
    df = df.sort_values('pseudo_date').reset_index(drop=True)

    # 3. Single Helper Function for Weighted Averages
    def get_weighted_avg(hist, player_id, default_val):
        """
        Calculates weighted rolling average based on history.
        weight = i (linear), where i=1 is the most recent match.
        """
        last_matches = hist[player_id][-window:]

        if not last_matches:
            return default_val

        total_numerator = 0
        total_denominator = 0

        # Iterate backwards (most recent first) but apply weights linearly
        # (oldest=1, newest=window is usually better, but let's stick to your logic:
        # Your logic was: reversed(last_matches) with enumerate 1..N.
        # This gives the *most recent* match a weight of 1 and the *oldest* a higher weight?
        # WAIT: Let's correct the weighting logic to standard "Recency Bias".
        # Standard: Newest match gets highest weight.

        # CORRECTED LOGIC:
        # 1. Take last N matches.
        # 2. Iterate normal order (Oldest -> Newest).
        # 3. Weight increases with index.

        for i, (val, total) in enumerate(last_matches, 1):
            weight = i  # Oldest match in window = 1, Newest = 5
            total_numerator += val * weight
            total_denominator += total * weight

        return total_numerator / total_denominator if total_denominator > 0 else default_val

    # 4. Iterate through matches
    # Using itertuples for speed as discussed
    for row in df.itertuples():
        playerA, playerB = row.playerA_id, row.playerB_id

        # --- EXTRACT CURRENT STATS ---

        # Service & Return Points
        pA_sv_won = row.playerA_1stWon + row.playerA_2ndWon
        pA_sv_tot = row.playerA_svpt
        pB_sv_won = row.playerB_1stWon + row.playerB_2ndWon
        pB_sv_tot = row.playerB_svpt

        pA_ret_won = pB_sv_tot - pB_sv_won
        pB_ret_won = pA_sv_tot - pA_sv_won

        # First Serve In
        pA_1st_in = row.playerA_1stIn
        pB_1st_in = row.playerB_1stIn

        # Aces (New)
        pA_aces = row.playerA_ace
        pB_aces = row.playerB_ace

        # --- CALCULATE PRE-MATCH METRICS (Using History) ---

        # 1. Service Advantage (Default 0.5)
        pA_srv_perf = get_weighted_avg(service_hist, playerA, 0.5)
        pB_ret_perf = get_weighted_avg(return_hist, playerB, 0.5)
        playerA_adv.append(pA_srv_perf - pB_ret_perf)

        pB_srv_perf = get_weighted_avg(service_hist, playerB, 0.5)
        pA_ret_perf = get_weighted_avg(return_hist, playerA, 0.5)
        playerB_adv.append(pB_srv_perf - pA_ret_perf)

        # 2. First Serve % (Default 0.65)
        playerA_first_pct.append(get_weighted_avg(first_serve_hist, playerA, 0.65))
        playerB_first_pct.append(get_weighted_avg(first_serve_hist, playerB, 0.65))

        # 3. Ace % (Default 0.05 or 5%) <--- New Calculation
        # We use 0.05 as a neutral starting point for aces
        playerA_ace_pct.append(get_weighted_avg(ace_hist, playerA, 0.05))
        playerB_ace_pct.append(get_weighted_avg(ace_hist, playerB, 0.05))

        # --- UPDATE HISTORY (Post-Match) ---

        service_hist[playerA].append((pA_sv_won, pA_sv_tot))
        service_hist[playerB].append((pB_sv_won, pB_sv_tot))

        return_hist[playerA].append((pA_ret_won, pB_sv_tot))
        return_hist[playerB].append((pB_ret_won, pA_sv_tot))

        first_serve_hist[playerA].append((pA_1st_in, pA_sv_tot))
        first_serve_hist[playerB].append((pB_1st_in, pB_sv_tot))

        # Update Ace History
        ace_hist[playerA].append((pA_aces, pA_sv_tot))
        ace_hist[playerB].append((pB_aces, pB_sv_tot))

    # 5. Assign Columns
    df['playerA_service_advantage'] = playerA_adv
    df['playerB_service_advantage'] = playerB_adv
    df['service_advantage_diff'] = df['playerA_service_advantage'] - df['playerB_service_advantage']

    df['playerA_first_serve_pct'] = playerA_first_pct
    df['playerB_first_serve_pct'] = playerB_first_pct
    df['first_serve_pct_diff'] = df['playerA_first_serve_pct'] - df['playerB_first_serve_pct']

    # New Ace Columns
    df['playerA_ace_pct'] = playerA_ace_pct
    df['playerB_ace_pct'] = playerB_ace_pct
    df['ace_pct_diff'] = df['playerA_ace_pct'] - df['playerB_ace_pct']

    return df

def compute_tournament_history(df, alpha=2):
    """
    Tracks how well a player has performed at a specific tournament in the past.
    Uses Laplace smoothing to handle players with no/little history.
    """
    # 1. Sort chronologically to prevent leakage
    df = df.sort_values(['pseudo_date', 'match_num']).reset_index(drop=True)

    # 2. Key: (player_id, tourney_name) -> [wins, total_matches]
    tourney_hist = defaultdict(lambda: [0, 0])

    playerA_tourney_win_pct = []
    playerB_tourney_win_pct = []

    # 3. Iterate through the matches
    for row in df.itertuples():
        pA = row.playerA_id
        pB = row.playerB_id
        t_name = row.tourney_name

        # --- PRE-MATCH CALCULATION ---
        # Get history for Player A
        a_wins, a_total = tourney_hist[(pA, t_name)]
        # Laplace Smoothing: (wins + alpha) / (total + 2*alpha)
        # alpha=2 ensures 1-0 records aren't treated as 100% win rates
        a_pct = (a_wins + alpha) / (a_total + 2 * alpha)

        # Get history for Player B
        b_wins, b_total = tourney_hist[(pB, t_name)]
        b_pct = (b_wins + alpha) / (b_total + 2 * alpha)

        playerA_tourney_win_pct.append(a_pct)
        playerB_tourney_win_pct.append(b_pct)

        # --- POST-MATCH UPDATE ---
        # Check who won (assuming log_target=1 means Player A won)
        if row.log_target == 1:
            tourney_hist[(pA, t_name)][0] += 1  # A wins
        else:
            tourney_hist[(pB, t_name)][0] += 1  # B wins

        tourney_hist[(pA, t_name)][1] += 1  # A total
        tourney_hist[(pB, t_name)][1] += 1  # B total

    # 4. Assign new features
    df['playerA_tourney_history'] = playerA_tourney_win_pct
    df['playerB_tourney_history'] = playerB_tourney_win_pct
    df['tourney_history_diff'] = df['playerA_tourney_history'] - df['playerB_tourney_history']

    return df

def compute_elo_velocity(df):
    """
    Calculates the change in Elo over the last 3 months and 1 year.
    'Velocity' identifies players on a steep upward or downward trajectory.
    """
    # Ensure chronological order
    df = df.sort_values(['pseudo_date', 'match_num']).reset_index(drop=True)

    # Store history as {player_id: [(date, elo), (date, elo), ...]}
    elo_history = {}

    velocity_features = {
        'pA_elo_3m_delta': [], 'pB_elo_3m_delta': [],
        'pA_elo_1y_delta': [], 'pB_elo_1y_delta': []
    }

    for row in df.itertuples():
        current_date = row.pseudo_date

        for prefix in ['pA', 'pB']:
            p_id = getattr(row, f'playerA_id' if prefix == 'pA' else 'playerB_id')
            current_elo = getattr(row, f'playerA_global_elo' if prefix == 'pA' else 'playerB_global_elo')

            # Initialize history if new player
            if p_id not in elo_history:
                elo_history[p_id] = []

            # Look back logic
            history = elo_history[p_id]

            for period_name, days in [('3m', 90), ('1y', 365)]:
                target_date = current_date - timedelta(days=days)

                # Find the elo closest to the target date (but not after it)
                past_elo = None
                for hist_date, hist_elo in reversed(history):
                    if hist_date <= target_date:
                        past_elo = hist_elo
                        break

                # Calculate delta (Current - Past). If no history, delta is 0.
                delta = current_elo - past_elo if past_elo is not None else 0
                velocity_features[f'{prefix}_elo_{period_name}_delta'].append(delta)

            # Update history AFTER calculation (to prevent leakage)
            elo_history[p_id].append((current_date, current_elo))

    # Add to dataframe
    for feat_name, values in velocity_features.items():
        df[feat_name] = values

    # Create difference features for the model
    df['elo_velocity_3m_diff'] = df['pA_elo_3m_delta'] - df['pB_elo_3m_delta']
    df['elo_velocity_1y_diff'] = df['pA_elo_1y_delta'] - df['pB_elo_1y_delta']

    return df

def compute_dominance_ratio(df, window=15):
    """
    Calculates the rolling Dominance Ratio.
    A DR > 1.0 means the player wins a higher percentage of return points
    than they lose on their own serve.
    """
    # 1. Ensure chronological order
    df = df.sort_values(['pseudo_date', 'match_num']).reset_index(drop=True)

    # 2. Track historical match efficiency {player_id: [(ret_won_pct, srv_lost_pct)]}
    player_stats = {}

    pA_dr_rolling = []
    pB_dr_rolling = []

    for row in df.itertuples():
        pA_id, pB_id = row.playerA_id, row.playerB_id

        # --- PRE-MATCH CALCULATION (Using past history) ---
        for p_id, target_list in [(pA_id, pA_dr_rolling), (pB_id, pB_dr_rolling)]:
            if p_id in player_stats and len(player_stats[p_id]) > 0:
                recent = player_stats[p_id][-window:]
                avg_ret_won = np.mean([x[0] for x in recent])
                avg_srv_lost = np.mean([x[1] for x in recent])

                # DR = (Return Pts Won %) / (Serve Pts Lost %)
                # Use max(..., 0.01) to avoid division by zero
                dr = avg_ret_won / max(avg_srv_lost, 0.01)
                target_list.append(dr)
            else:
                target_list.append(1.0)  # Neutral baseline

        # --- POST-MATCH UPDATE (Adding this match to player history) ---
        # Calculate stats for Player A in THIS match
        pA_srv_pts = row.playerA_svpt
        pA_srv_won = row.playerA_1stWon + row.playerA_2ndWon
        pA_srv_lost_pct = (pA_srv_pts - pA_srv_won) / max(pA_srv_pts, 1)

        pB_srv_pts = row.playerB_svpt
        pB_srv_won = row.playerB_1stWon + row.playerB_2ndWon
        pA_ret_won_pct = (pB_srv_pts - pB_srv_won) / max(pB_srv_pts, 1)

        # Calculate stats for Player B in THIS match
        pB_srv_lost_pct = (pB_srv_pts - pB_srv_won) / max(pB_srv_pts, 1)
        pB_ret_won_pct = (pA_srv_pts - pA_srv_won) / max(pA_srv_pts, 1)

        # Append to history
        player_stats.setdefault(pA_id, []).append((pA_ret_won_pct, pA_srv_lost_pct))
        player_stats.setdefault(pB_id, []).append((pB_ret_won_pct, pB_srv_lost_pct))

    df['playerA_DR'] = pA_dr_rolling
    df['playerB_DR'] = pB_dr_rolling
    df['dominance_ratio_diff'] = df['playerA_DR'] - df['playerB_DR']

    return df

def data_cleaning():

    df = load_raw_data()
    df = clean_basic_fields(df)
    df = clean_score_fields(df)
    df = add_bets(df)
    df.to_csv("post_bets.csv", index=False)
    df = create_rank_order_features(df)
    df = compute_dates(df)
    df = compute_elo_features(df)
    df = compute_fatigue(df)
    df = compute_age_features(df)
    df  = compute_height_features(df)
    df = compute_service_stats(df)
    df = compute_rolling_h2h(df)
    df = compute_tournament_history(df)
    df = compute_elo_velocity(df)
    df = compute_dominance_ratio(df)
    df.to_csv("data_cleaned_shuffled.csv", index=False)

    return df

def feature_engineering():
    # ---------------------------------------
    # 1. Load dataset
    # ---------------------------------------
    df = pd.read_csv("data_cleaned_shuffled.csv")
    df = df.sort_values(["pseudo_date", "match_num"]).reset_index(drop=True)
    df = df.iloc[3000:].reset_index(drop=True)  # warm-up drop

    # ---------------------------------------
    # 2. Select features
    # ---------------------------------------
    FEATURES_NUM = [
        "surface_elo_diff",
        "global_elo_diff",
        "ace_pct_diff",
        "fatigue_10d_diff",
        "year_fatigue_diff",
        "prime_age_diff",
        "raw_age_diff",
        "prime_height_diff",
        "service_advantage_diff",
        "tourney_history_diff",
        "dominance_ratio_diff",
        "raw_age_diff_sq",
        "age_fatigue_diff",
    ]
    FEATURES_CAT = []
    LOG_TARGET = "log_target"
    LIN_TARGET = df["playerA_points_won_pct"]
    LIN_TARGET2 = df["minutes"]

    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[LOG_TARGET].copy()

    print("Selected feature matrix shape:", X.shape)

    # ---------------------------------------
    # 3. Handle Outliers (IQR Winsorization)
    # ---------------------------------------
    IQR_cap = ["surface_elo_diff",
        "global_elo_diff",
        "fatigue_10d_diff",
        "year_fatigue_diff",
        "prime_age_diff",
        "raw_age_diff",
        "prime_height_diff",
               "raw_age_diff_sq",
               "age_fatigue_diff",
                       ]
    for col in IQR_cap:
        Q1, Q3 = X[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X[col] = X[col].clip(lower, upper)

    print("Finished outlier winsorization.")

    X_export = X.copy()
    y_export = y.copy()

    phaseII = X_export.copy()
    phaseII[LOG_TARGET] = y_export
    phaseII["playerA_points_won_pct"] = LIN_TARGET
    phaseII["minutes"] = LIN_TARGET2

    phaseII.to_csv("phaseII.csv", index=False)
    print("Exported dataset ready for phaseII.")

    #Split the data
    cutoff = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    print("Train/Test Split Complete:")
    print(f"X_train_raw shape: {X_train_raw.shape}, X_test_raw shape: {X_test_raw.shape}")

    scaler = StandardScaler()

    # FIT the scaler ONLY on the training data
    X_train_num_scaled = scaler.fit_transform(X_train_raw[FEATURES_NUM])
    # TRANSFORM the test data using the TR AINING fit
    X_test_num_scaled = scaler.transform(X_test_raw[FEATURES_NUM])

    #Encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop='first')

    # FIT the encoder ONLY on the training data
    X_train_cat = encoder.fit_transform(X_train_raw[FEATURES_CAT])
    # TRANSFORM the test data using the TRAINING fit
    X_test_cat = encoder.transform(X_test_raw[FEATURES_CAT])

    # Get feature names for final list
    cat_feature_names = encoder.get_feature_names_out(FEATURES_CAT)

    # ---------------------------------------
    # 4. Combine into final feature matrices
    # ---------------------------------------
    # Stack the scaled numerical features and the encoded categorical features
    X_train = np.hstack([X_train_num_scaled, X_train_cat])
    X_test = np.hstack([X_test_num_scaled, X_test_cat])

    # Create the full list of feature names
    full_feature_list = FEATURES_NUM + list(cat_feature_names)

    print("\nFinal Processed Data Shapes:")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"Total features: {len(full_feature_list)}")


    # ---------------------------------------
    # 5a. Singular Value Decomposition (SVD)
    # ---------------------------------------
    U, S, VT = np.linalg.svd(X_train_num_scaled, full_matrices=False)
    print("\nSVD singular values:", S)
    print("Explained variance ratio by SVD (normalized):", S ** 2 / np.sum(S ** 2))

    # 🚨 NEW: Calculate the Condition Number (Ratio of largest to smallest singular value)
    if len(S) > 1:
        # S[0] is the largest, S[-1] is the smallest
        condition_number = S[0] / S[-1]
        print(f"\nCondition Number (S_max / S_min): {condition_number:.2f} ⚠️")
        print("Interpretation: A high value (> 30-100) indicates significant multicollinearity/instability.")
    else:
        print("\nCondition Number: Not applicable (only one feature).")

    # ---------------------------------------
    # 5b. Variance Inflation Factor (VIF)
    # ---------------------------------------
    vif_data = pd.DataFrame()
    vif_data["Feature"] = FEATURES_NUM
    vif_data["VIF"] = [variance_inflation_factor(X_train_num_scaled, i) for i in range(X_train_num_scaled.shape[1])]
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data)

    # ---------------------------------------
    # 7. Covariance matrix (UNSTANDARDIZED numerical features)
    # ---------------------------------------
    # Use X_train_raw[FEATURES_NUM] because this is before scaling
    cov_matrix = np.cov(X_train_raw[FEATURES_NUM].to_numpy(), rowvar=False)

    print("\nRaw Covariance Matrix (Numerical Features Only):")
    print(pd.DataFrame(cov_matrix, index=FEATURES_NUM, columns=FEATURES_NUM))

    plt.figure(figsize=(16, 8))
    sns.heatmap(
        pd.DataFrame(cov_matrix, index=FEATURES_NUM, columns=FEATURES_NUM),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
    )
    plt.title("Covariance Matrix Heatmap")
    plt.tight_layout()
    plt.savefig("covariance_heatmap.png", dpi=300)
    plt.show()

    # ---------------------------------------
    # 8. Pearson correlation matrix
    # ---------------------------------------
    corr_matrix = pd.DataFrame(X_train_num_scaled, columns=FEATURES_NUM).corr()
    plt.figure(figsize=(16, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Correlation — Numerical Features")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # 10. PCA (numerical only)
    # ---------------------------------------
    pca = PCA(n_components=len(FEATURES_NUM))
    X_train_pca = pca.fit_transform(X_train_num_scaled)
    X_test_pca = pca.transform(X_test_num_scaled)
    print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)
    explained_variance_ratio = np.array(pca.explained_variance_ratio_)
    # Calculate the cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Create component indices for the x-axis
    n_components = len(explained_variance_ratio)
    components = np.arange(1, n_components + 1)

    # Plotting setup
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    plt.plot(components, cumulative_variance, marker='o', linestyle='-', color='purple')

    # Add 90% target line (you can change 0.90 to 0.95 or another target)
    target_variance = 0.95
    n_components_target = np.argmax(cumulative_variance >= target_variance) + 1

    plt.axhline(y=target_variance, color='r', linestyle='--', label=f'{target_variance*100:.0f}% Variance')
    plt.axvline(x=n_components_target, color='r', linestyle='--')
    plt.text(n_components_target + 0.5, target_variance - 0.05,
            f'{n_components_target} Components', color='r', fontsize=12)

    plt.title('Cumulative Explained Variance Plot', fontsize=16)
    plt.xlabel('Number of Principal Components', fontsize=14)
    plt.ylabel('Cumulative Explained Variance Ratio', fontsize=14)
    plt.xticks(components)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('your_actual_cumulative_explained_variance_plot.png')

    # ---------------------------------------
    # 11. LDA (full feature set)
    # ---------------------------------------
    lda = LDA(n_components=1)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    print("LDA explained variance:", lda.explained_variance_ratio_)

    plt.figure(figsize=(8, 4))
    plt.hist(X_train_lda[y_train == 1], alpha=.5, label="Player A wins (1)")
    plt.hist(X_train_lda[y_train == 0], alpha=.5, label="Player B wins (0)")
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

    #PCA random forest
    pca = PCA(n_components=len(FEATURES_NUM)-1)
    X_train_pca = pca.fit_transform(X_train_num_scaled)
    X_test_pca = pca.transform(X_test_num_scaled)
    X_train_pca_full = np.hstack([X_train_pca, X_train_cat])
    X_test_pca_full = np.hstack([X_test_pca, X_test_cat])

    rf_pca = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
    )

    rf_pca.fit(X_train_pca_full, y_train)

    # --- Step 3: Evaluate the Baseline Model ---
    y_pred_pca = rf_pca.predict(X_test_pca_full)

    print("\n" + "=" * 50)
    print("       Random Forest on PCA")
    print("=" * 50)
    print(f"PCA Baseline Accuracy: {accuracy_score(y_test, y_pred_pca):.4f}")
    print("\nClassification Report (PCA Baseline):\n", classification_report(y_test, y_pred_pca))

    # ---------------------------------------
    # 13. Feature importance
    # ---------------------------------------
    pca_feature_names = [f'PC{i + 1}' for i in range(X_train_pca.shape[1])]
    # The categorical features names are still correct
    categorical_feature_names = list(cat_feature_names)

    # The new, correct list of 12 features
    pca_full_feature_list = pca_feature_names + categorical_feature_names
    importance = rf_pca.feature_importances_
    fi_df_pca = pd.DataFrame({
        "Feature": pca_full_feature_list,  # <-- CORRECTED
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    print("\nRandom Forest Feature Importances (PCA Components):")
    print(fi_df_pca)

    # ---------------------------------------
    # 8. Pearson correlation matrix
    # ---------------------------------------
    corr_matrix = pd.DataFrame(X_train_pca, columns=pca_feature_names).corr()
    plt.figure(figsize=(16, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Correlation — Numerical Features")
    plt.tight_layout()
    plt.show()
    # ---------------------------------------

    # a. Stack all features back together (Training and Testing)
    # Combine the PCA numerical features and the categorical features
    X_pca_full = np.vstack([X_train_pca_full, X_test_pca_full])

    # b. Combine all target variables (Training and Testing)
    y_full = np.concatenate([y_train, y_test])

    # c. Get the list of PCA feature names for the DataFrame
    pca_feature_names = [f'PC{i + 1}' for i in range(X_train_pca.shape[1])]
    categorical_feature_names = list(encoder.get_feature_names_out(FEATURES_CAT))

    pca_export_columns = pca_feature_names + categorical_feature_names

    # d. Create the DataFrame
    df_pca_export = pd.DataFrame(X_pca_full, columns=pca_export_columns)

    # Add target, log target, and other analysis columns
    df_pca_export[LOG_TARGET] = y_full
    # These lines are now CORRECT because 'df' was already subsetted and index reset!
    df_pca_export["playerA_points_won_pct"] = df["playerA_points_won_pct"]
    df_pca_export["minutes"] = df["minutes"]
    df_pca_export["playerA"] = df["playerA_name"]
    df_pca_export["playerB"] = df["playerB_name"]
    df_pca_export["pseudo_date"] = df["pseudo_date"]  # Changed to tourney_date for consistency
    # based on your sorting at the top

    # f. Export to CSV
    df_pca_export.to_csv("phaseII_pca_reduced.csv", index=False)

    print("\nExported PCA-reduced dataset (phaseII_pca_reduced.csv) ready for phase II.")

    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    df_non_pca_export = pd.DataFrame(
        X_full,
        columns=full_feature_list
    )

    df_non_pca_export[LOG_TARGET] = y_full
    df_non_pca_export["playerA_points_won_pct"] = df["playerA_points_won_pct"]
    df_non_pca_export["minutes"] = df["minutes"]
    df_non_pca_export["playerA"] = df["playerA_name"]
    df_non_pca_export["playerB"] = df["playerB_name"]
    df_non_pca_export["pseudo_date"] = df["pseudo_date"]

    df_non_pca_export.to_csv("phaseII_non_pca_classification.csv", index=False)
    print("✅ Exported NON-PCA classification dataset.")

if __name__ == '__main__':
    data_cleaning()
    #feature_engineering()