import random

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
    df['winner_id'] = df['winner_id'].astype(str)
    df['loser_id'] = df['loser_id'].astype(str)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["winner_ht"] = df["winner_ht"].fillna(df["winner_ht"].median())
    df["loser_ht"] = df["loser_ht"].fillna(df["loser_ht"].median())

    # --- Handedness normalization: R, L, U (Unknown), N (Unknown?) ---
    df["winner_hand"] = df["winner_hand"].fillna("U").replace("N", "U")
    df["loser_hand"] = df["loser_hand"].fillna("U").replace("N", "U")

    # --- Surface cleaning ---
    df["surface"] = df["surface"].fillna(df["surface"].mode())

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


    df["round"] = df["round"].fillna(df["round"].mode())

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

def create_rank_order_features(df, seed = 42):
    np.random.seed(seed)
    def reorder(row):

        new_row = {
            "tourney_id": row["tourney_id"],
            "tourney_name": row["tourney_name"],
            "tourney_date": row["tourney_date"],
            "tourney_level": row["tourney_level"],
            "round": row["round"],
            "surface": row["surface"],
            "match_num": row["match_num"],
            "minutes": row["minutes"],
        }
        val = np.random.choice([True, False])
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
            })

        if row['winner_id'] == new_row['playerA_id']:
            new_row['log_target'] = 1
        else:
            new_row['log_target'] = 0
        return pd.Series(new_row)
    df_new = df.apply(reorder, axis=1).reset_index(drop=True)
    return df_new

def compute_pseudo_dates(df):
    grand_slams = {"Wimbledon", "Roland Garros", "Australian Open", "US Open"}

    df['pseudo_date'] = df['tourney_date']
    tourney_dict = defaultdict(lambda: defaultdict(int))

    for idx, row in df.sort_values(['tourney_date', 'match_num']).iterrows():

        tid = row['tourney_id']
        playerA, playerB = row['playerA_id'], row['playerB_id']

        is_gs = str(row['tourney_name']) in grand_slams
        inc = 2 if is_gs else 1

        c_w = tourney_dict[tid].get(playerA, 0)
        c_l = tourney_dict[tid].get(playerB, 0)
        prior = max(c_w, c_l)

        df.at[idx, 'pseudo_date'] = row['tourney_date'] + pd.Timedelta(days=inc * prior)

        tourney_dict[tid][playerA] = prior + 1
        tourney_dict[tid][playerB] = prior + 1

    df = df.sort_values('pseudo_date').reset_index(drop=True)
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
    THIRTY_DAYS = pd.Timedelta(days=30)

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
            return 1 if last_date < date - THIRTY_DAYS else 0

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

    df["raw_age_diff"] = df["playerA_age"] - df["playerB_age"]

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

def compute_grand_slam_champion(df):
    """
    Adds features indicating if a player has won a Grand Slam up to that point.
    Assumes df is sorted by pseudo_date.
    Winner of GS final is determined by log_target: 1=playerA wins, 0=playerB wins.
    """
    df['playerA_grand_slam'] = 0
    df['playerB_grand_slam'] = 0

    gs_winners = defaultdict(bool)  # player_id -> has won GS yet?

    for idx, row in df.iterrows():
        pA = row['playerA_id']
        pB = row['playerB_id']

        # Mark if they've won GS before this match
        df.at[idx, 'playerA_grand_slam'] = int(gs_winners[pA])
        df.at[idx, 'playerB_grand_slam'] = int(gs_winners[pB])

        # Update GS winner if this match is a Grand Slam final
        if row['round'] == 'F' and row['tourney_level'] == 'G':
            if row['log_target'] == 1:
                gs_winners[pA] = True
            elif row['log_target'] == 0:
                gs_winners[pB] = True

    # Optional: difference column
    df['grand_slam_diff'] = df['playerA_grand_slam'] - df['playerB_grand_slam']

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

def compute_win_percentages(df):
    # Ensure sorted by date
    df = df.sort_values("pseudo_date").reset_index(drop=True)

    # Track tournament-level wins/losses
    tournament_stats = defaultdict(lambda: {"wins": 0, "losses": 0})

    # Track round-level wins/losses
    round_stats = defaultdict(lambda: {"wins": 0, "losses": 0})

    # Global fallback win percentage
    global_stats = defaultdict(lambda: {"wins": 0, "losses": 0})

    # Output columns
    df["playerA_tournament_win_pct"] = 0.5
    df["playerB_tournament_win_pct"] = 0.5
    df["playerA_round_win_pct"] = 0.5
    df["playerB_round_win_pct"] = 0.5

    # Laplace smoothing parameter
    alpha = 1

    for idx, row in df.iterrows():
        playerA = row["playerA_id"]
        playerB = row["playerB_id"]
        tour = row["tourney_name"]
        rnd = row["round"]
        winner = row["log_target"]  # 1 if playerA wins, 0 if playerB wins

        # --- Tournament Win % ---
        playerA_tourney = tournament_stats[(playerA, tour)]
        playerB_tourney = tournament_stats[(playerB, tour)]
        playerA_global = global_stats[playerA]
        playerB_global = global_stats[playerB]

        # Player A tournament win %
        if (playerA_tourney["wins"] + playerA_tourney["losses"]) > 0:
            df.at[idx, "playerA_tournament_win_pct"] = (
                (playerA_tourney["wins"] + alpha) /
                (playerA_tourney["wins"] + playerA_tourney["losses"] + 2 * alpha)
            )
        else:
            df.at[idx, "playerA_tournament_win_pct"] = (
                (playerA_global["wins"] + alpha) /
                (playerA_global["wins"] + playerA_global["losses"] + 2 * alpha)
            )

        # Player B tournament win %
        if (playerB_tourney["wins"] + playerB_tourney["losses"]) > 0:
            df.at[idx, "playerB_tournament_win_pct"] = (
                (playerB_tourney["wins"] + alpha) /
                (playerB_tourney["wins"] + playerB_tourney["losses"] + 2 * alpha)
            )
        else:
            df.at[idx, "playerB_tournament_win_pct"] = (
                (playerB_global["wins"] + alpha) /
                (playerB_global["wins"] + playerB_global["losses"] + 2 * alpha)
            )

        # --- Round Win % ---
        playerA_round = round_stats[(playerA, rnd)]
        playerB_round = round_stats[(playerB, rnd)]

        # Player A round win %
        if (playerA_round["wins"] + playerA_round["losses"]) > 0:
            df.at[idx, "playerA_round_win_pct"] = (
                (playerA_round["wins"] + alpha) /
                (playerA_round["wins"] + playerA_round["losses"] + 2 * alpha)
            )
        else:
            df.at[idx, "playerA_round_win_pct"] = (
                (playerA_global["wins"] + alpha) /
                (playerA_global["wins"] + playerA_global["losses"] + 2 * alpha)
            )

        # Player B round win %
        if (playerB_round["wins"] + playerB_round["losses"]) > 0:
            df.at[idx, "playerB_round_win_pct"] = (
                (playerB_round["wins"] + alpha) /
                (playerB_round["wins"] + playerB_round["losses"] + 2 * alpha)
            )
        else:
            df.at[idx, "playerB_round_win_pct"] = (
                (playerB_global["wins"] + alpha) /
                (playerB_global["wins"] + playerB_global["losses"] + 2 * alpha)
            )

        # --- Update after match ---
        if winner == 1:  # Player A wins
            tournament_stats[(playerA, tour)]["wins"] += 1
            tournament_stats[(playerB, tour)]["losses"] += 1
            round_stats[(playerA, rnd)]["wins"] += 1
            round_stats[(playerB, rnd)]["losses"] += 1
            global_stats[playerA]["wins"] += 1
            global_stats[playerB]["losses"] += 1
        else:  # Player B wins
            tournament_stats[(playerB, tour)]["wins"] += 1
            tournament_stats[(playerA, tour)]["losses"] += 1
            round_stats[(playerB, rnd)]["wins"] += 1
            round_stats[(playerA, rnd)]["losses"] += 1
            global_stats[playerB]["wins"] += 1
            global_stats[playerA]["losses"] += 1

    df["tournament_win_pct_diff"] = df["playerA_tournament_win_pct"] - df["playerB_tournament_win_pct"]
    df["round_win_pct_diff"] = df["playerA_round_win_pct"] - df["playerB_round_win_pct"]
    return df

def compute_rolling_ace_pct(df):
    # Ensure sorted by date
    df = df.sort_values("pseudo_date").reset_index(drop=True)

    # Initialize new columns
    df["playerA_ace_pct_rolling_10"] = 0.0
    df["playerB_ace_pct_rolling_10"] = 0.0

    # Track per-player ace percentages
    player_ace_pct = defaultdict(list)

    for idx, row in df.iterrows():
        playerA = row["playerA_id"]
        playerB = row["playerB_id"]

        # --- Compute rolling for playerA ---
        if playerA in player_ace_pct:
            rolling_10 = player_ace_pct[playerA][-10:]
            df.at[idx, "playerA_ace_pct_rolling_10"] = sum(rolling_10) / max(1, len(rolling_10))
        else:
            df.at[idx, "playerA_ace_pct_rolling_10"] = 0.0

        # --- Compute rolling for playerB ---
        if playerB in player_ace_pct:
            rolling_10 = player_ace_pct[playerB][-10:]
            df.at[idx, "playerB_ace_pct_rolling_10"] = sum(rolling_10) / max(1, len(rolling_10))
        else:
            df.at[idx, "playerB_ace_pct_rolling_10"] = 0.0

        # --- Update ace % history ---
        playerA_pct = row["playerA_ace"] / max(1, row["playerA_svpt"])
        playerB_pct = row["playerB_ace"] / max(1, row["playerB_svpt"])

        player_ace_pct[playerA].append(playerA_pct)
        player_ace_pct[playerB].append(playerB_pct)

    # Optional: compute difference for model
    df["ace_pct_diff_rolling_10"] = df["playerA_ace_pct_rolling_10"] - df["playerB_ace_pct_rolling_10"]

    return df

def compute_rolling_first_serve_win_pct(df):
    # Ensure sorted by date
    df = df.sort_values("pseudo_date").reset_index(drop=True)
    df["pseudo_date"] = pd.to_datetime(df["pseudo_date"])

    # Output columns
    df["playerA_first_won_pct_rolling"] = 0.5
    df["playerB_first_won_pct_rolling"] = 0.5

    # Rolling calculation for each player
    for player in ["A", "B"]:
        player_col = f"player{player}_id"
        first_won_col = f"player{player}_1stWon"
        first_in_col = f"player{player}_1stIn"
        out_col = f"player{player}_first_won_pct_rolling"

        # Group by player, rolling 365 days on date
        rolling = (
            df
            .set_index("pseudo_date")
            .groupby(player_col)[[first_won_col, first_in_col]]
            .rolling("365D")
            .sum()
            .reset_index(level=0, drop=True)
        )

        # Compute percentage safely
        df[out_col] = rolling[first_won_col] / rolling[first_in_col].replace(0, 1)

    # Difference feature
    df["first_won_pct_diff_rolling"] = df["playerA_first_won_pct_rolling"] - df["playerB_first_won_pct_rolling"]

    return df



def export_final_dataset(df):

    cols = [
        "tourney_prefix",
        "tourney_level",
        "round",
        "surface",
        "pseudo_date",
        "higher_id", "higher_name", "higher_rank", "higher_age",
        "lower_id", "lower_name", "lower_rank", "lower_age",
        "height_diff",
        "age_advantage",
        "age_diff_z",
        "higher_tournament_win_pct", "higher_round_win_pct",
        "lower_tournament_win_pct", "lower_round_win_pct",
        "higher_global_elo", "lower_global_elo", "global_elo_diff",
        "higher_surface_elo", "lower_surface_elo", "surface_elo_diff",
        "higher_combined_elo", "lower_combined_elo", "combined_elo_diff",
        "higher_short_fatigue", "lower_short_fatigue", "fatigue_diff",
        "higher_last_minutes", "lower_last_minutes", "last_minutes_diff",
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
    df = df.sort_values(["tourney_date", "match_num"])
    df = compute_elo_features(df)
    df = compute_pseudo_dates(df)
    df = compute_fatigue(df)
    df = compute_age_features(df)
    df  = compute_height_features(df)
    #df = compute_grand_slam_champion(df)
    df = compute_rolling_ace_pct(df)
    df = compute_win_percentages(df)
    #df = compute_rolling_first_serve_win_pct(df)
    df.to_csv("data_cleaned_shuffled.csv", index=False)
    #df = compute_service_stats(df)

    #df = export_final_dataset(df)

    return df

def feature_engineering():

    # ---------------------------------------
    # 1. Load dataset
    # ---------------------------------------
    df = pd.read_csv("data_cleaned_shuffled.csv")
    df = df.sort_values("pseudo_date").reset_index(drop=True)

    # Drop first 3000 rows for warm-up period
    df = df.iloc[3000:].reset_index(drop=True)
    print("After warm-up drop:", df.shape)

    # ---------------------------------------
    # 2. Select Features
    # ---------------------------------------
    # Keep only the features you defined
    FEATURES_NUM = [
        "combined_elo_diff",
        "last_minutes_diff",
        "fatigue_10d_diff",
         "year_fatigue_diff",
        "raw_age_diff",
        "prime_age_diff",
        "raw_height_diff",
        "prime_height_diff",
    ]

    FEATURES_CAT = ["playerA_rusty", "playerB_rusty", "rusty_diff"]

    TARGET = "log_target"

    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[TARGET].copy()

    print("Selected feature matrix shape:", X.shape)

    # ---------------------------------------
    # 3. Handle Outliers (IQR Winsorization)
    # ---------------------------------------
    IQR_cap = FEATURES_NUM

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
    #data_cleaning()
    feature_engineering()