# ==========================================
# TRUE XG-ELO HYBRID MODEL (STRUCTURAL)
# ==========================================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("whl_2025.csv")

# Sort chronologically (important)
df = df.sort_values("game_id").reset_index(drop=True)

# ==========================================
# 2. INITIALIZE ELO
# ==========================================

teams = pd.concat([df["home_team"], df["away_team"]]).unique()
elo_ratings = {team: 1500 for team in teams}

K = 20
HOME_ADVANTAGE = 50

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

# Lists to store pre-game features
elo_home_list = []
elo_away_list = []
elo_diff_list = []
elo_prob_list = []
home_xg_form_list = []
away_xg_form_list = []
form_diff_list = []

# Track rolling xG form manually (pre-game only)
team_xg_history = {team: [] for team in teams}

# ==========================================
# 3. LOOP THROUGH GAMES (BUILD FEATURES + UPDATE ELO)
# ==========================================

for idx, row in df.iterrows():
    
    home = row["home_team"]
    away = row["away_team"]
    
    home_rating = elo_ratings[home]
    away_rating = elo_ratings[away]
    
    adj_home_rating = home_rating + HOME_ADVANTAGE
    exp_home = expected_score(adj_home_rating, away_rating)
    
    # ----- PRE-GAME FEATURES -----
    
    elo_home_list.append(home_rating)
    elo_away_list.append(away_rating)
    elo_diff_list.append(home_rating - away_rating)
    elo_prob_list.append(exp_home)
    
    # Rolling xG form (last 5 games BEFORE this one)
    home_form = np.mean(team_xg_history[home][-5:]) if len(team_xg_history[home]) > 0 else 0
    away_form = np.mean(team_xg_history[away][-5:]) if len(team_xg_history[away]) > 0 else 0
    
    home_xg_form_list.append(home_form)
    away_xg_form_list.append(away_form)
    form_diff_list.append(home_form - away_form)
    
    # ----- ACTUAL GAME OUTCOME -----
    
    goal_result = 1 if row["home_goals"] > row["away_goals"] else 0
    
    # xG performance scaling (continuous)
    xg_diff = row["home_xg"] - row["away_xg"]
    xg_factor = np.tanh(xg_diff)  # smooth scaling
    xg_score = 0.5 + 0.5 * xg_factor
    
    # Blend actual result + xG dominance
    performance_score = 0.7 * goal_result + 0.3 * xg_score
    
    goal_diff = abs(row["home_goals"] - row["away_goals"])
    margin_multiplier = np.log(goal_diff + 1)
    
    # ----- UPDATE ELO (STRUCTURAL HYBRID) -----
    
    elo_ratings[home] += K * margin_multiplier * (performance_score - exp_home)
    elo_ratings[away] += K * margin_multiplier * ((1 - performance_score) - (1 - exp_home))
    
    # Update rolling xG history AFTER game
    team_xg_history[home].append(row["home_xg"])
    team_xg_history[away].append(row["away_xg"])

# ==========================================
# 4. ADD FEATURES TO DATAFRAME
# ==========================================

df["elo_diff"] = elo_diff_list
df["elo_prob_home"] = elo_prob_list
df["home_xg_form"] = home_xg_form_list
df["away_xg_form"] = away_xg_form_list
df["form_diff"] = form_diff_list

df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)

# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

features = [
    "elo_diff",
    "elo_prob_home",
    "form_diff"
]

X_train = train[features]
y_train = train["home_win"]

X_test = test[features]
y_test = test["home_win"]

# ==========================================
# 6. TRAIN XGBOOST
# ==========================================

model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# 7. EVALUATE
# ==========================================

xgb_probs = model.predict_proba(X_test)[:, 1]
xgb_preds = (xgb_probs > 0.5).astype(int)

print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print("XGBoost Log Loss:", log_loss(y_test, xgb_probs))

# ==========================================
# 8. HYBRID BLEND (ML + STRUCTURAL ELO)
# ==========================================

alpha = 0.65  # ML weight

final_probs = alpha * xgb_probs + (1 - alpha) * test["elo_prob_home"].values
final_preds = (final_probs > 0.5).astype(int)

print("Hybrid Accuracy:", accuracy_score(y_test, final_preds))
print("Hybrid Log Loss:", log_loss(y_test, final_probs))

# ==========================================
# 9. FINAL POWER RANKINGS
# ==========================================

print("\nFinal XG-Elo Power Rankings:")

final_rankings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)

for rank, (team, rating) in enumerate(final_rankings, 1):
    print(f"{rank}. {team} - {round(rating, 2)}")
