# ===============================
# XG-ELO HYBRID MODEL
# ===============================
# run the model: py xg_elo_model.py


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 1. LOAD AND SORT DATA
# ===============================

df = pd.read_csv("whl_2025.csv")

df = df.sort_values("game_id")
df = df.reset_index(drop=True)
# ===============================
# 2. INITIALIZE ELO SYSTEM
# ===============================

teams = pd.concat([df["home_team"], df["away_team"]]).unique()
elo_ratings = {team: 1500 for team in teams}

K = 20
HOME_ADVANTAGE = 50

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

# Lists to store Elo features
elo_home = []
elo_away = []
elo_diff = []
elo_prob_home = []

# ===============================
# 3. BUILD ELO RATINGS
# ===============================

for idx, row in df.iterrows():
    
    home = row["home_team"]
    away = row["away_team"]
    
    home_rating = elo_ratings[home]
    away_rating = elo_ratings[away]
    
    adj_home_rating = home_rating + HOME_ADVANTAGE
    
    exp_home = expected_score(adj_home_rating, away_rating)
    
    # Store PRE-GAME ratings
    elo_home.append(home_rating)
    elo_away.append(away_rating)
    elo_diff.append(home_rating - away_rating)
    elo_prob_home.append(exp_home)
    
    # Actual result
    actual_home = 1 if row["home_goals"] > row["away_goals"] else 0
    
    # Margin of victory multiplier
    goal_diff = abs(row["home_goals"] - row["away_goals"])
    margin_multiplier = np.log(goal_diff + 1)
    
    # Update ratings
    elo_ratings[home] += K * margin_multiplier * (actual_home - exp_home)
    elo_ratings[away] += K * margin_multiplier * ((1 - actual_home) - (1 - exp_home))

# Attach Elo features
df["elo_home"] = elo_home
df["elo_away"] = elo_away
df["elo_diff"] = elo_diff
df["elo_prob_home"] = elo_prob_home

# ===============================
# 4. CREATE TARGET VARIABLE
# ===============================

df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)

# ===============================
# 5. TRAIN / TEST SPLIT (CHRONOLOGICAL)
# ===============================

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

features = [
    "elo_diff",
    "elo_prob_home"
]

X_train = train[features]
y_train = train["home_win"]

X_test = test[features]
y_test = test["home_win"]

# ===============================
# 6. TRAIN XGBOOST MODEL
# ===============================

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False
)

model.fit(X_train, y_train)

# ===============================
# 7. PREDICTIONS
# ===============================

xgb_probs = model.predict_proba(X_test)[:, 1]
xgb_preds = (xgb_probs > 0.5).astype(int)

print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print("XGBoost Log Loss:", log_loss(y_test, xgb_probs))

# ===============================
# 8. HYBRID BLEND (XG + ELO)
# ===============================

alpha = 0.6  # You can tune this value

final_probs = alpha * xgb_probs + (1 - alpha) * test["elo_prob_home"].values
final_preds = (final_probs > 0.5).astype(int)

print("Hybrid Accuracy:", accuracy_score(y_test, final_preds))
print("Hybrid Log Loss:", log_loss(y_test, final_probs))

# ===============================
# 9. FINAL POWER RANKINGS
# ===============================

print("\nFinal Power Rankings:")
final_rankings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)

for rank, (team, rating) in enumerate(final_rankings, 1):
    print(f"{rank}. {team} - {round(rating, 2)}")
