#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass
class HybridEloParams:
    k: float
    hfa: float
    xg_scale: float
    w_result: float
    mov_mult: bool
    scale: float = 400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WHL phase 1 pipeline")
    parser.add_argument(
        "--data-csv",
        default="whl_2025 - whl_2025(1).csv",
        help="Path to WHL game-level line summary CSV.",
    )
    parser.add_argument(
        "--matchups-csv",
        default=None,
        help="Optional path to round-1 matchups CSV (must include home_team and away_team).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--tune-elo",
        action="store_true",
        help="Run grid search for hybrid Elo params. If false, use high-performing defaults.",
    )
    parser.add_argument(
        "--order-robust",
        action="store_true",
        help="Use shuffle-ensemble hybrid Elo to remove dependence on arbitrary game order.",
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=300,
        help="Number of random game-order permutations for order-robust ensemble.",
    )
    parser.add_argument(
        "--tune-shuffles",
        type=int,
        default=40,
        help="Number of shuffles used during order-robust Elo tuning.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def load_and_validate(data_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(data_csv)
    game_num = pd.to_numeric(df["game_id"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    if game_num.isna().any():
        bad = df.loc[game_num.isna(), "game_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Could not parse numeric game_id for values: {bad}")
    df["game_num"] = game_num.astype(int)
    return df


def aggregate_games(df: pd.DataFrame) -> pd.DataFrame:
    games = (
        df.groupby("game_id")
        .agg(
            game_num=("game_num", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            went_ot=("went_ot", "first"),
            home_goals=("home_goals", "sum"),
            away_goals=("away_goals", "sum"),
            home_xg=("home_xg", "sum"),
            away_xg=("away_xg", "sum"),
            home_shots=("home_shots", "sum"),
            away_shots=("away_shots", "sum"),
        )
        .reset_index()
        .sort_values("game_num")
        .reset_index(drop=True)
    )
    games["home_win"] = (games["home_goals"] > games["away_goals"]).astype(int)
    games["goal_margin"] = games["home_goals"] - games["away_goals"]
    games["xg_margin"] = games["home_xg"] - games["away_xg"]
    return games


def build_team_summary(games: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "team": games["home_team"],
            "opp": games["away_team"],
            "gf": games["home_goals"],
            "ga": games["away_goals"],
            "xgf": games["home_xg"],
            "xga": games["away_xg"],
            "sf": games["home_shots"],
            "sa": games["away_shots"],
            "win": (games["home_goals"] > games["away_goals"]).astype(int),
            "loss": (games["home_goals"] < games["away_goals"]).astype(int),
            "went_ot": games["went_ot"],
        }
    )
    away = pd.DataFrame(
        {
            "team": games["away_team"],
            "opp": games["home_team"],
            "gf": games["away_goals"],
            "ga": games["home_goals"],
            "xgf": games["away_xg"],
            "xga": games["home_xg"],
            "sf": games["away_shots"],
            "sa": games["home_shots"],
            "win": (games["away_goals"] > games["home_goals"]).astype(int),
            "loss": (games["away_goals"] < games["home_goals"]).astype(int),
            "went_ot": games["went_ot"],
        }
    )
    tg = pd.concat([home, away], ignore_index=True)
    tg["pts"] = np.where(tg["win"] == 1, 2, np.where((tg["loss"] == 1) & (tg["went_ot"] == 1), 1, 0))

    team = tg.groupby("team").agg(
        GP=("team", "size"),
        W=("win", "sum"),
        L=("loss", "sum"),
        OT=("went_ot", "sum"),
        PTS=("pts", "sum"),
        GF=("gf", "sum"),
        GA=("ga", "sum"),
        xGF=("xgf", "sum"),
        xGA=("xga", "sum"),
        SF=("sf", "sum"),
        SA=("sa", "sum"),
    )
    team["PCT"] = team["PTS"] / (2 * team["GP"])
    team["GD"] = team["GF"] - team["GA"]
    team["xGD"] = team["xGF"] - team["xGA"]
    team["GFpg"] = team["GF"] / team["GP"]
    team["GApg"] = team["GA"] / team["GP"]
    team["xGFpg"] = team["xGF"] / team["GP"]
    team["xGApg"] = team["xGA"] / team["GP"]
    team["xGDpg"] = team["xGD"] / team["GP"]
    team["SH_pct"] = team["GF"] / team["SF"]
    team["SV_pct"] = 1 - (team["GA"] / team["SA"])
    team["PDO"] = team["SH_pct"] + team["SV_pct"]
    return team.sort_values(["PCT", "GD", "xGD"], ascending=False)


def run_hybrid_elo(games: pd.DataFrame, params: HybridEloParams) -> tuple[pd.DataFrame, dict[str, float]]:
    teams = sorted(set(games["home_team"]).union(set(games["away_team"])))
    ratings = {t: 1500.0 for t in teams}

    recs = []
    for row in games.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        rh = ratings[home]
        ra = ratings[away]

        p_home = 1.0 / (1.0 + 10 ** (-((rh + params.hfa) - ra) / params.scale))
        soft_result = 1.0 / (1.0 + np.exp(-(row.xg_margin) / params.xg_scale))
        target = params.w_result * row.home_win + (1.0 - params.w_result) * soft_result
        mov = np.log1p(abs(row.goal_margin)) if params.mov_mult else 1.0

        delta = params.k * mov * (target - p_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta

        recs.append(
            {
                "game_id": row.game_id,
                "game_num": row.game_num,
                "home_team": home,
                "away_team": away,
                "home_win": row.home_win,
                "home_win_prob": p_home,
                "away_win_prob": 1.0 - p_home,
                "pred_home_win_prob": p_home,
                "pre_home_rating": rh,
                "pre_away_rating": ra,
                "post_home_rating": ratings[home],
                "post_away_rating": ratings[away],
            }
        )

    pred_df = pd.DataFrame(recs)
    return pred_df, ratings


def rolling_eval_mask(n_games: int, start: int = 300, step: int = 170, min_chunk: int = 80) -> np.ndarray:
    mask = np.zeros(n_games, dtype=bool)
    cur = start
    while cur < n_games:
        end = min(cur + step, n_games)
        if (end - cur) < min_chunk:
            break
        mask[cur:end] = True
        cur += step
    return mask


def tune_hybrid_elo(games: pd.DataFrame) -> HybridEloParams:
    mask = rolling_eval_mask(len(games))
    y = games["home_win"].to_numpy()

    grid_k = [4, 6, 8, 10, 12, 15]
    grid_hfa = [30, 40, 50, 60]
    grid_xg_scale = [0.4, 0.5, 0.7, 1.0]
    grid_w_result = [0.5, 0.6, 0.7]
    grid_mov = [True, False]

    best: tuple[float, HybridEloParams] | None = None
    for k in grid_k:
        for hfa in grid_hfa:
            for xg_scale in grid_xg_scale:
                for w_result in grid_w_result:
                    for mov_mult in grid_mov:
                        params = HybridEloParams(
                            k=k,
                            hfa=hfa,
                            xg_scale=xg_scale,
                            w_result=w_result,
                            mov_mult=mov_mult,
                        )
                        pred_df, _ = run_hybrid_elo(games, params)
                        p = np.clip(pred_df["pred_home_win_prob"].to_numpy(), 1e-6, 1 - 1e-6)
                        score = log_loss(y[mask], p[mask])
                        if best is None or score < best[0]:
                            best = (score, params)

    assert best is not None
    return best[1]


def tune_hybrid_elo_order_robust(
    games: pd.DataFrame,
    n_shuffles: int = 40,
    warmup_games: int = 150,
    seed: int = 42,
) -> HybridEloParams:
    # Smaller grid than sequential tuning; objective is mean prequential log-loss across random orders.
    grid_k = [6, 8, 10, 12]
    grid_hfa = [35, 40, 45, 50]
    grid_xg_scale = [0.4, 0.5, 0.7, 1.0]
    grid_w_result = [0.5, 0.6, 0.7]
    grid_mov = [True, False]

    best: tuple[float, HybridEloParams] | None = None
    n_shuffles = max(5, int(n_shuffles))

    for k in grid_k:
        for hfa in grid_hfa:
            for xg_scale in grid_xg_scale:
                for w_result in grid_w_result:
                    for mov_mult in grid_mov:
                        params = HybridEloParams(k=k, hfa=hfa, xg_scale=xg_scale, w_result=w_result, mov_mult=mov_mult)
                        scores = []
                        for i in range(n_shuffles):
                            shuffled = games.sample(frac=1.0, random_state=seed + i).reset_index(drop=True)
                            pred_df, _ = run_hybrid_elo(shuffled, params)
                            y = pred_df["home_win"].to_numpy()
                            p = np.clip(pred_df["pred_home_win_prob"].to_numpy(), 1e-6, 1 - 1e-6)
                            # Burn-in removes the highly prior-driven earliest predictions in each random order.
                            start = min(max(0, warmup_games), len(pred_df) - 1)
                            scores.append(log_loss(y[start:], p[start:]))
                        score = float(np.mean(scores))
                        if best is None or score < best[0]:
                            best = (score, params)

    assert best is not None
    return best[1]


def run_hybrid_elo_shuffle_ensemble(
    games: pd.DataFrame,
    params: HybridEloParams,
    n_shuffles: int = 300,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float], pd.DataFrame]:
    teams = sorted(set(games["home_team"]).union(set(games["away_team"])))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_games = len(games)
    n_shuffles = max(10, int(n_shuffles))

    rating_samples = np.zeros((n_shuffles, len(teams)), dtype=float)
    game_prob_samples = np.zeros((n_shuffles, n_games), dtype=float)
    game_to_idx = {g: i for i, g in enumerate(games["game_id"].tolist())}

    for i in range(n_shuffles):
        shuffled = games.sample(frac=1.0, random_state=seed + i).reset_index(drop=True)
        pred_df_shuf, ratings = run_hybrid_elo(shuffled, params)
        rvec = np.array([ratings[t] for t in teams], dtype=float)
        rating_samples[i, :] = rvec
        # Leak-free: probability for each game is taken from the pre-game prediction
        # at the point that game appears in this shuffled season order.
        probs = np.zeros(n_games, dtype=float)
        for row in pred_df_shuf.itertuples(index=False):
            probs[game_to_idx[row.game_id]] = row.home_win_prob
        game_prob_samples[i, :] = probs

    rating_mean = rating_samples.mean(axis=0)
    rating_std = rating_samples.std(axis=0)
    rating_mean_map = {t: float(rating_mean[j]) for j, t in enumerate(teams)}
    rating_std_map = {t: float(rating_std[j]) for j, t in enumerate(teams)}

    rating_samples_df = pd.DataFrame(rating_samples, columns=teams)
    rating_samples_df.insert(0, "shuffle_id", np.arange(1, n_shuffles + 1))

    game_prob_df = games[["game_id", "game_num", "home_team", "away_team", "home_win"]].copy()
    game_prob_df["home_win_prob"] = game_prob_samples.mean(axis=0)
    game_prob_df["home_win_prob_std"] = game_prob_samples.std(axis=0)
    game_prob_df["home_win_prob_p05"] = np.quantile(game_prob_samples, 0.05, axis=0)
    game_prob_df["home_win_prob_p95"] = np.quantile(game_prob_samples, 0.95, axis=0)
    game_prob_df["away_win_prob"] = 1.0 - game_prob_df["home_win_prob"]

    order_sensitivity = pd.DataFrame(
        {
            "metric": [
                "n_shuffles",
                "mean_game_prob_std",
                "median_game_prob_std",
                "max_game_prob_std",
                "mean_team_rating_std",
                "max_team_rating_std",
            ],
            "value": [
                float(n_shuffles),
                float(game_prob_df["home_win_prob_std"].mean()),
                float(game_prob_df["home_win_prob_std"].median()),
                float(game_prob_df["home_win_prob_std"].max()),
                float(np.mean(rating_std)),
                float(np.max(rating_std)),
            ],
        }
    )
    return game_prob_df, rating_samples_df, rating_mean_map, rating_std_map, order_sensitivity


def build_power_rankings(
    team_summary: pd.DataFrame,
    final_ratings: dict[str, float],
    rating_std: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = team_summary.copy()
    out["elo_final"] = out.index.to_series().map(final_ratings)
    out["elo_std"] = 0.0 if rating_std is None else out.index.to_series().map(rating_std).fillna(0.0)

    def zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std(ddof=0)

    out["power_score"] = (
        0.70 * zscore(out["elo_final"])
        + 0.20 * zscore(out["xGDpg"])
        + 0.10 * zscore(out["PCT"])
    )
    out = out.sort_values("power_score", ascending=False).reset_index().rename(columns={"index": "team"})
    out["power_rank"] = np.arange(1, len(out) + 1)
    return out[
        [
            "power_rank",
            "team",
            "power_score",
            "elo_final",
            "elo_std",
            "PCT",
            "PTS",
            "GD",
            "xGD",
            "xGDpg",
            "GF",
            "GA",
            "xGF",
            "xGA",
        ]
    ]


def compute_line_disparity(df: pd.DataFrame) -> pd.DataFrame:
    home = df[
        ["home_team", "home_off_line", "away_def_pairing", "toi", "home_xg"]
    ].rename(
        columns={
            "home_team": "team",
            "home_off_line": "off_line",
            "away_def_pairing": "opp_def",
            "home_xg": "xg",
        }
    )
    away = df[
        ["away_team", "away_off_line", "home_def_pairing", "toi", "away_xg"]
    ].rename(
        columns={
            "away_team": "team",
            "away_off_line": "off_line",
            "home_def_pairing": "opp_def",
            "away_xg": "xg",
        }
    )
    lines = pd.concat([home, away], ignore_index=True)
    lines = lines[lines["off_line"].isin(["first_off", "second_off"])].copy()
    lines = lines[lines["opp_def"].isin(["first_def", "second_def", "empty_net_line"])].copy()

    raw = lines.groupby(["team", "off_line"]).agg(xg=("xg", "sum"), toi=("toi", "sum"))
    raw["xg60"] = raw["xg"] / (raw["toi"] / 60.0)
    raw_p = raw.reset_index().pivot(index="team", columns="off_line", values="xg60")
    raw_p = raw_p.rename(columns={"first_off": "first_xg60_raw", "second_off": "second_xg60_raw"})
    raw_p["ratio_raw"] = raw_p["first_xg60_raw"] / raw_p["second_xg60_raw"]

    mix = lines.groupby("opp_def")["toi"].sum()
    mix = mix / mix.sum()

    cell = lines.groupby(["team", "off_line", "opp_def"]).agg(xg=("xg", "sum"), toi=("toi", "sum")).reset_index()
    cell["xg60"] = cell["xg"] / (cell["toi"] / 60.0)
    lg = (
        cell.groupby(["off_line", "opp_def"])
        .apply(lambda g: g["xg"].sum() / (g["toi"].sum() / 60.0), include_groups=False)
        .rename("league_xg60")
        .reset_index()
    )

    all_cells = []
    for team in sorted(lines["team"].unique()):
        for off_line in ["first_off", "second_off"]:
            for opp_def in mix.index:
                all_cells.append((team, off_line, opp_def))
    all_cells = pd.DataFrame(all_cells, columns=["team", "off_line", "opp_def"])
    all_cells = all_cells.merge(cell[["team", "off_line", "opp_def", "xg60"]], on=["team", "off_line", "opp_def"], how="left")
    all_cells = all_cells.merge(lg, on=["off_line", "opp_def"], how="left")
    all_cells["xg60_filled"] = all_cells["xg60"].fillna(all_cells["league_xg60"])
    all_cells = all_cells.merge(mix.rename("w"), left_on="opp_def", right_index=True, how="left")

    adj = (
        all_cells.groupby(["team", "off_line"])
        .apply(lambda g: (g["xg60_filled"] * g["w"]).sum(), include_groups=False)
        .rename("xg60_adj")
        .reset_index()
    )
    adj_p = adj.pivot(index="team", columns="off_line", values="xg60_adj")
    adj_p = adj_p.rename(columns={"first_off": "first_xg60_adj", "second_off": "second_xg60_adj"})
    adj_p["ratio_adj"] = adj_p["first_xg60_adj"] / adj_p["second_xg60_adj"]

    out = raw_p.join(adj_p, how="inner").reset_index()
    out = out.sort_values("ratio_adj", ascending=False).reset_index(drop=True)
    out["disparity_rank"] = np.arange(1, len(out) + 1)
    return out[
        [
            "disparity_rank",
            "team",
            "first_xg60_raw",
            "second_xg60_raw",
            "ratio_raw",
            "first_xg60_adj",
            "second_xg60_adj",
            "ratio_adj",
        ]
    ]


def pick_matchup_columns(df: pd.DataFrame) -> tuple[str, str]:
    lc = {c.lower(): c for c in df.columns}
    home_candidates = ["home_team", "home", "team_home", "hometeam"]
    away_candidates = ["away_team", "away", "team_away", "awayteam"]
    home_col = next((lc[c] for c in home_candidates if c in lc), None)
    away_col = next((lc[c] for c in away_candidates if c in lc), None)
    if home_col is None or away_col is None:
        raise ValueError("Could not identify home/away team columns in matchup file.")
    return home_col, away_col


def predict_matchups(
    matchups: pd.DataFrame,
    ratings: dict[str, float],
    hfa: float,
    scale: float,
) -> pd.DataFrame:
    out = matchups.copy()
    home_col, away_col = pick_matchup_columns(out)
    out["home_team"] = out[home_col].astype(str)
    out["away_team"] = out[away_col].astype(str)
    out["home_rating"] = out["home_team"].map(ratings)
    out["away_rating"] = out["away_team"].map(ratings)

    missing = out[out["home_rating"].isna() | out["away_rating"].isna()]
    if not missing.empty:
        missing_teams = sorted(set(missing["home_team"]).union(set(missing["away_team"])))
        raise ValueError(f"Matchup teams not found in learned ratings: {missing_teams}")

    out["home_win_prob"] = 1.0 / (1.0 + 10 ** (-((out["home_rating"] + hfa) - out["away_rating"]) / scale))
    out["away_win_prob"] = 1.0 - out["home_win_prob"]
    return out


def predict_matchups_order_robust(
    matchups: pd.DataFrame,
    rating_samples: pd.DataFrame,
    rating_mean: dict[str, float],
    hfa: float,
    scale: float,
) -> pd.DataFrame:
    out = matchups.copy()
    home_col, away_col = pick_matchup_columns(out)
    out["home_team"] = out[home_col].astype(str)
    out["away_team"] = out[away_col].astype(str)

    team_cols = [c for c in rating_samples.columns if c != "shuffle_id"]
    teams = set(team_cols)
    missing = set(out["home_team"]).union(set(out["away_team"])) - teams
    if missing:
        raise ValueError(f"Matchup teams not found in rating samples: {sorted(missing)}")

    sample_matrix = rating_samples[team_cols]

    means = []
    stds = []
    p05s = []
    p95s = []
    mean_from_mean_ratings = []
    for row in out.itertuples(index=False):
        home = getattr(row, "home_team")
        away = getattr(row, "away_team")
        home_samples = sample_matrix[home].to_numpy()
        away_samples = sample_matrix[away].to_numpy()
        p = 1.0 / (1.0 + 10 ** (-((home_samples + hfa) - away_samples) / scale))
        means.append(float(np.mean(p)))
        stds.append(float(np.std(p)))
        p05s.append(float(np.quantile(p, 0.05)))
        p95s.append(float(np.quantile(p, 0.95)))
        pm = 1.0 / (1.0 + 10 ** (-((rating_mean[home] + hfa) - rating_mean[away]) / scale))
        mean_from_mean_ratings.append(float(pm))

    out["home_win_prob"] = means
    out["away_win_prob"] = 1.0 - out["home_win_prob"]
    out["home_win_prob_std"] = stds
    out["home_win_prob_p05"] = p05s
    out["home_win_prob_p95"] = p95s
    out["home_win_prob_from_mean_rating"] = mean_from_mean_ratings
    out["away_win_prob_p05"] = 1.0 - out["home_win_prob_p95"]
    out["away_win_prob_p95"] = 1.0 - out["home_win_prob_p05"]
    return out


def save_phase1c_plot(
    rankings: pd.DataFrame,
    disparity: pd.DataFrame,
    out_png: Path,
) -> None:
    plot_df = rankings.merge(disparity[["team", "ratio_adj"]], on="team", how="inner")
    plot_df = plot_df.rename(columns={"ratio_adj": "line_disparity_ratio"})

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11.5, 7))
    sns.regplot(
        data=plot_df,
        x="line_disparity_ratio",
        y="power_score",
        scatter_kws={"s": 70, "alpha": 0.85},
        line_kws={"color": "#c0392b", "linewidth": 2},
        ax=ax,
    )

    ax.set_title(
        "WHL Team Strength vs Offensive Line Disparity\n(Phase 1c: Are balanced lines linked to stronger teams?)",
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("Offensive Line Disparity (First-Line xG/60 ÷ Second-Line xG/60, adjusted)")
    ax.set_ylabel("Team Power Score (Hybrid: Elo + xGD + PCT)")

    to_label = pd.concat([plot_df.nsmallest(4, "line_disparity_ratio"), plot_df.nlargest(4, "line_disparity_ratio")]).drop_duplicates("team")
    for row in to_label.itertuples(index=False):
        ax.text(row.line_disparity_ratio + 0.003, row.power_score + 0.02, row.team, fontsize=9)

    corr = plot_df["line_disparity_ratio"].corr(plot_df["power_score"])
    ax.text(
        0.02,
        0.97,
        f"Correlation = {corr:.3f}\nLower ratio = more balanced top-2 lines",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def jsonable_metrics(y_true: Iterable[int], p_pred: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.clip(np.asarray(list(p_pred), dtype=float), 1e-6, 1 - 1e-6)
    return {
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def main() -> None:
    args = parse_args()
    data_csv = Path(args.data_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_validate(data_csv)
    games = aggregate_games(df)
    team_summary = build_team_summary(games)
    mode = "order_robust_shuffle_ensemble" if args.order_robust else "sequential_single_order"

    if args.order_robust:
        if args.tune_elo:
            params = tune_hybrid_elo_order_robust(games, n_shuffles=args.tune_shuffles, seed=args.seed)
        else:
            params = HybridEloParams(k=8, hfa=40, xg_scale=0.4, w_result=0.6, mov_mult=True)
        pred_df, rating_samples_df, final_ratings, rating_std, order_sensitivity_df = run_hybrid_elo_shuffle_ensemble(
            games=games,
            params=params,
            n_shuffles=args.n_shuffles,
            seed=args.seed,
        )
        rankings = build_power_rankings(team_summary, final_ratings, rating_std=rating_std)
    else:
        if args.tune_elo:
            params = tune_hybrid_elo(games)
        else:
            params = HybridEloParams(k=8, hfa=40, xg_scale=0.5, w_result=0.6, mov_mult=True)
        pred_df, final_ratings = run_hybrid_elo(games, params)
        rating_samples_df = None
        order_sensitivity_df = None
        rankings = build_power_rankings(team_summary, final_ratings)

    disparity = compute_line_disparity(df)
    top10_disparity = disparity.head(10).copy()

    eval_mask = rolling_eval_mask(len(games))
    metrics_cv = jsonable_metrics(games.loc[eval_mask, "home_win"], pred_df.loc[eval_mask, "home_win_prob"])
    metrics_full = jsonable_metrics(games["home_win"], pred_df["home_win_prob"])
    baseline_cv = float(games.loc[eval_mask, "home_win"].mean())
    baseline_full = float(games["home_win"].mean())
    baseline_cv_metrics = jsonable_metrics(
        games.loc[eval_mask, "home_win"],
        np.repeat(baseline_cv, int(eval_mask.sum())),
    )
    baseline_full_metrics = jsonable_metrics(
        games["home_win"],
        np.repeat(baseline_full, len(games)),
    )

    rankings.to_csv(output_dir / "phase1a_power_rankings.csv", index=False)
    pred_df.to_csv(output_dir / "phase1a_sequential_game_predictions.csv", index=False)
    disparity.to_csv(output_dir / "phase1b_line_disparity_all_teams.csv", index=False)
    top10_disparity.to_csv(output_dir / "phase1b_top10_line_disparity.csv", index=False)
    save_phase1c_plot(rankings, disparity, output_dir / "phase1c_disparity_vs_team_strength.png")
    if rating_samples_df is not None:
        rating_samples_df.to_csv(output_dir / "phase1a_rating_samples_by_shuffle.csv", index=False)
    if order_sensitivity_df is not None:
        order_sensitivity_df.to_csv(output_dir / "phase1a_order_sensitivity_summary.csv", index=False)

    round1_path = None
    if args.matchups_csv is not None:
        matchups = pd.read_csv(args.matchups_csv)
        if args.order_robust:
            assert rating_samples_df is not None
            round1 = predict_matchups_order_robust(
                matchups=matchups,
                rating_samples=rating_samples_df,
                rating_mean=final_ratings,
                hfa=params.hfa,
                scale=params.scale,
            )
        else:
            round1 = predict_matchups(matchups, final_ratings, hfa=params.hfa, scale=params.scale)
        round1_path = output_dir / "phase1a_round1_matchup_probabilities.csv"
        round1.to_csv(round1_path, index=False)

    run_summary = {
        "modeling_mode": mode,
        "data_csv": str(data_csv),
        "rows": int(len(df)),
        "games": int(len(games)),
        "teams": int(pd.concat([games["home_team"], games["away_team"]], ignore_index=True).nunique()),
        "random_seed": int(args.seed),
        "hybrid_elo_params": {
            "k": params.k,
            "hfa": params.hfa,
            "xg_scale": params.xg_scale,
            "w_result": params.w_result,
            "mov_mult": params.mov_mult,
            "scale": params.scale,
        },
        "order_robust_settings": {
            "enabled": bool(args.order_robust),
            "n_shuffles": int(args.n_shuffles) if args.order_robust else None,
            "tune_shuffles": int(args.tune_shuffles) if (args.order_robust and args.tune_elo) else None,
        },
        "cv_metrics": metrics_cv,
        "full_metrics": metrics_full,
        "baseline_cv_metrics": baseline_cv_metrics,
        "baseline_full_metrics": baseline_full_metrics,
        "improvement_vs_baseline": {
            "cv_logloss_delta": float(baseline_cv_metrics["logloss"] - metrics_cv["logloss"]),
            "cv_brier_delta": float(baseline_cv_metrics["brier"] - metrics_cv["brier"]),
            "full_logloss_delta": float(baseline_full_metrics["logloss"] - metrics_full["logloss"]),
            "full_brier_delta": float(baseline_full_metrics["brier"] - metrics_full["brier"]),
        },
        "outputs": {
            "phase1a_power_rankings": str(output_dir / "phase1a_power_rankings.csv"),
            "phase1a_sequential_game_predictions": str(output_dir / "phase1a_sequential_game_predictions.csv"),
            "phase1a_rating_samples_by_shuffle": str(output_dir / "phase1a_rating_samples_by_shuffle.csv") if args.order_robust else None,
            "phase1a_order_sensitivity_summary": str(output_dir / "phase1a_order_sensitivity_summary.csv") if args.order_robust else None,
            "phase1b_all_disparity": str(output_dir / "phase1b_line_disparity_all_teams.csv"),
            "phase1b_top10_disparity": str(output_dir / "phase1b_top10_line_disparity.csv"),
            "phase1c_png": str(output_dir / "phase1c_disparity_vs_team_strength.png"),
            "phase1a_round1_probabilities": str(round1_path) if round1_path else None,
        },
    }
    (output_dir / "phase1_run_summary.json").write_text(json.dumps(run_summary, indent=2))

    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
