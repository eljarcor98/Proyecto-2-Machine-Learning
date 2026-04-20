from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

XG_FEATURES = [
    "distance",
    "angle",
    "is_header",
    "is_big_chance",
    "is_penalty",
    "is_counter",
]
XG_EXTRA_FEATURES = [
    "is_right_foot",
    "is_left_foot",
    "is_from_corner",
    "is_volley",
    "is_first_touch",
]
XG_MODEL_FEATURES = XG_FEATURES + XG_EXTRA_FEATURES
XG_POLY_DEGREES = [1, 2, 3]
XG_LOGISTIC_C_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
XG_SCALED_FEATURES = ["distance", "angle"]
HOME_FEATURES = ["hs", "hst", "hc", "hf", "hy", "hr"]
AWAY_FEATURES = ["as_", "ast", "ac", "af", "ay", "ar"]
MATCH_ODDS_FEATURES = [
    "b365h",
    "b365d",
    "b365a",
    "bwh",
    "bwd",
    "bwa",
    "maxh",
    "maxd",
    "maxa",
    "avgh",
    "avgd",
    "avga",
]
MATCH_IMPLIED_PROBS = ["implied_prob_h", "implied_prob_d", "implied_prob_a"]
MATCH_NUMERIC_FEATURES = HOME_FEATURES + AWAY_FEATURES + MATCH_ODDS_FEATURES + MATCH_IMPLIED_PROBS
MATCH_CATEGORICAL_FEATURES = ["referee"]


st.set_page_config(
    page_title="Premier League Match Lab",
    page_icon="PL",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(205, 44, 64, 0.14), transparent 28%),
            radial-gradient(circle at top left, rgba(11, 78, 162, 0.10), transparent 24%),
            linear-gradient(180deg, #f5f1e8 0%, #ece5d7 100%);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 252, 246, 0.88);
        border: 1px solid rgba(22, 30, 50, 0.10);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        box-shadow: 0 10px 30px rgba(46, 30, 20, 0.07);
    }
    .panel {
        background: rgba(255, 252, 246, 0.84);
        border: 1px solid rgba(22, 30, 50, 0.10);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 14px 38px rgba(46, 30, 20, 0.08);
    }
    .eyebrow {
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.76rem;
        color: #9d3c2f;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .hero-title {
        font-size: 2.4rem;
        line-height: 1.0;
        color: #132033;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .hero-copy {
        color: #3a4251;
        font-size: 1rem;
        max-width: 48rem;
    }
    .tag-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }
    .tag {
        background: #132033;
        color: #fff9f1;
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_matches() -> pd.DataFrame:
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    matches["date"] = pd.to_datetime(matches["date"], format="%d/%m/%Y")
    matches["season_label"] = matches["date"].dt.strftime("%d %b %Y")
    return matches


@st.cache_data(show_spinner=False)
def load_shot_features() -> pd.DataFrame:
    shots = pd.read_csv(PROCESSED_DIR / "shots_features.csv")
    shots["is_goal"] = shots["is_goal"].astype(int)
    return shots


@st.cache_data(show_spinner=False)
def load_shot_events() -> pd.DataFrame:
    usecols = [
        "id",
        "match_id",
        "minute",
        "period",
        "team_name",
        "player_name",
        "x",
        "y",
        "is_shot",
        "is_goal",
    ]
    events = pd.read_csv(RAW_DIR / "events.csv", usecols=usecols)
    events = events[events["is_shot"] == True].copy()
    events["is_goal"] = events["is_goal"].astype(int)

    direction = (
        events.groupby(["match_id", "team_name", "period"])["x"]
        .mean()
        .reset_index(name="avg_x")
    )
    direction["should_flip"] = direction["avg_x"] < 50
    events = events.merge(
        direction[["match_id", "team_name", "period", "should_flip"]],
        on=["match_id", "team_name", "period"],
        how="left",
    )
    events["x_plot"] = np.where(events["should_flip"], 100 - events["x"], events["x"])
    events["y_plot"] = np.where(events["should_flip"], 100 - events["y"], events["y"])
    return events


@st.cache_data(show_spinner=False)
def load_standings() -> pd.DataFrame:
    standings = pd.read_csv(DATA_DIR / "standings.csv")
    parsed = standings["standings"].apply(ast.literal_eval).apply(pd.Series)
    parsed = parsed.rename(columns={"team": "team_name"})
    return parsed


@st.cache_data(show_spinner=False)
def build_shot_dataset() -> pd.DataFrame:
    shots = load_shot_features()
    shot_events = load_shot_events()
    merged = shots.merge(
        shot_events[
            [
                "id",
                "match_id",
                "minute",
                "team_name",
                "player_name",
                "x_plot",
                "y_plot",
            ]
        ],
        on=["id", "match_id", "team_name"],
        how="left",
    )
    return merged


@st.cache_data(show_spinner=False)
def load_match_events() -> pd.DataFrame:
    usecols = [
        "id",
        "match_id",
        "minute",
        "second",
        "period",
        "event_type",
        "outcome",
        "team_name",
        "player_name",
        "x",
        "y",
        "end_x",
        "end_y",
        "is_shot",
        "is_goal",
    ]
    events = pd.read_csv(RAW_DIR / "events.csv", usecols=usecols)
    events = events.dropna(subset=["x", "y"]).copy()
    events["second"] = pd.to_numeric(events["second"], errors="coerce").fillna(0)
    events["minute"] = pd.to_numeric(events["minute"], errors="coerce").fillna(0)
    events["is_shot"] = events["is_shot"].astype(bool)
    events["is_goal"] = events["is_goal"].astype(bool)
    period_offsets = {
        "FirstHalf": 0,
        "SecondHalf": 45 * 60,
        "FirstPeriodOfExtraTime": 90 * 60,
        "SecondPeriodOfExtraTime": 105 * 60,
        "PenaltyShootout": 120 * 60,
        "PostGame": 120 * 60,
        "PreMatch": 0,
    }
    events["clock_seconds"] = (
        events["minute"] * 60
        + events["second"]
        + events["period"].map(period_offsets).fillna(0)
    )
    direction = (
        events.groupby(["match_id", "team_name", "period"])["x"]
        .mean()
        .reset_index(name="avg_x")
    )
    direction["should_flip"] = direction["avg_x"] < 50
    events = events.merge(
        direction[["match_id", "team_name", "period", "should_flip"]],
        on=["match_id", "team_name", "period"],
        how="left",
    )
    events["x_plot"] = np.where(events["should_flip"], 100 - events["x"], events["x"])
    events["y_plot"] = np.where(events["should_flip"], 100 - events["y"], events["y"])
    events["end_x_plot"] = np.where(
        events["end_x"].notna(),
        np.where(events["should_flip"], 100 - events["end_x"], events["end_x"]),
        events["x_plot"],
    )
    events["end_y_plot"] = np.where(
        events["end_y"].notna(),
        np.where(events["should_flip"], 100 - events["end_y"], events["end_y"]),
        events["y_plot"],
    )
    events["outcome_label"] = events["outcome"].fillna("Unknown")
    events["player_name"] = events["player_name"].fillna("Unknown")
    events["event_label"] = (
        events["minute"].astype(int).astype(str).str.zfill(2)
        + ":"
        + events["second"].astype(int).astype(str).str.zfill(2)
        + " | "
        + events["team_name"].fillna("Unknown")
        + " | "
        + events["event_type"].fillna("Unknown")
    )
    events = events.sort_values(["match_id", "clock_seconds", "id"]).reset_index(drop=True)
    return events


def build_threshold_rows(scores: np.ndarray, labels: pd.Series) -> list[dict[str, float]]:
    rows = []
    labels = labels.reset_index(drop=True)
    for threshold in np.arange(0.05, 0.55, 0.05):
        preds = (scores > threshold).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / len(labels)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return rows


def merge_threshold_frames(left: pd.DataFrame, right: pd.DataFrame, left_prefix: str, right_prefix: str) -> pd.DataFrame:
    left_frame = left.copy().add_prefix(f"{left_prefix}_")
    right_frame = right.copy().add_prefix(f"{right_prefix}_")
    merged = left_frame.merge(
        right_frame,
        left_on=f"{left_prefix}_threshold",
        right_on=f"{right_prefix}_threshold",
        how="inner",
    )
    return merged.rename(columns={f"{left_prefix}_threshold": "threshold"}).drop(columns=[f"{right_prefix}_threshold"])


@st.cache_resource(show_spinner=False)
def fit_xg_model() -> dict[str, object]:
    shots = load_shot_features().dropna(subset=XG_MODEL_FEATURES + ["is_goal"]).copy()
    X = shots[XG_MODEL_FEATURES].astype(float)
    y = shots["is_goal"].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_test, X_reval, y_test, y_reval = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    candidates: list[dict[str, object]] = []
    for degree in XG_POLY_DEGREES:
        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("linear", LinearRegression()),
            ]
        )
        model.fit(X_train, y_train)

        train_scores = np.clip(model.predict(X_train), 0, 1)
        test_scores = np.clip(model.predict(X_test), 0, 1)
        reval_scores = np.clip(model.predict(X_reval), 0, 1)

        test_thresholds = pd.DataFrame(build_threshold_rows(test_scores, y_test))
        reval_thresholds = pd.DataFrame(build_threshold_rows(reval_scores, y_reval))
        best_test_row = test_thresholds.sort_values(
            ["f1", "recall", "precision", "accuracy", "threshold"],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        best_threshold = float(best_test_row["threshold"])
        reval_best_row = reval_thresholds.loc[reval_thresholds["threshold"] == best_threshold].iloc[0]

        candidates.append(
            {
                "degree": degree,
                "model": model,
                "train_rmse": math.sqrt(mean_squared_error(y_train, train_scores)),
                "test_rmse": math.sqrt(mean_squared_error(y_test, test_scores)),
                "reval_rmse": math.sqrt(mean_squared_error(y_reval, reval_scores)),
                "train_r2": r2_score(y_train, train_scores),
                "test_r2": r2_score(y_test, test_scores),
                "reval_r2": r2_score(y_reval, reval_scores),
                "train_mean_xg": float(train_scores.mean()),
                "test_mean_xg": float(test_scores.mean()),
                "reval_mean_xg": float(reval_scores.mean()),
                "best_threshold": best_threshold,
                "test_f1": float(best_test_row["f1"]),
                "reval_f1": float(reval_best_row["f1"]),
                "overfit_gap_rmse": math.sqrt(mean_squared_error(y_reval, reval_scores))
                - math.sqrt(mean_squared_error(y_train, train_scores)),
                "overfit_gap_r2": r2_score(y_train, train_scores) - r2_score(y_reval, reval_scores),
                "test_thresholds": test_thresholds,
                "reval_thresholds": reval_thresholds,
            }
        )

    candidate_table = pd.DataFrame(
        [
            {
                "degree": candidate["degree"],
                "train_rmse": candidate["train_rmse"],
                "test_rmse": candidate["test_rmse"],
                "revalidation_rmse": candidate["reval_rmse"],
                "train_r2": candidate["train_r2"],
                "test_r2": candidate["test_r2"],
                "revalidation_r2": candidate["reval_r2"],
                "best_threshold": candidate["best_threshold"],
                "test_f1": candidate["test_f1"],
                "revalidation_f1": candidate["reval_f1"],
                "overfit_gap_rmse": candidate["overfit_gap_rmse"],
                "overfit_gap_r2": candidate["overfit_gap_r2"],
            }
            for candidate in candidates
        ]
    )
    best_idx = candidate_table.sort_values(
        ["revalidation_rmse", "revalidation_f1", "overfit_gap_rmse", "degree"],
        ascending=[True, False, True, True],
    ).index[0]
    best_candidate = candidates[int(best_idx)]

    return {
        "model": best_candidate["model"],
        "best_degree": best_candidate["degree"],
        "best_threshold": best_candidate["best_threshold"],
        "candidate_table": candidate_table,
        "candidates": candidates,
        "test_thresholds": best_candidate["test_thresholds"],
        "reval_thresholds": best_candidate["reval_thresholds"],
        "y_test": y_test.reset_index(drop=True),
        "y_reval": y_reval.reset_index(drop=True),
    }


@st.cache_resource(show_spinner=False)
def fit_xg_logistic_model() -> dict[str, object]:
    shots = load_shot_features().dropna(subset=XG_MODEL_FEATURES + ["is_goal"]).copy()
    X = shots[XG_MODEL_FEATURES].astype(float)
    y = shots["is_goal"].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=123, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=123, stratify=y_temp
    )

    preprocessor = ColumnTransformer(
        [
            ("scale_distance_angle", StandardScaler(), XG_SCALED_FEATURES),
        ],
        remainder="passthrough",
    )

    candidates: list[dict[str, object]] = []
    for c_value in XG_LOGISTIC_C_VALUES:
        model = Pipeline(
            [
                ("prep", preprocessor),
                (
                    "logistic",
                    LogisticRegression(
                        class_weight="balanced",
                        C=c_value,
                        max_iter=2000,
                        random_state=123,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

        train_probs = model.predict_proba(X_train)[:, 1]
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]

        val_thresholds = pd.DataFrame(build_threshold_rows(val_probs, y_val))
        test_thresholds = pd.DataFrame(build_threshold_rows(test_probs, y_test))
        best_val_row = val_thresholds.sort_values(
            ["f1", "recall", "precision", "accuracy", "threshold"],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        best_threshold = float(best_val_row["threshold"])
        test_best_row = test_thresholds.loc[test_thresholds["threshold"] == best_threshold].iloc[0]

        train_preds = (train_probs > best_threshold).astype(int)
        val_preds = (val_probs > best_threshold).astype(int)
        test_preds = (test_probs > best_threshold).astype(int)

        candidates.append(
            {
                "C": c_value,
                "model": model,
                "best_threshold": best_threshold,
                "train_accuracy": accuracy_score(y_train, train_preds),
                "validation_accuracy": accuracy_score(y_val, val_preds),
                "test_accuracy": accuracy_score(y_test, test_preds),
                "train_precision": precision_score(y_train, train_preds, zero_division=0),
                "validation_precision": precision_score(y_val, val_preds, zero_division=0),
                "test_precision": precision_score(y_test, test_preds, zero_division=0),
                "train_recall": recall_score(y_train, train_preds, zero_division=0),
                "validation_recall": recall_score(y_val, val_preds, zero_division=0),
                "test_recall": recall_score(y_test, test_preds, zero_division=0),
                "train_f1": f1_score(y_train, train_preds, zero_division=0),
                "validation_f1": float(best_val_row["f1"]),
                "test_f1": float(test_best_row["f1"]),
                "train_auc": roc_auc_score(y_train, train_probs),
                "validation_auc": roc_auc_score(y_val, val_probs),
                "test_auc": roc_auc_score(y_test, test_probs),
                "overfit_gap_auc": roc_auc_score(y_train, train_probs) - roc_auc_score(y_test, test_probs),
                "overfit_gap_f1": f1_score(y_train, train_preds, zero_division=0) - float(test_best_row["f1"]),
                "validation_thresholds": val_thresholds,
                "test_thresholds": test_thresholds,
            }
        )

    candidate_table = pd.DataFrame(
        [
            {
                "C": candidate["C"],
                "best_threshold": candidate["best_threshold"],
                "train_accuracy": candidate["train_accuracy"],
                "validation_accuracy": candidate["validation_accuracy"],
                "test_accuracy": candidate["test_accuracy"],
                "train_precision": candidate["train_precision"],
                "validation_precision": candidate["validation_precision"],
                "test_precision": candidate["test_precision"],
                "train_recall": candidate["train_recall"],
                "validation_recall": candidate["validation_recall"],
                "test_recall": candidate["test_recall"],
                "train_f1": candidate["train_f1"],
                "validation_f1": candidate["validation_f1"],
                "test_f1": candidate["test_f1"],
                "train_auc": candidate["train_auc"],
                "validation_auc": candidate["validation_auc"],
                "test_auc": candidate["test_auc"],
                "overfit_gap_auc": candidate["overfit_gap_auc"],
                "overfit_gap_f1": candidate["overfit_gap_f1"],
            }
            for candidate in candidates
        ]
    )
    best_idx = candidate_table.sort_values(
        ["validation_auc", "validation_f1", "overfit_gap_auc", "C"],
        ascending=[False, False, True, True],
    ).index[0]
    best_candidate = candidates[int(best_idx)]

    return {
        "model": best_candidate["model"],
        "best_c": best_candidate["C"],
        "best_threshold": best_candidate["best_threshold"],
        "candidate_table": candidate_table,
        "candidates": candidates,
        "validation_thresholds": best_candidate["validation_thresholds"],
        "test_thresholds": best_candidate["test_thresholds"],
    }


@st.cache_data(show_spinner=False)
def compute_threshold_metrics() -> pd.DataFrame:
    payload = fit_xg_model()
    return merge_threshold_frames(payload["test_thresholds"], payload["reval_thresholds"], "test", "reval")


@st.cache_data(show_spinner=False)
def compute_logistic_threshold_metrics() -> pd.DataFrame:
    payload = fit_xg_logistic_model()
    return merge_threshold_frames(payload["validation_thresholds"], payload["test_thresholds"], "validation", "test")


@st.cache_data(show_spinner=False)
def attach_xg_scores() -> pd.DataFrame:
    shot_map = build_shot_dataset().dropna(subset=XG_MODEL_FEATURES).copy()
    linear_model = fit_xg_model()["model"]
    logistic_model = fit_xg_logistic_model()["model"]
    shot_map["xg_linear"] = np.clip(linear_model.predict(shot_map[XG_MODEL_FEATURES].astype(float)), 0, 1)
    shot_map["xg_logistic"] = np.clip(
        logistic_model.predict_proba(shot_map[XG_MODEL_FEATURES].astype(float))[:, 1],
        0,
        1,
    )
    return shot_map


@st.cache_resource(show_spinner=False)
def fit_goal_models() -> dict[str, object]:
    matches = load_matches().copy()
    matches = matches.dropna(subset=MATCH_NUMERIC_FEATURES + MATCH_CATEGORICAL_FEATURES + ["total_goals", "ftr"]).copy()

    X = matches[MATCH_NUMERIC_FEATURES + MATCH_CATEGORICAL_FEATURES].copy()
    y_total_goals = matches["total_goals"].astype(float)
    y_result = matches["ftr"].astype(str)

    X_train, X_temp, y_total_train, y_total_temp, y_result_train, y_result_temp = train_test_split(
        X,
        y_total_goals,
        y_result,
        test_size=0.30,
        random_state=42,
        stratify=y_result,
    )
    X_val, X_test, y_total_val, y_total_test, y_result_val, y_result_test = train_test_split(
        X_temp,
        y_total_temp,
        y_result_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_result_temp,
    )

    linear_preprocessor = ColumnTransformer(
        [("numeric", StandardScaler(), MATCH_NUMERIC_FEATURES)],
        remainder="drop",
    )
    logistic_preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), MATCH_NUMERIC_FEATURES),
            ("referee", OneHotEncoder(handle_unknown="ignore"), MATCH_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    total_goals_model = Pipeline(
        [
            ("prep", linear_preprocessor),
            ("linear", LinearRegression()),
        ]
    )
    total_goals_model.fit(X_train, y_total_train)

    result_model = Pipeline(
        [
            ("prep", logistic_preprocessor),
            ("logistic", LogisticRegression(max_iter=3000, random_state=42)),
        ]
    )
    result_model.fit(X_train, y_result_train)

    val_goal_pred = np.clip(total_goals_model.predict(X_val), 0, 8)
    test_goal_pred = np.clip(total_goals_model.predict(X_test), 0, 8)

    val_result_pred = result_model.predict(X_val)
    test_result_pred = result_model.predict(X_test)
    val_result_prob = result_model.predict_proba(X_val)
    test_result_prob = result_model.predict_proba(X_test)

    b365_val_pred = X_val[["b365h", "b365d", "b365a"]].idxmin(axis=1).map(
        {"b365h": "H", "b365d": "D", "b365a": "A"}
    )
    b365_test_pred = X_test[["b365h", "b365d", "b365a"]].idxmin(axis=1).map(
        {"b365h": "H", "b365d": "D", "b365a": "A"}
    )

    result_classes = list(result_model.named_steps["logistic"].classes_)

    return {
        "total_goals_model": total_goals_model,
        "result_model": result_model,
        "val_goal_rmse": math.sqrt(mean_squared_error(y_total_val, val_goal_pred)),
        "test_goal_rmse": math.sqrt(mean_squared_error(y_total_test, test_goal_pred)),
        "val_goal_r2": r2_score(y_total_val, val_goal_pred),
        "test_goal_r2": r2_score(y_total_test, test_goal_pred),
        "val_result_accuracy": accuracy_score(y_result_val, val_result_pred),
        "test_result_accuracy": accuracy_score(y_result_test, test_result_pred),
        "val_result_f1_macro": f1_score(y_result_val, val_result_pred, average="macro"),
        "test_result_f1_macro": f1_score(y_result_test, test_result_pred, average="macro"),
        "val_result_classes": result_classes,
        "test_result_classes": result_classes,
        "val_result_prob": val_result_prob,
        "test_result_prob": test_result_prob,
        "val_b365_accuracy": accuracy_score(y_result_val, b365_val_pred),
        "test_b365_accuracy": accuracy_score(y_result_test, b365_test_pred),
        "referees": sorted(matches["referee"].dropna().unique().tolist()),
        "avg_odds": {
            feature: float(matches[feature].mean()) for feature in MATCH_ODDS_FEATURES
        },
    }


@st.cache_data(show_spinner=False)
def team_profiles() -> dict[str, pd.DataFrame]:
    matches = load_matches().copy()

    home_attack = (
        matches.groupby("home_team")[HOME_FEATURES + ["fthg"]]
        .mean()
        .rename(columns={"fthg": "goals_for"})
    )
    home_defense = (
        matches.groupby("home_team")[AWAY_FEATURES + ["ftag"]]
        .mean()
        .rename(
            columns={
                "ftag": "goals_against",
            }
        )
    )
    away_attack = (
        matches.groupby("away_team")[AWAY_FEATURES + ["ftag"]]
        .mean()
        .rename(
            columns={
                "as_": "as_",
                "ast": "ast",
                "ac": "ac",
                "af": "af",
                "ay": "ay",
                "ar": "ar",
                "ftag": "goals_for",
            }
        )
    )
    away_defense = (
        matches.groupby("away_team")[HOME_FEATURES + ["fthg"]]
        .mean()
        .rename(
            columns={
                "fthg": "goals_against",
            }
        )
    )

    return {
        "home_attack": home_attack,
        "home_defense": home_defense,
        "away_attack": away_attack,
        "away_defense": away_defense,
    }


def build_matchup_features(home_team: str, away_team: str, referee: str, odds: dict[str, float]) -> pd.Series:
    profiles = team_profiles()
    home_attack = profiles["home_attack"].loc[home_team]
    home_defense = profiles["home_defense"].loc[home_team]
    away_attack = profiles["away_attack"].loc[away_team]
    away_defense = profiles["away_defense"].loc[away_team]

    row = {
        feature: float((home_attack[feature] + away_defense[feature]) / 2)
        for feature in HOME_FEATURES
    }
    row.update(
        {
            feature: float((away_attack[feature] + home_defense[feature]) / 2)
            for feature in AWAY_FEATURES
        }
    )
    row.update(odds)
    row["implied_prob_h"] = 1 / max(odds["b365h"], 0.01)
    row["implied_prob_d"] = 1 / max(odds["b365d"], 0.01)
    row["implied_prob_a"] = 1 / max(odds["b365a"], 0.01)
    row["referee"] = referee
    return pd.Series(row)


def matchup_projection(home_team: str, away_team: str, referee: str, odds: dict[str, float]) -> dict[str, object]:
    models = fit_goal_models()
    features = build_matchup_features(home_team, away_team, referee, odds)
    feature_frame = pd.DataFrame(
        [features],
        columns=MATCH_NUMERIC_FEATURES + MATCH_CATEGORICAL_FEATURES,
    )
    total_goals = float(np.clip(models["total_goals_model"].predict(feature_frame)[0], 0, 8))
    probs_raw = models["result_model"].predict_proba(feature_frame)[0]
    probs = dict(zip(models["test_result_classes"], probs_raw))
    return {
        "total_goals": total_goals,
        "probabilities": probs,
        "features": features,
        "predicted_result": max(probs, key=probs.get),
    }


def draw_pitch() -> go.Figure:
    fig = go.Figure()
    stripe_colors = ["#1f5f4a", "#245f4e", "#1f5f4a", "#245f4e", "#1f5f4a"]
    for idx, color in enumerate(stripe_colors):
        fig.add_shape(
            type="rect",
            x0=idx * 20,
            x1=(idx + 1) * 20,
            y0=0,
            y1=100,
            line=dict(width=0),
            fillcolor=color,
            layer="below",
        )
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color="#f8f4ea", width=2)),
        dict(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color="#f8f4ea", width=2)),
        dict(type="circle", x0=40.85, y0=40.85, x1=59.15, y1=59.15, line=dict(color="#f8f4ea", width=2)),
        dict(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color="#f8f4ea", width=2)),
        dict(type="rect", x0=83.5, y0=21.1, x1=100, y1=78.9, line=dict(color="#f8f4ea", width=2)),
        dict(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color="#f8f4ea", width=2)),
        dict(type="rect", x0=94.5, y0=36.8, x1=100, y1=63.2, line=dict(color="#f8f4ea", width=2)),
    ]
    fig.update_layout(
        shapes=shapes,
        paper_bgcolor="#132033",
        plot_bgcolor="#1b4d3e",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-4, 104], visible=False),
        yaxis=dict(range=[-4, 104], visible=False, scaleanchor="x", scaleratio=1),
        hovermode="closest",
        height=640,
    )
    return fig


def build_match_options(matches: pd.DataFrame) -> dict[str, int]:
    options: dict[str, int] = {}
    for row in matches.sort_values(["date", "time", "id"]).itertuples():
        label = (
            f"{row.home_team} vs {row.away_team} | "
            f"{pd.Timestamp(row.date).strftime('%d %b %Y')} | "
            f"ID {row.id}"
        )
        options[label] = int(row.id)
    return options


def _recent_path_trace(frame: pd.DataFrame) -> go.Scatter:
    xs: list[float | None] = []
    ys: list[float | None] = []
    for row in frame.itertuples():
        xs.extend([row.x_plot, row.end_x_plot, None])
        ys.extend([row.y_plot, row.end_y_plot, None])
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color="rgba(255,249,241,0.32)", width=2),
        hoverinfo="skip",
        name="Recent actions",
    )


def _current_path_trace(event: pd.Series, team_colors: dict[str, str]) -> go.Scatter:
    return go.Scatter(
        x=[event["x_plot"], event["end_x_plot"]],
        y=[event["y_plot"], event["end_y_plot"]],
        mode="lines",
        line=dict(color=team_colors.get(event["team_name"], "#ffb703"), width=5),
        hoverinfo="skip",
        name="Current action",
    )


def _current_start_trace(event: pd.Series, team_colors: dict[str, str]) -> go.Scatter:
    return go.Scatter(
        x=[event["x_plot"]],
        y=[event["y_plot"]],
        mode="markers",
        marker=dict(
            size=13,
            color=team_colors.get(event["team_name"], "#ffb703"),
            line=dict(color="#10231a", width=1.4),
            symbol="circle",
        ),
        customdata=[[
            event["team_name"],
            event["player_name"],
            event["event_type"],
            event["outcome_label"],
            int(event["minute"]),
            int(event["second"]),
        ]],
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Team: %{customdata[0]}<br>"
            "Action: %{customdata[2]}<br>"
            "Outcome: %{customdata[3]}<br>"
            "Time: %{customdata[4]}:%{customdata[5]}<extra></extra>"
        ),
        name="Start",
    )


def _current_end_trace(event: pd.Series, team_colors: dict[str, str]) -> go.Scatter:
    symbol = "star" if bool(event["is_goal"]) else ("diamond" if bool(event["is_shot"]) else "x")
    return go.Scatter(
        x=[event["end_x_plot"]],
        y=[event["end_y_plot"]],
        mode="markers+text",
        marker=dict(
            size=15,
            color=team_colors.get(event["team_name"], "#ffb703"),
            line=dict(color="#fff9f1", width=1.2),
            symbol=symbol,
        ),
        text=[event["event_type"]],
        textposition="top center",
        textfont=dict(color="#fff9f1", size=11),
        customdata=[[
            event["team_name"],
            event["player_name"],
            event["event_type"],
            event["outcome_label"],
            int(event["minute"]),
            int(event["second"]),
        ]],
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Team: %{customdata[0]}<br>"
            "Action: %{customdata[2]}<br>"
            "Outcome: %{customdata[3]}<br>"
            "Time: %{customdata[4]}:%{customdata[5]}<extra></extra>"
        ),
        name="End",
    )


def build_match_replay_figure(events: pd.DataFrame, trail_length: int = 8) -> go.Figure:
    pitch = draw_pitch()
    if events.empty:
        pitch.add_annotation(
            x=50,
            y=50,
            text="No events available for this selection",
            showarrow=False,
            font=dict(color="#fff9f1", size=18),
        )
        return pitch

    teams = events["team_name"].dropna().unique().tolist()
    palette = ["#ffb703", "#8ecae6", "#fb8500", "#d90429"]
    team_colors = {team: palette[idx % len(palette)] for idx, team in enumerate(teams)}

    static_trace = go.Scatter(
        x=events["x_plot"],
        y=events["y_plot"],
        mode="markers",
        marker=dict(
            size=6,
            color=events["team_name"].map(team_colors),
            opacity=0.22,
            line=dict(color="#10231a", width=0.5),
        ),
        customdata=np.stack(
            [
                events["team_name"],
                events["player_name"],
                events["event_type"],
                events["outcome_label"],
                events["minute"].astype(int),
                events["second"].astype(int),
            ],
            axis=-1,
        ),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Team: %{customdata[0]}<br>"
            "Action: %{customdata[2]}<br>"
            "Outcome: %{customdata[3]}<br>"
            "Time: %{customdata[4]}:%{customdata[5]}<extra></extra>"
        ),
        name="Event cloud",
    )

    first_event = events.iloc[0]
    initial_recent = events.iloc[:1]
    pitch.add_trace(static_trace)
    pitch.add_trace(_recent_path_trace(initial_recent))
    pitch.add_trace(_current_path_trace(first_event, team_colors))
    pitch.add_trace(_current_start_trace(first_event, team_colors))
    pitch.add_trace(_current_end_trace(first_event, team_colors))

    frames: list[go.Frame] = []
    slider_steps: list[dict[str, object]] = []
    for idx in range(len(events)):
        event = events.iloc[idx]
        recent = events.iloc[max(0, idx - trail_length + 1) : idx + 1]
        frame_name = str(idx)
        frames.append(
            go.Frame(
                name=frame_name,
                data=[
                    _recent_path_trace(recent),
                    _current_path_trace(event, team_colors),
                    _current_start_trace(event, team_colors),
                    _current_end_trace(event, team_colors),
                ],
                traces=[1, 2, 3, 4],
            )
        )
        slider_steps.append(
            {
                "args": [
                    [frame_name],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": f"{int(event['minute']):02d}:{int(event['second']):02d}",
                "method": "animate",
            }
        )

    pitch.frames = frames
    pitch.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.01,
                "y": 1.08,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 420, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "y": -0.03,
                "len": 0.88,
                "pad": {"b": 10, "t": 35},
                "currentvalue": {"prefix": "Replay time: ", "font": {"color": "#fff9f1"}},
                "steps": slider_steps,
            }
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(19,32,51,0.75)",
            font=dict(color="#fff9f1"),
        ),
    )
    return pitch


def standings_badge(team_name: str) -> str:
    standings = load_standings().set_index("team_name")
    if team_name not in standings.index:
        return "No standings"
    row = standings.loc[team_name]
    return f"#{int(row['pos'])} | {int(row['pts'])} pts | GD {int(row['gd'])}"


matches = load_matches()
match_events = load_match_events()
shot_map = attach_xg_scores()
thresholds = compute_threshold_metrics()
xg_model = fit_xg_model()
logistic_thresholds = compute_logistic_threshold_metrics()
xg_logistic_model = fit_xg_logistic_model()
models = fit_goal_models()
standings = load_standings()
match_options = build_match_options(matches)

teams = sorted(set(matches["home_team"]).union(matches["away_team"]))


st.markdown(
    """
    <div class="panel">
        <div class="eyebrow">Arnold Avances</div>
        <div class="hero-title">Premier League Match Lab</div>
        <div class="hero-copy">
            Dashboard local inspirado en la lectura rapida de WhoScored: mucho contexto arriba,
            comparacion rapida entre equipos, shot map, replay interactivo de eventos y performance de modelos en una sola vista.
            Esta version compara xG lineal polinomico y xG logistico, ambos evaluados con multiples segmentos
            para medir generalizacion real y no quedarnos con un resultado inflado por overfitting.
        </div>
        <div class="tag-row">
            <span class="tag">Local files only</span>
            <span class="tag">WhoScored-inspired layout</span>
            <span class="tag">Ready to evolve with team work</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

with st.sidebar:
    st.header("Control Room")
    home_team = st.selectbox("Home team", teams, index=0)
    away_team_default = 1 if len(teams) > 1 else 0
    away_team = st.selectbox("Away team", teams, index=away_team_default)
    if home_team == away_team:
        st.warning("Choose two different teams to build the matchup.")

    shot_team_filter = st.multiselect(
        "Shot map teams",
        sorted(shot_map["team_name"].dropna().unique()),
        default=[],
    )
    st.markdown("**Match predictor inputs**")
    selected_referee = st.selectbox("Referee", models["referees"], index=0)
    default_odds = models["avg_odds"]
    b365_home = st.number_input("Bet365 home", min_value=1.01, value=float(default_odds["b365h"]), step=0.05)
    b365_draw = st.number_input("Bet365 draw", min_value=1.01, value=float(default_odds["b365d"]), step=0.05)
    b365_away = st.number_input("Bet365 away", min_value=1.01, value=float(default_odds["b365a"]), step=0.05)
    shot_model_choice = st.selectbox("Shot xG model", ["Logistic xG", "Linear xG"], index=0)
    max_shots = st.slider("Shots shown on map", min_value=50, max_value=1500, value=350, step=50)
    active_thresholds = logistic_thresholds if shot_model_choice == "Logistic xG" else thresholds
    active_best_threshold = (
        xg_logistic_model["best_threshold"] if shot_model_choice == "Logistic xG" else xg_model["best_threshold"]
    )
    threshold_selected = st.select_slider(
        "Goal threshold",
        options=active_thresholds["threshold"].round(2).tolist(),
        value=float(active_best_threshold),
    )
    st.caption("The chosen threshold compares the selected xG model on its evaluation splits.")


overview_1, overview_2, overview_3, overview_4 = st.columns(4)
overview_1.metric("Matches loaded", f"{len(matches)}")
overview_2.metric("Shots with features", f"{len(load_shot_features()):,}")
overview_3.metric("Match test accuracy", f"{models['test_result_accuracy']:.3f}")
overview_4.metric("Best shot logistic AUC", f"{xg_logistic_model['candidate_table']['validation_auc'].max():.3f}")

tab_overview, tab_predictor, tab_shots, tab_replay, tab_performance, tab_eda = st.tabs(
    ["Overview", "Match Predictor", "Shot Lab", "Match Replay", "Performance", "EDA"]
)


with tab_overview:
    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        st.subheader("League pulse")
        result_counts = matches["ftr"].map({"H": "Home win", "D": "Draw", "A": "Away win"}).value_counts()
        result_fig = px.bar(
            result_counts,
            x=result_counts.index,
            y=result_counts.values,
            color=result_counts.index,
            color_discrete_map={
                "Home win": "#0b4ea2",
                "Draw": "#d4a017",
                "Away win": "#a61e2a",
            },
            labels={"x": "", "y": "Matches"},
        )
        result_fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
        )
        st.plotly_chart(result_fig, use_container_width=True)

    with col_b:
        st.subheader("Table leaders")
        leaders = standings.sort_values("pts", ascending=False)[["pos", "team_name", "pts", "gd", "gf", "ga"]].head(8)
        st.dataframe(leaders, use_container_width=True, hide_index=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Top attacks")
        home_attack_rank = (
            matches.groupby("home_team")["fthg"].mean().sort_values(ascending=False).head(8).reset_index()
        )
        fig_attack = px.bar(
            home_attack_rank,
            x="fthg",
            y="home_team",
            orientation="h",
            color="fthg",
            color_continuous_scale=["#ffbe0b", "#fb5607", "#c1121f"],
            labels={"home_team": "", "fthg": "Avg home goals"},
        )
        fig_attack.update_layout(coloraxis_showscale=False, paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig_attack, use_container_width=True)

    with col_d:
        st.subheader("Best shot conversion")
        conversion = (
            shot_map.groupby("team_name")
            .agg(goals=("is_goal", "sum"), shots=("id", "count"))
            .assign(conversion=lambda d: d["goals"] / d["shots"])
            .sort_values("conversion", ascending=False)
            .head(8)
            .reset_index()
        )
        fig_conv = px.bar(
            conversion,
            x="conversion",
            y="team_name",
            orientation="h",
            color="conversion",
            color_continuous_scale=["#415a77", "#1b263b", "#0d1b2a"],
            labels={"team_name": "", "conversion": "Goal conversion"},
        )
        fig_conv.update_layout(coloraxis_showscale=False, paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig_conv, use_container_width=True)


with tab_predictor:
    if home_team == away_team:
        st.info("Select two different teams in the sidebar to unlock the projection.")
    else:
        odds_payload = {
            "b365h": float(b365_home),
            "b365d": float(b365_draw),
            "b365a": float(b365_away),
            "bwh": float(models["avg_odds"]["bwh"]),
            "bwd": float(models["avg_odds"]["bwd"]),
            "bwa": float(models["avg_odds"]["bwa"]),
            "maxh": float(models["avg_odds"]["maxh"]),
            "maxd": float(models["avg_odds"]["maxd"]),
            "maxa": float(models["avg_odds"]["maxa"]),
            "avgh": float(models["avg_odds"]["avgh"]),
            "avgd": float(models["avg_odds"]["avgd"]),
            "avga": float(models["avg_odds"]["avga"]),
        }
        projection = matchup_projection(home_team, away_team, selected_referee, odds_payload)
        probs = projection["probabilities"]

        st.subheader(f"{home_team} vs {away_team}")
        st.caption(
            f"{home_team}: {standings_badge(home_team)} | "
            f"{away_team}: {standings_badge(away_team)}"
        )

        met_1, met_2, met_3, met_4, met_5 = st.columns(5)
        met_1.metric("Expected total goals", f"{projection['total_goals']:.2f}")
        met_2.metric("Predicted result", {"H": "Home", "D": "Draw", "A": "Away"}[projection["predicted_result"]])
        met_3.metric("Home win", f"{probs['H']:.1%}")
        met_4.metric("Draw", f"{probs['D']:.1%}")
        met_5.metric("Away win", f"{probs['A']:.1%}")

        prob_df = pd.DataFrame(
            {
                "result": ["Home", "Draw", "Away"],
                "probability": [probs["H"], probs["D"], probs["A"]],
            }
        )
        prob_fig = px.bar(
            prob_df,
            x="result",
            y="probability",
            color="result",
            color_discrete_map={"Home": "#0b4ea2", "Draw": "#d4a017", "Away": "#a61e2a"},
            labels={"result": "", "probability": "Probability"},
        )
        prob_fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            yaxis_tickformat=".0%",
        )

        feature_col_1, feature_col_2 = st.columns([1.2, 1])
        with feature_col_1:
            st.plotly_chart(prob_fig, use_container_width=True)
        with feature_col_2:
            feature_table = pd.DataFrame(
                [
                    {"feature": feature, "value": float(projection["features"][feature])}
                    for feature in HOME_FEATURES + AWAY_FEATURES
                ]
                + [
                    {"feature": "referee", "value": selected_referee},
                    {"feature": "b365h", "value": float(b365_home)},
                    {"feature": "b365d", "value": float(b365_draw)},
                    {"feature": "b365a", "value": float(b365_away)},
                ]
            )
            st.dataframe(feature_table, use_container_width=True, hide_index=True)

        st.caption(
            "This section now follows the rubric split: linear regression estimates total goals, "
            "and multinomial logistic regression predicts H, D or A using match stats, referee and odds."
        )


with tab_shots:
    st.subheader("Interactive shot map")
    st.caption("Now the pitch has its own controls so the key information does not get buried.")
    active_xg_column = "xg_logistic" if shot_model_choice == "Logistic xG" else "xg_linear"
    active_xg_label = "xG logistic" if shot_model_choice == "Logistic xG" else "xG linear"

    shot_controls_1, shot_controls_2, shot_controls_3, shot_controls_4 = st.columns(4)
    with shot_controls_1:
        shot_scope = st.selectbox(
            "Scope",
            ["All shots", "Goals only", "Non-goals", "High xG only"],
            index=0,
        )
    with shot_controls_2:
        color_mode = st.selectbox(
            "Color by",
            ["xG value", "Result"],
            index=0,
        )
    with shot_controls_3:
        min_xg = st.slider("Min xG", min_value=0.0, max_value=0.9, value=0.0, step=0.05)
    with shot_controls_4:
        minute_cap = st.slider("Max minute", min_value=0, max_value=130, value=130, step=5)

    shot_controls_5, shot_controls_6, shot_controls_7 = st.columns([1.2, 1, 1])
    with shot_controls_5:
        highlight_team = st.selectbox("Highlight team", ["None"] + teams, index=0)
    with shot_controls_6:
        show_labels = st.toggle("Show top labels", value=True)
    with shot_controls_7:
        label_count = st.slider("Labels", min_value=3, max_value=20, value=8, step=1)

    shot_view = shot_map.copy()
    if shot_team_filter:
        shot_view = shot_view[shot_view["team_name"].isin(shot_team_filter)]
    shot_view = shot_view.dropna(subset=["x_plot", "y_plot"]).copy()
    shot_view = shot_view[shot_view[active_xg_column] >= min_xg]
    shot_view = shot_view[shot_view["minute"].fillna(0) <= minute_cap]

    if shot_scope == "Goals only":
        shot_view = shot_view[shot_view["is_goal"] == 1]
    elif shot_scope == "Non-goals":
        shot_view = shot_view[shot_view["is_goal"] == 0]
    elif shot_scope == "High xG only":
        shot_view = shot_view[shot_view[active_xg_column] >= max(float(threshold_selected), 0.20)]

    shot_view = shot_view.sort_values(active_xg_column, ascending=False).head(max_shots).copy()
    shot_view["goal_label"] = np.where(shot_view["is_goal"] == 1, "Goal", "No goal")
    shot_view["marker_size"] = np.clip(8 + shot_view[active_xg_column] * 18, 8, 24)
    shot_view["marker_opacity"] = np.where(shot_view["is_goal"] == 1, 0.98, 0.58)
    if highlight_team != "None":
        shot_view["marker_opacity"] = np.where(
            shot_view["team_name"] == highlight_team,
            np.where(shot_view["is_goal"] == 1, 1.0, 0.88),
            0.18,
        )

    pitch_col, info_col = st.columns([2.1, 1])

    with pitch_col:
        pitch = draw_pitch()
        if not shot_view.empty:
            if color_mode == "xG value":
                goals = shot_view[shot_view["is_goal"] == 1]
                non_goals = shot_view[shot_view["is_goal"] == 0]
                for frame, name, symbol in [(non_goals, "No goal", "circle"), (goals, "Goal", "star")]:
                    if frame.empty:
                        continue
                    pitch.add_trace(
                        go.Scatter(
                            x=frame["x_plot"],
                            y=frame["y_plot"],
                            mode="markers",
                            marker=dict(
                                size=frame["marker_size"],
                                color=frame[active_xg_column],
                                colorscale="YlOrRd",
                                cmin=0,
                                cmax=1,
                                opacity=frame["marker_opacity"],
                                line=dict(width=1.2, color="#f7f2e8"),
                                symbol=symbol,
                                colorbar=dict(title=active_xg_label, x=1.02) if name == "Goal" else None,
                            ),
                            customdata=np.stack(
                                [
                                    frame["player_name"].fillna("Unknown"),
                                    frame["team_name"].fillna("Unknown"),
                                    frame["minute"].fillna(0),
                                    frame[active_xg_column].round(3),
                                    frame["goal_label"],
                                    frame["distance"].round(2),
                                ],
                                axis=-1,
                            ),
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"
                                "Team: %{customdata[1]}<br>"
                                "Minute: %{customdata[2]}<br>"
                                "xG: %{customdata[3]}<br>"
                                "Outcome: %{customdata[4]}<br>"
                                "Distance: %{customdata[5]}<extra></extra>"
                            ),
                            name=name,
                        )
                    )
            else:
                palette = {"Goal": "#ffb703", "No goal": "#d8e2dc"}
                for label in ["No goal", "Goal"]:
                    frame = shot_view[shot_view["goal_label"] == label]
                    if frame.empty:
                        continue
                    pitch.add_trace(
                        go.Scatter(
                            x=frame["x_plot"],
                            y=frame["y_plot"],
                            mode="markers",
                            marker=dict(
                                size=frame["marker_size"],
                                color=palette[label],
                                opacity=frame["marker_opacity"],
                                line=dict(width=1.2, color="#10231a"),
                                symbol="star" if label == "Goal" else "circle",
                            ),
                            customdata=np.stack(
                                [
                                    frame["player_name"].fillna("Unknown"),
                                    frame["team_name"].fillna("Unknown"),
                                    frame["minute"].fillna(0),
                                    frame[active_xg_column].round(3),
                                    frame["goal_label"],
                                ],
                                axis=-1,
                            ),
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"
                                "Team: %{customdata[1]}<br>"
                                "Minute: %{customdata[2]}<br>"
                                "xG: %{customdata[3]}<br>"
                                "Outcome: %{customdata[4]}<extra></extra>"
                            ),
                            name=label,
                        )
                    )

            if show_labels:
                label_df = shot_view.head(label_count)
                pitch.add_trace(
                    go.Scatter(
                        x=label_df["x_plot"],
                        y=label_df["y_plot"],
                        mode="text",
                        text=label_df["player_name"].fillna("Unknown"),
                        textposition="top center",
                        textfont=dict(color="#fff9f1", size=11),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        pitch.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(19,32,51,0.75)",
                font=dict(color="#fff9f1"),
            )
        )
        st.plotly_chart(
            pitch,
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )

    with info_col:
        shot_stats_1, shot_stats_2 = st.columns(2)
        shot_stats_3, shot_stats_4 = st.columns(2)
        shot_stats_1.metric("Shots shown", f"{len(shot_view)}")
        shot_stats_2.metric("Goals", f"{int(shot_view['is_goal'].sum()) if not shot_view.empty else 0}")
        shot_stats_3.metric(f"Average {active_xg_label}", f"{shot_view[active_xg_column].mean():.3f}" if not shot_view.empty else "0.000")
        shot_stats_4.metric("High xG", f"{int((shot_view[active_xg_column] >= 0.20).sum()) if not shot_view.empty else 0}")

        shot_summary = (
            shot_view.groupby("team_name")
            .agg(shots=("id", "count"), goals=("is_goal", "sum"), avg_xg=(active_xg_column, "mean"))
            .sort_values(["goals", "avg_xg", "shots"], ascending=False)
            .head(10)
            .reset_index()
        )
        st.markdown("**Team summary on the current view**")
        st.dataframe(shot_summary.round(3), use_container_width=True, hide_index=True)

        top_shots = (
            shot_view[["player_name", "team_name", active_xg_column, "is_goal", "distance", "minute"]]
            .sort_values([active_xg_column, "is_goal"], ascending=[False, False])
            .head(12)
            .rename(columns={active_xg_column: "xg", "is_goal": "goal"})
        )
        st.markdown("**Most dangerous shots**")
        st.dataframe(top_shots.round(3), use_container_width=True, hide_index=True)


with tab_replay:
    st.subheader("Match replay")
    st.caption(
        "This replay uses event data, not tracking data. It animates actions and trajectories from the match feed, "
        "so it is ideal for understanding sequences, pressure zones and player involvement."
    )

    replay_label = st.selectbox("Match", list(match_options.keys()), index=0)
    replay_match_id = match_options[replay_label]
    replay_meta = matches.loc[matches["id"] == replay_match_id].iloc[0]
    replay_events = match_events.loc[match_events["match_id"] == replay_match_id].copy()

    replay_controls_1, replay_controls_2, replay_controls_3 = st.columns(3)
    with replay_controls_1:
        replay_team_filter = st.multiselect(
            "Teams in replay",
            sorted(replay_events["team_name"].dropna().unique().tolist()),
            default=sorted(replay_events["team_name"].dropna().unique().tolist()),
        )
    with replay_controls_2:
        replay_player_options = ["All players"] + sorted(replay_events["player_name"].dropna().unique().tolist())
        replay_player = st.selectbox("Player focus", replay_player_options, index=0)
    with replay_controls_3:
        replay_successful_only = st.toggle("Successful only", value=False)

    replay_controls_4, replay_controls_5, replay_controls_6 = st.columns(3)
    with replay_controls_4:
        replay_type_options = sorted(replay_events["event_type"].dropna().unique().tolist())
        replay_default_types = [
            event_type
            for event_type in ["Pass", "TakeOn", "SavedShot", "MissedShots", "BallRecovery", "Tackle"]
            if event_type in replay_type_options
        ]
        replay_event_types = st.multiselect(
            "Event types",
            replay_type_options,
            default=replay_default_types,
        )
    with replay_controls_5:
        replay_minute_range = st.slider("Minute range", min_value=0, max_value=120, value=(0, 95), step=1)
    with replay_controls_6:
        replay_event_limit = st.slider("Replay steps", min_value=20, max_value=200, value=80, step=10)

    replay_controls_7, replay_controls_8 = st.columns(2)
    with replay_controls_7:
        replay_trail_length = st.slider("Trail length", min_value=2, max_value=20, value=8, step=1)
    with replay_controls_8:
        replay_include_stops = st.toggle("Include start/end events", value=False)

    filtered_replay = replay_events.copy()
    if replay_team_filter:
        filtered_replay = filtered_replay[filtered_replay["team_name"].isin(replay_team_filter)]
    if replay_player != "All players":
        filtered_replay = filtered_replay[filtered_replay["player_name"] == replay_player]
    if replay_event_types:
        filtered_replay = filtered_replay[filtered_replay["event_type"].isin(replay_event_types)]
    filtered_replay = filtered_replay[
        filtered_replay["minute"].between(replay_minute_range[0], replay_minute_range[1])
    ]
    if replay_successful_only:
        filtered_replay = filtered_replay[filtered_replay["outcome_label"] == "Successful"]
    if not replay_include_stops:
        filtered_replay = filtered_replay[~filtered_replay["event_type"].isin(["Start", "End"])]

    filtered_replay = filtered_replay.sort_values(["clock_seconds", "id"]).copy()
    if len(filtered_replay) > replay_event_limit:
        sampled_idx = np.linspace(0, len(filtered_replay) - 1, replay_event_limit).astype(int)
        filtered_replay = filtered_replay.iloc[sampled_idx].copy()

    replay_col, replay_info_col = st.columns([2.1, 1])
    with replay_col:
        replay_fig = build_match_replay_figure(filtered_replay, trail_length=replay_trail_length)
        st.plotly_chart(
            replay_fig,
            use_container_width=True,
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )

    with replay_info_col:
        replay_metric_1, replay_metric_2 = st.columns(2)
        replay_metric_3, replay_metric_4 = st.columns(2)
        replay_metric_1.metric("Replay actions", f"{len(filtered_replay)}")
        replay_metric_2.metric("Goals in view", f"{int(filtered_replay['is_goal'].sum()) if not filtered_replay.empty else 0}")
        replay_metric_3.metric("Shots in view", f"{int(filtered_replay['is_shot'].sum()) if not filtered_replay.empty else 0}")
        replay_metric_4.metric("Teams", f"{filtered_replay['team_name'].nunique() if not filtered_replay.empty else 0}")

        st.markdown("**Match context**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"field": "Date", "value": replay_meta["season_label"]},
                    {"field": "Fixture", "value": f"{replay_meta['home_team']} vs {replay_meta['away_team']}"},
                    {"field": "Score", "value": f"{int(replay_meta['fthg'])}-{int(replay_meta['ftag'])}"},
                    {"field": "Result", "value": replay_meta["ftr"]},
                    {"field": "Referee", "value": replay_meta["referee"]},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        team_event_summary = (
            filtered_replay.groupby("team_name")
            .agg(
                actions=("id", "count"),
                shots=("is_shot", "sum"),
                goals=("is_goal", "sum"),
            )
            .sort_values(["goals", "shots", "actions"], ascending=False)
            .reset_index()
        )
        st.markdown("**Replay summary by team**")
        st.dataframe(team_event_summary, use_container_width=True, hide_index=True)

        recent_actions = (
            filtered_replay[["minute", "second", "team_name", "player_name", "event_type", "outcome_label"]]
            .sort_values(["minute", "second"], ascending=[False, False])
            .head(12)
        )
        st.markdown("**Latest actions in current filter**")
        st.dataframe(recent_actions, use_container_width=True, hide_index=True)


with tab_performance:
    st.subheader("Current model performance")
    perf_1, perf_2, perf_3, perf_4 = st.columns(4)
    perf_1.metric("Goals val RMSE", f"{models['val_goal_rmse']:.3f}")
    perf_2.metric("Goals test RMSE", f"{models['test_goal_rmse']:.3f}")
    perf_3.metric("Goals val R2", f"{models['val_goal_r2']:.3f}")
    perf_4.metric("Goals test R2", f"{models['test_goal_r2']:.3f}")

    match_perf_1, match_perf_2, match_perf_3, match_perf_4 = st.columns(4)
    match_perf_1.metric("Result val accuracy", f"{models['val_result_accuracy']:.3f}")
    match_perf_2.metric("Result test accuracy", f"{models['test_result_accuracy']:.3f}")
    match_perf_3.metric("Bet365 test accuracy", f"{models['test_b365_accuracy']:.3f}")
    match_perf_4.metric("Result test F1 macro", f"{models['test_result_f1_macro']:.3f}")

    linear_selected_row = thresholds.loc[thresholds["threshold"] == float(threshold_selected)].iloc[0]
    best_row = xg_model["candidate_table"].loc[
        xg_model["candidate_table"]["degree"] == xg_model["best_degree"]
    ].iloc[0]
    logistic_selected_row = logistic_thresholds.loc[
        logistic_thresholds["threshold"] == float(threshold_selected)
    ].iloc[0]
    logistic_best_row = xg_logistic_model["candidate_table"].loc[
        xg_logistic_model["candidate_table"]["C"] == xg_logistic_model["best_c"]
    ].iloc[0]

    st.markdown("**Selected polynomial linear xG model**")
    perf_5, perf_6, perf_7, perf_8 = st.columns(4)
    perf_5.metric("Best degree", int(best_row["degree"]))
    perf_6.metric("Best threshold", f"{best_row['best_threshold']:.2f}")
    perf_7.metric("Revalidation RMSE", f"{best_row['revalidation_rmse']:.3f}")
    perf_8.metric("Overfit gap RMSE", f"{best_row['overfit_gap_rmse']:.3f}")

    split_metrics_1, split_metrics_2, split_metrics_3 = st.columns(3)
    split_metrics_1.metric("Train R2", f"{best_row['train_r2']:.3f}")
    split_metrics_2.metric("Test R2", f"{best_row['test_r2']:.3f}")
    split_metrics_3.metric("Revalidation R2", f"{best_row['revalidation_r2']:.3f}")

    linear_threshold_fig = px.line(
        thresholds,
        x="threshold",
        y=["test_precision", "test_recall", "test_f1", "reval_precision", "reval_recall", "reval_f1"],
        markers=True,
        labels={"value": "Metric", "threshold": "Threshold", "variable": "Series"},
    )
    linear_threshold_fig.add_vline(x=float(threshold_selected), line_dash="dash", line_color="#a61e2a")
    linear_threshold_fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        legend_title_text="Split metric",
    )
    st.plotly_chart(linear_threshold_fig, use_container_width=True)

    compare_1, compare_2 = st.columns(2)
    with compare_1:
        st.markdown("**Linear xG threshold snapshot on test split**")
        linear_test_snapshot = pd.DataFrame(
            {
                "metric": ["accuracy", "precision", "recall", "f1", "TP", "FP", "TN", "FN"],
                "value": [
                    linear_selected_row["test_accuracy"],
                    linear_selected_row["test_precision"],
                    linear_selected_row["test_recall"],
                    linear_selected_row["test_f1"],
                    int(linear_selected_row["test_tp"]),
                    int(linear_selected_row["test_fp"]),
                    int(linear_selected_row["test_tn"]),
                    int(linear_selected_row["test_fn"]),
                ],
            }
        )
        st.dataframe(linear_test_snapshot, use_container_width=True, hide_index=True)

    with compare_2:
        st.markdown("**Linear xG threshold snapshot on revalidation split**")
        linear_reval_snapshot = pd.DataFrame(
            {
                "metric": ["accuracy", "precision", "recall", "f1", "TP", "FP", "TN", "FN"],
                "value": [
                    linear_selected_row["reval_accuracy"],
                    linear_selected_row["reval_precision"],
                    linear_selected_row["reval_recall"],
                    linear_selected_row["reval_f1"],
                    int(linear_selected_row["reval_tp"]),
                    int(linear_selected_row["reval_fp"]),
                    int(linear_selected_row["reval_tn"]),
                    int(linear_selected_row["reval_fn"]),
                ],
            }
        )
        st.dataframe(linear_reval_snapshot, use_container_width=True, hide_index=True)

    st.dataframe(
        xg_model["candidate_table"][
            [
                "degree",
                "train_rmse",
                "test_rmse",
                "revalidation_rmse",
                "train_r2",
                "test_r2",
                "revalidation_r2",
                "best_threshold",
                "test_f1",
                "revalidation_f1",
                "overfit_gap_rmse",
            ]
        ].round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Selected logistic xG model from the PDF workflow**")
    log_1, log_2, log_3, log_4 = st.columns(4)
    log_1.metric("Best C", f"{logistic_best_row['C']:.2f}")
    log_2.metric("Best threshold", f"{logistic_best_row['best_threshold']:.2f}")
    log_3.metric("Validation AUC", f"{logistic_best_row['validation_auc']:.3f}")
    log_4.metric("Test AUC", f"{logistic_best_row['test_auc']:.3f}")

    log_5, log_6, log_7, log_8 = st.columns(4)
    log_5.metric("Validation F1", f"{logistic_best_row['validation_f1']:.3f}")
    log_6.metric("Test F1", f"{logistic_best_row['test_f1']:.3f}")
    log_7.metric("Validation Recall", f"{logistic_best_row['validation_recall']:.3f}")
    log_8.metric("Overfit gap AUC", f"{logistic_best_row['overfit_gap_auc']:.3f}")

    logistic_threshold_fig = px.line(
        logistic_thresholds,
        x="threshold",
        y=[
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "test_precision",
            "test_recall",
            "test_f1",
        ],
        markers=True,
        labels={"value": "Metric", "threshold": "Threshold", "variable": "Series"},
    )
    logistic_threshold_fig.add_vline(x=float(threshold_selected), line_dash="dash", line_color="#0b4ea2")
    logistic_threshold_fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        legend_title_text="Split metric",
    )
    st.plotly_chart(logistic_threshold_fig, use_container_width=True)

    logistic_compare_1, logistic_compare_2 = st.columns(2)
    with logistic_compare_1:
        st.markdown("**Logistic xG snapshot on validation split**")
        logistic_val_snapshot = pd.DataFrame(
            {
                "metric": ["accuracy", "precision", "recall", "f1", "AUC", "TP", "FP", "TN", "FN"],
                "value": [
                    logistic_selected_row["validation_accuracy"],
                    logistic_selected_row["validation_precision"],
                    logistic_selected_row["validation_recall"],
                    logistic_selected_row["validation_f1"],
                    logistic_best_row["validation_auc"],
                    int(logistic_selected_row["validation_tp"]),
                    int(logistic_selected_row["validation_fp"]),
                    int(logistic_selected_row["validation_tn"]),
                    int(logistic_selected_row["validation_fn"]),
                ],
            }
        )
        st.dataframe(logistic_val_snapshot, use_container_width=True, hide_index=True)

    with logistic_compare_2:
        st.markdown("**Logistic xG snapshot on test split**")
        logistic_test_snapshot = pd.DataFrame(
            {
                "metric": ["accuracy", "precision", "recall", "f1", "AUC", "TP", "FP", "TN", "FN"],
                "value": [
                    logistic_selected_row["test_accuracy"],
                    logistic_selected_row["test_precision"],
                    logistic_selected_row["test_recall"],
                    logistic_selected_row["test_f1"],
                    logistic_best_row["test_auc"],
                    int(logistic_selected_row["test_tp"]),
                    int(logistic_selected_row["test_fp"]),
                    int(logistic_selected_row["test_tn"]),
                    int(logistic_selected_row["test_fn"]),
                ],
            }
        )
        st.dataframe(logistic_test_snapshot, use_container_width=True, hide_index=True)

    st.dataframe(
        xg_logistic_model["candidate_table"][
            [
                "C",
                "best_threshold",
                "train_accuracy",
                "validation_accuracy",
                "test_accuracy",
                "train_f1",
                "validation_f1",
                "test_f1",
                "train_auc",
                "validation_auc",
                "test_auc",
                "overfit_gap_auc",
            ]
        ].round(3),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "The xG panel now compares a polynomial linear model and a logistic model aligned with the PDF rubric. "
        "Linear xG is selected by revalidation stability, while logistic xG is selected by validation AUC and then checked on the final test split."
    )


with tab_eda:
    st.subheader("Exploratory view")
    eda_left, eda_right = st.columns(2)

    with eda_left:
        goals_by_team = (
            matches.groupby("home_team")[["fthg", "ftag"]]
            .sum()
            .rename(columns={"fthg": "home_goals", "ftag": "away_goals_against"})
            .sort_values("home_goals", ascending=False)
            .head(10)
            .reset_index()
        )
        fig_goals = px.bar(
            goals_by_team,
            x="home_team",
            y="home_goals",
            color="home_goals",
            color_continuous_scale=["#ccd5ae", "#588157", "#344e41"],
            labels={"home_team": "", "home_goals": "Home goals"},
        )
        fig_goals.update_layout(coloraxis_showscale=False, paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig_goals, use_container_width=True)

    with eda_right:
        fig_scatter = px.scatter(
            matches,
            x="hst",
            y="fthg",
            color="ftr",
            color_discrete_map={"H": "#0b4ea2", "D": "#d4a017", "A": "#a61e2a"},
            hover_data=["home_team", "away_team", "date"],
            labels={"hst": "Home shots on target", "fthg": "Home goals", "ftr": "Result"},
        )
        fig_scatter.update_layout(paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    eda_bottom_left, eda_bottom_right = st.columns(2)

    with eda_bottom_left:
        shot_bins = shot_map.assign(distance_band=pd.cut(shot_map["distance"], bins=[0, 8, 14, 22, 40, 80]))
        xg_by_band = (
            shot_bins.groupby("distance_band", observed=False)
            .agg(avg_xg=(active_xg_column, "mean"), goals=("is_goal", "sum"), shots=("id", "count"))
            .reset_index()
        )
        xg_by_band["distance_band"] = xg_by_band["distance_band"].astype(str)
        fig_band = px.bar(
            xg_by_band,
            x="distance_band",
            y="avg_xg",
            color="avg_xg",
            color_continuous_scale=["#ffe8d6", "#ffb703", "#d00000"],
            labels={"distance_band": "Distance band", "avg_xg": f"Average {active_xg_label}"},
        )
        fig_band.update_layout(coloraxis_showscale=False, paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig_band, use_container_width=True)

    with eda_bottom_right:
        draw_heat = (
            matches.assign(is_draw=(matches["ftr"] == "D").astype(int))
            .groupby("referee")
            .agg(draw_rate=("is_draw", "mean"), matches=("id", "count"))
            .query("matches >= 5")
            .sort_values("draw_rate", ascending=False)
            .head(12)
            .reset_index()
        )
        fig_ref = px.bar(
            draw_heat,
            x="draw_rate",
            y="referee",
            orientation="h",
            color="matches",
            color_continuous_scale=["#90e0ef", "#0077b6"],
            labels={"draw_rate": "Draw rate", "referee": ""},
        )
        fig_ref.update_layout(xaxis_tickformat=".0%", paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)")
        st.plotly_chart(fig_ref, use_container_width=True)
