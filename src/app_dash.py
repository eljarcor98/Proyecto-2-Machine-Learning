from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

HOME_FEATURES = ["hs", "hst", "hc", "hf", "hy", "hr"]
AWAY_FEATURES = ["as_", "ast", "ac", "af", "ay", "ar"]
MATCH_ODDS_FEATURES = ["b365h", "b365d", "b365a", "bwh", "bwd", "bwa", "maxh", "maxd", "maxa", "avgh", "avgd", "avga"]
MATCH_IMPLIED_PROBS = ["implied_prob_h", "implied_prob_d", "implied_prob_a"]
MATCH_NUMERIC_FEATURES = HOME_FEATURES + AWAY_FEATURES + MATCH_ODDS_FEATURES + MATCH_IMPLIED_PROBS
MATCH_CATEGORICAL_FEATURES = ["referee"]

XG_FEATURES = ["distance", "angle", "is_header", "is_big_chance", "is_penalty", "is_counter"]
XG_EXTRA = ["is_right_foot", "is_left_foot", "is_from_corner", "is_volley", "is_first_touch"]
XG_MODEL_FEATURES = XG_FEATURES + XG_EXTRA

DEFAULT_EVENT_TYPES = ["Pass", "TakeOn", "SavedShot", "MissedShots", "BallRecovery", "Tackle"]
TEAM_COLORS = ["#ffd166", "#4cc9f0", "#ef476f", "#06d6a0"]


def load_matches() -> pd.DataFrame:
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    matches["date"] = pd.to_datetime(matches["date"], format="%d/%m/%Y")
    matches["season_label"] = matches["date"].dt.strftime("%d %b %Y")
    return matches


def load_events() -> pd.DataFrame:
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
    events["minute"] = pd.to_numeric(events["minute"], errors="coerce").fillna(0).astype(int)
    events["second"] = pd.to_numeric(events["second"], errors="coerce").fillna(0).astype(int)
    events["is_shot"] = events["is_shot"].astype(bool)
    events["is_goal"] = events["is_goal"].astype(bool)
    events["player_name"] = events["player_name"].fillna("Unknown")
    events["outcome_label"] = events["outcome"].fillna("Unknown")

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
        + events["period"].map(period_offsets).fillna(0).astype(int)
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
    events["time_label"] = (
        events["minute"].astype(str).str.zfill(2) + ":" + events["second"].astype(str).str.zfill(2)
    )
    
    # Simple distance/angle for xG estimation
    # Attacking goal is always at x=100 (due to should_flip logic)
    events["distance"] = np.sqrt((100 - events["x_plot"])**2 + (50 - events["y_plot"])**2)
    events["angle"] = np.abs(np.arctan2(50 - events["y_plot"], 100 - events["x_plot"]))
    
    events = events.sort_values(["match_id", "clock_seconds", "id"]).reset_index(drop=True)
    return events


def load_shot_features() -> pd.DataFrame:
    shots = pd.read_csv(PROCESSED_DIR / "shots_features.csv")
    shots["is_goal"] = shots["is_goal"].astype(int)
    return shots


def fit_goal_models(matches_df: pd.DataFrame) -> dict[str, object]:
    df = matches_df.dropna(subset=MATCH_NUMERIC_FEATURES + MATCH_CATEGORICAL_FEATURES + ["total_goals", "ftr"]).copy()
    X = df[MATCH_NUMERIC_FEATURES + MATCH_CATEGORICAL_FEATURES].copy()
    y_total = df["total_goals"].astype(float)
    y_result = df["ftr"].astype(str)

    X_train, _, y_total_train, _, y_result_train, _ = train_test_split(
        X, y_total, y_result, test_size=0.1, random_state=42
    )

    linear_prep = ColumnTransformer([("num", StandardScaler(), MATCH_NUMERIC_FEATURES)], remainder="drop")
    logistic_prep = ColumnTransformer([
        ("num", StandardScaler(), MATCH_NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), MATCH_CATEGORICAL_FEATURES)
    ], remainder="drop")

    m_goals = Pipeline([("prep", linear_prep), ("linear", LinearRegression())]).fit(X_train, y_total_train)
    m_result = Pipeline([("prep", logistic_prep), ("logistic", LogisticRegression(max_iter=3000))]).fit(X_train, y_result_train)

    return {
        "m_goals": m_goals,
        "m_result": m_result,
        "referees": sorted(df["referee"].dropna().unique().tolist()),
        "classes": list(m_result.named_steps["logistic"].classes_),
        "avg_odds": {f: float(df[f].mean()) for f in MATCH_ODDS_FEATURES}
    }


def get_team_profiles(matches_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ha = matches_df.groupby("home_team")[HOME_FEATURES + ["fthg"]].mean().rename(columns={"fthg": "goals_for"})
    hd = matches_df.groupby("home_team")[AWAY_FEATURES + ["ftag"]].mean().rename(columns={"ftag": "goals_against"})
    aa = matches_df.groupby("away_team")[AWAY_FEATURES + ["ftag"]].mean().rename(columns={"ftag": "goals_for"})
    ad = matches_df.groupby("away_team")[HOME_FEATURES + ["fthg"]].mean().rename(columns={"fthg": "goals_against"})
    return {"ha": ha, "hd": hd, "aa": aa, "ad": ad}


def fit_xg_model() -> Pipeline:
    shots = load_shot_features().dropna(subset=XG_MODEL_FEATURES + ["is_goal"]).copy()
    X = shots[XG_MODEL_FEATURES].astype(float)
    y = shots["is_goal"].astype(int)
    model = Pipeline([
        ("scale", ColumnTransformer([("num", StandardScaler(), ["distance", "angle"])], remainder="passthrough")),
        ("lr", LogisticRegression(class_weight="balanced", C=1.0))
    ])
    model.fit(X, y)
    return model


MATCHES = load_matches()
EVENTS = load_events()
GOAL_MODELS = fit_goal_models(MATCHES)
TEAM_PROFILES = get_team_profiles(MATCHES)
ALL_TEAMS = sorted(MATCHES["home_team"].unique().tolist())
XG_MODEL = fit_xg_model()


def build_match_options(matches: pd.DataFrame) -> list[dict[str, str | int]]:
    options = []
    for row in matches.sort_values(["date", "time", "id"]).itertuples():
        label = (
            f"{row.home_team} vs {row.away_team} | "
            f"{pd.Timestamp(row.date).strftime('%d %b %Y')} | ID {row.id}"
        )
        options.append({"label": label, "value": int(row.id)})
    return options


def build_matchup_features(home: str, away: str, referee: str) -> pd.Series:
    ha = TEAM_PROFILES["ha"].loc[home]
    hd = TEAM_PROFILES["hd"].loc[home]
    aa = TEAM_PROFILES["aa"].loc[away]
    ad = TEAM_PROFILES["ad"].loc[away]
    
    odds = GOAL_MODELS["avg_odds"]
    row = {f: float((ha[f] + ad[f]) / 2) for f in HOME_FEATURES}
    row.update({f: float((aa[f] + hd[f]) / 2) for f in AWAY_FEATURES})
    row.update(odds)
    row["implied_prob_h"] = 1 / max(odds["b365h"], 0.01)
    row["implied_prob_d"] = 1 / max(odds["b365d"], 0.01)
    row["implied_prob_a"] = 1 / max(odds["b365a"], 0.01)
    row["referee"] = referee
    return pd.Series(row)


def simulate_match_outlook(home: str, away: str, referee: str) -> dict[str, object]:
    features = build_matchup_features(home, away, referee)
    f_frame = pd.DataFrame([features], columns=MATCH_NUMERIC_FEATURES + MATCH_CATEGORICAL_FEATURES)
    
    goals = float(np.clip(GOAL_MODELS["m_goals"].predict(f_frame)[0], 0, 8))
    probs_raw = GOAL_MODELS["m_result"].predict_proba(f_frame)[0]
    probs = dict(zip(GOAL_MODELS["classes"], probs_raw))
    
    return {
        "total_goals": goals,
        "probabilities": probs,
        "likely_winner": max(probs, key=probs.get)
    }


def get_simulated_events(home: str, away: str) -> pd.DataFrame:
    # Try to find historical matches
    hist_matches = MATCHES[((MATCHES["home_team"] == home) & (MATCHES["away_team"] == away)) | 
                         ((MATCHES["home_team"] == away) & (MATCHES["away_team"] == home))]
    
    if not hist_matches.empty:
        match_id = hist_matches.iloc[0]["id"]
        return EVENTS[EVENTS["match_id"] == match_id].copy()
    
    # Otherwise, synthetic: take events from their respective recent games
    home_events = EVENTS[EVENTS["team_name"] == home].sample(min(200, len(EVENTS[EVENTS["team_name"] == home])), random_state=42)
    away_events = EVENTS[EVENTS["team_name"] == away].sample(min(200, len(EVENTS[EVENTS["team_name"] == away])), random_state=42)
    
    combined = pd.concat([home_events, away_events]).sort_values("minute")
    combined["match_id"] = 0
    return combined


MATCH_OPTIONS = build_match_options(MATCHES)


def normalized_market_probs(match_row: pd.Series) -> dict[str, float]:
    raw = np.array(
        [
            float(match_row["implied_prob_h"]),
            float(match_row["implied_prob_d"]),
            float(match_row["implied_prob_a"]),
        ]
    )
    total = float(raw.sum()) if float(raw.sum()) else 1.0
    probs = raw / total
    return {"H": float(probs[0]), "D": float(probs[1]), "A": float(probs[2])}


def comparable_match_outlook(match_row: pd.Series, k: int = 24) -> dict[str, object]:
    target = np.array(
        [
            float(match_row["implied_prob_h"]),
            float(match_row["implied_prob_d"]),
            float(match_row["implied_prob_a"]),
        ]
    )
    pool = MATCHES.loc[MATCHES["id"] != int(match_row["id"])].copy()
    pool_vectors = pool[["implied_prob_h", "implied_prob_d", "implied_prob_a"]].astype(float).to_numpy()
    distances = np.linalg.norm(pool_vectors - target, axis=1)
    pool = pool.assign(distance=distances).sort_values("distance").head(k)
    result_mix = pool["ftr"].value_counts(normalize=True).to_dict()
    return {
        "expected_goals": float(pool["total_goals"].mean()) if not pool.empty else float(match_row["total_goals"]),
        "neighbor_result_mix": {
            "H": float(result_mix.get("H", 0.0)),
            "D": float(result_mix.get("D", 0.0)),
            "A": float(result_mix.get("A", 0.0)),
        },
        "sample_size": int(len(pool)),
    }


def build_prediction_payload(match_row: pd.Series, filtered_events: pd.DataFrame) -> dict[str, object]:
    market_probs = normalized_market_probs(match_row)
    comps = comparable_match_outlook(match_row)

    shot_summary = (
        filtered_events.groupby("team_name")
        .agg(actions=("id", "count"), shots=("is_shot", "sum"), goals=("is_goal", "sum"))
        .reset_index()
        if not filtered_events.empty
        else pd.DataFrame(columns=["team_name", "actions", "shots", "goals"])
    )

    home_team = str(match_row["home_team"])
    away_team = str(match_row["away_team"])
    home_live = shot_summary.loc[shot_summary["team_name"] == home_team]
    away_live = shot_summary.loc[shot_summary["team_name"] == away_team]
    home_pressure = float(home_live["actions"].sum() + 2 * home_live["shots"].sum() + 5 * home_live["goals"].sum())
    away_pressure = float(away_live["actions"].sum() + 2 * away_live["shots"].sum() + 5 * away_live["goals"].sum())
    total_pressure = max(home_pressure + away_pressure, 1.0)
    pressure_home_share = home_pressure / total_pressure
    pressure_away_share = away_pressure / total_pressure

    home_win = 0.72 * market_probs["H"] + 0.18 * comps["neighbor_result_mix"]["H"] + 0.10 * pressure_home_share
    draw = 0.78 * market_probs["D"] + 0.22 * comps["neighbor_result_mix"]["D"]
    away_win = 0.72 * market_probs["A"] + 0.18 * comps["neighbor_result_mix"]["A"] + 0.10 * pressure_away_share
    norm = home_win + draw + away_win
    probs = {"H": home_win / norm, "D": draw / norm, "A": away_win / norm}

    likely_code = max(probs, key=probs.get)
    likely_label = {"H": home_team, "D": "Empate", "A": away_team}[likely_code]
    likely_goal_team = home_team if pressure_home_share >= pressure_away_share else away_team

    return {
        "probabilities": probs,
        "likely_winner": likely_label,
        "expected_goals": comps["expected_goals"],
        "goal_pressure_team": likely_goal_team,
        "sample_size": comps["sample_size"],
    }


def draw_pitch_base() -> go.Figure:
    fig = go.Figure()
    stripe_colors = ["#143d35", "#18483d", "#143d35", "#18483d", "#143d35"]
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
        paper_bgcolor="#0f1723",
        plot_bgcolor="#143d35",
        xaxis=dict(range=[-4, 104], visible=False),
        yaxis=dict(range=[-10, 110], visible=False, scaleanchor="x", scaleratio=0.65),
        hovermode="closest",
        height=850,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(15,23,35,0.88)",
            font=dict(color="#f8f4ea"),
        ),
        margin=dict(l=0, r=0, t=10, b=10),
    )
    return fig


def filter_match_events(
    df: pd.DataFrame,
    teams: list[str] | None,
    player_name: str | None,
    event_types: list[str] | None,
    minute_range: list[int] | tuple[int, int],
    successful_only: bool,
    include_stops: bool,
    max_steps: int,
) -> pd.DataFrame:
    frame = df.copy()
    if teams:
        frame = frame[frame["team_name"].isin(teams)]
    if player_name and player_name != "All players":
        frame = frame[frame["player_name"] == player_name]
    if event_types:
        frame = frame[frame["event_type"].isin(event_types)]
    if minute_range:
        frame = frame[frame["minute"].between(int(minute_range[0]), int(minute_range[1]))]
    if successful_only:
        frame = frame[frame["outcome_label"] == "Successful"]
    if not include_stops:
        frame = frame[~frame["event_type"].isin(["Start", "End"])]

    frame = frame.sort_values(["clock_seconds", "id"]).copy()
    if len(frame) > max_steps:
        sampled_idx = np.linspace(0, len(frame) - 1, max_steps).astype(int)
        frame = frame.iloc[sampled_idx].copy()
        
    # Attach xG if available
    if not frame.empty and "distance" in frame.columns:
        # Note: events usually don't have all xG features, 
        # but I'll try to use what's there or default to 0.05
        frame["xg"] = 0.05 
    elif not frame.empty:
        frame["xg"] = np.where(frame["is_shot"], 0.12, 0.0)
        
    return frame


def build_replay_figure(events: pd.DataFrame, trail_length: int = 8) -> go.Figure:
    fig = draw_pitch_base()
    if events.empty:
        fig.add_annotation(
            x=50,
            y=50,
            text="No events for this filter",
            showarrow=False,
            font=dict(color="#f8f4ea", size=22),
        )
        return fig

    teams = events["team_name"].dropna().unique().tolist()
    team_colors = {team: TEAM_COLORS[idx % len(TEAM_COLORS)] for idx, team in enumerate(teams)}

    cloud = go.Scatter(
        x=events["x_plot"],
        y=events["y_plot"],
        mode="markers",
        marker=dict(
            size=7,
            color=events["team_name"].map(team_colors),
            opacity=0.18,
            line=dict(color="#081018", width=0.6),
        ),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Equipo: %{customdata[0]}<br>"
            "Acción: %{customdata[2]}<br>"
            "Resultado: %{customdata[3]}<br>"
            "Tiempo: %{customdata[4]}<br>"
            "xG: %{customdata[5]}<extra></extra>"
        ),
        customdata=np.stack(
            [
                events["team_name"],
                events["player_name"],
                events["event_type"],
                events["outcome_label"],
                events["time_label"],
                events["xg"].round(3) if "xg" in events.columns else [0] * len(events),
            ],
            axis=-1,
        ),
        name="Event cloud",
    )
    fig.add_trace(cloud)

    def recent_trace(frame: pd.DataFrame) -> go.Scatter:
        xs: list[float | None] = []
        ys: list[float | None] = []
        for row in frame.itertuples():
            xs.extend([row.x_plot, row.end_x_plot, None])
            ys.extend([row.y_plot, row.end_y_plot, None])
        return go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="rgba(248,244,234,0.30)", width=2.5),
            hoverinfo="skip",
            name="Rastro",
        )

    def action_line(event: pd.Series) -> go.Scatter:
        return go.Scatter(
            x=[event["x_plot"], event["end_x_plot"]],
            y=[event["y_plot"], event["end_y_plot"]],
            mode="lines",
            line=dict(color=team_colors.get(event["team_name"], "#ffd166"), width=6),
            hoverinfo="skip",
            name="Acción actual",
        )

    def action_points(event: pd.Series) -> list[go.Scatter]:
        symbol = "star" if bool(event["is_goal"]) else ("diamond" if bool(event["is_shot"]) else "circle")
        common = [
            event["team_name"],
            event["player_name"],
            event["event_type"],
            event["outcome_label"],
            event["time_label"],
        ]
        start = go.Scatter(
            x=[event["x_plot"]],
            y=[event["y_plot"]],
            mode="markers",
            marker=dict(
                size=14,
                color=team_colors.get(event["team_name"], "#ffd166"),
                line=dict(color="#f8f4ea", width=1.4),
                symbol="circle",
            ),
            customdata=[common],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Team: %{customdata[0]}<br>"
                "Action: %{customdata[2]}<br>"
                "Outcome: %{customdata[3]}<br>"
                "Time: %{customdata[4]}<extra></extra>"
            ),
            name="Start",
        )
        end = go.Scatter(
            x=[event["end_x_plot"]],
            y=[event["end_y_plot"]],
            mode="markers+text",
            marker=dict(
                size=15,
                color=team_colors.get(event["team_name"], "#ffd166"),
                line=dict(color="#081018", width=1.4),
                symbol=symbol,
            ),
            text=[event["event_type"]],
            textposition="top center",
            textfont=dict(color="#f8f4ea", size=11),
            customdata=[common],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Team: %{customdata[0]}<br>"
                "Action: %{customdata[2]}<br>"
                "Outcome: %{customdata[3]}<br>"
                "Time: %{customdata[4]}<extra></extra>"
            ),
            name="End",
        )
        return [start, end]

    first_event = events.iloc[0]
    fig.add_trace(recent_trace(events.iloc[:1]))
    fig.add_trace(action_line(first_event))
    for trace in action_points(first_event):
        fig.add_trace(trace)

    frames = []
    slider_steps = []
    for idx in range(len(events)):
        event = events.iloc[idx]
        recent = events.iloc[max(0, idx - trail_length + 1) : idx + 1]
        name = str(idx)
        frames.append(
            go.Frame(
                name=name,
                data=[
                    recent_trace(recent),
                    action_line(event),
                    *action_points(event),
                ],
                traces=[1, 2, 3, 4],
            )
        )
        slider_steps.append(
            {
                "args": [
                    [name],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": event["time_label"],
                "method": "animate",
            }
        )

    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.01,
                "y": 1.08,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Reproducir",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 380, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pausa",
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
                "currentvalue": {"prefix": "Tiempo de repetición: ", "font": {"color": "#f8f4ea"}},
                "steps": slider_steps,
            }
        ],
    )
    return fig


def build_momentum_bars(events: pd.DataFrame) -> list[html.Div]:
    if events.empty:
        return [html.Div("Sin datos", className="empty-note")]
    summary = (
        events.groupby("team_name")
        .agg(actions=("id", "count"), shots=("is_shot", "sum"), goals=("is_goal", "sum"))
        .sort_values(["goals", "shots", "actions"], ascending=False)
        .reset_index()
    )
    max_actions = max(int(summary["actions"].max()), 1)
    rows = []
    for idx, row in summary.iterrows():
        color = TEAM_COLORS[idx % len(TEAM_COLORS)]
        width = max(8, int((row["actions"] / max_actions) * 100))
        rows.append(
            html.Div(
                className="momentum-row",
                children=[
                    html.Div(row["team_name"], className="momentum-label"),
                    html.Div(
                        className="momentum-track",
                        children=html.Div(
                            className="momentum-fill",
                            style={"width": f"{width}%", "background": color},
                        ),
                    ),
                    html.Div(f"{int(row['actions'])} actos", className="momentum-value"),
                ],
            )
        )
    return rows


app = Dash(
    __name__,
    assets_folder=str(BASE_DIR / "assets"),
    title="Premier League Match Lab V2",
)
server = app.server


app.layout = html.Div(
    className="sportsbook-shell",
    children=[
        html.Div(
            className="topbar",
            children=[
                html.Div(
                    className="brand-block",
                    children=[
                        html.Div("Premier League Match Lab", className="brand-title"),
                        html.Div("Versión 2 | Repetición e inteligencia de partidos estilo Sportsbook", className="brand-subtitle"),
                    ],
                ),
                html.Div(
                    className="topbar-badges",
                    children=[
                        html.Span("Dash V2", className="badge"),
                        html.Span("Repetición", className="badge"),
                        html.Span("Datos locales", className="badge"),
                    ],
                ),
            ],
        ),
        html.Div(
            className="content-grid",
            children=[
                html.Div(
                    className="left-rail panel",
                    children=[
                        html.H3("Sala de control", className="panel-title"),
                        html.Label("Modo de Análisis", className="control-label"),
                        dcc.RadioItems(
                            id="mode-toggle",
                            options=[
                                {"label": "Repetición Histórica", "value": "historical"},
                                {"label": "Simulador de Partido", "value": "simulator"},
                            ],
                            value="historical",
                            className="radio-group",
                            inputStyle={"marginRight": "8px"},
                        ),
                        html.Div(
                            id="historical-controls",
                            children=[
                                html.Label("Partido", className="control-label"),
                                dcc.Dropdown(
                                    id="match-select",
                                    options=MATCH_OPTIONS,
                                    value=MATCH_OPTIONS[0]["value"],
                                    clearable=False,
                                ),
                            ],
                        ),
                        html.Div(
                            id="simulator-controls",
                            style={"display": "none"},
                            children=[
                                html.Label("Equipo Local", className="control-label"),
                                dcc.Dropdown(
                                    id="sim-home-team",
                                    options=[{"label": t, "value": t} for t in ALL_TEAMS],
                                    value=ALL_TEAMS[0],
                                    clearable=False,
                                ),
                                html.Label("Equipo Visitante", className="control-label"),
                                dcc.Dropdown(
                                    id="sim-away-team",
                                    options=[{"label": t, "value": t} for t in ALL_TEAMS],
                                    value=ALL_TEAMS[1],
                                    clearable=False,
                                ),
                                html.Label("Árbitro", className="control-label"),
                                dcc.Dropdown(
                                    id="sim-referee",
                                    options=[{"label": r, "value": r} for r in GOAL_MODELS["referees"]],
                                    value=GOAL_MODELS["referees"][0],
                                    clearable=False,
                                ),
                            ],
                        ),
                        html.Label("Equipos", className="control-label"),
                        dcc.Dropdown(id="team-filter", multi=True),
                        html.Label("Jugador destacado", className="control-label"),
                        dcc.Dropdown(id="player-filter", clearable=False),
                        html.Label("Tipos de evento", className="control-label"),
                        dcc.Dropdown(id="event-type-filter", multi=True),
                        html.Label("Ventana de tiempo (min)", className="control-label"),
                        dcc.RangeSlider(id="minute-range", min=0, max=120, step=1, value=[0, 95]),
                        html.Label("Pasos de repetición", className="control-label"),
                        dcc.Slider(id="max-steps", min=20, max=220, step=10, value=80),
                        html.Label("Longitud de rastro", className="control-label"),
                        dcc.Slider(id="trail-length", min=2, max=20, step=1, value=8),
                        html.Div(
                            className="switch-row",
                            children=[
                                dcc.Checklist(
                                    id="replay-flags",
                                    options=[
                                        {"label": "Solo exitosos", "value": "successful"},
                                        {"label": "Incluir Inicio/Fin", "value": "stops"},
                                    ],
                                    value=[],
                                    inputStyle={"marginRight": "8px", "marginLeft": "0"},
                                )
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="center-stage",
                    children=[
                        html.Div(id="match-header", className="match-header panel"),
                        html.Div(id="prediction-strip", className="prediction-strip"),
                        html.Div(
                            className="center-panels",
                            children=[
                                html.Div(
                                    className="replay-panel panel",
                                    children=[
                                        html.Div(
                                            className="panel-head",
                                            children=[
                                                html.H3("Repetición en vivo", className="panel-title"),
                                                html.Div("Emulación de partido basada en eventos", className="panel-caption"),
                                            ],
                                        ),
                                        dcc.Graph(id="replay-graph", config={"displaylogo": False}, className="replay-graph"),
                                    ],
                                ),
                                html.Div(
                                    className="summary-grid",
                                    children=[
                                        html.Div(id="summary-cards", className="summary-cards"),
                                        html.Div(
                                            className="panel compact-panel",
                                            children=[
                                                html.H3("Impulso (Momentum)", className="panel-title"),
                                                html.Div(id="momentum-bars", className="momentum-bars"),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="right-rail",
                    children=[
                        html.Div(
                            className="panel",
                            children=[
                                html.H3("Acciones recientes", className="panel-title"),
                                dash_table.DataTable(
                                    id="recent-events-table",
                                    columns=[
                                        {"name": "Tiempo", "id": "time_label"},
                                        {"name": "Equipo", "id": "team_name"},
                                        {"name": "Jugador", "id": "player_name"},
                                        {"name": "Acción", "id": "event_type"},
                                        {"name": "Resultado", "id": "outcome_label"},
                                    ],
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "backgroundColor": "#111827",
                                        "color": "#f8f4ea",
                                        "border": "1px solid rgba(255,255,255,0.04)",
                                        "fontFamily": "Arial, sans-serif",
                                        "fontSize": "12px",
                                        "padding": "8px",
                                        "textAlign": "left",
                                        "whiteSpace": "normal",
                                        "height": "auto",
                                    },
                                    style_header={
                                        "backgroundColor": "#0f1723",
                                        "color": "#f8f4ea",
                                        "fontWeight": "700",
                                        "border": "1px solid rgba(255,255,255,0.08)",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            className="panel",
                            children=[
                                html.H3("Resumen de equipo", className="panel-title"),
                                dash_table.DataTable(
                                    id="team-summary-table",
                                    columns=[
                                        {"name": "Equipo", "id": "team_name"},
                                        {"name": "Acciones", "id": "actions"},
                                        {"name": "Tiros", "id": "shots"},
                                        {"name": "Goles", "id": "goals"},
                                    ],
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "backgroundColor": "#111827",
                                        "color": "#f8f4ea",
                                        "border": "1px solid rgba(255,255,255,0.04)",
                                        "fontFamily": "Arial, sans-serif",
                                        "fontSize": "12px",
                                        "padding": "8px",
                                        "textAlign": "left",
                                    },
                                    style_header={
                                        "backgroundColor": "#0f1723",
                                        "color": "#f8f4ea",
                                        "fontWeight": "700",
                                        "border": "1px solid rgba(255,255,255,0.08)",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@callback(
    Output("historical-controls", "style"),
    Output("simulator-controls", "style"),
    Input("mode-toggle", "value"),
)
def toggle_controls(mode: str):
    if mode == "simulator":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@callback(
    Output("team-filter", "options"),
    Output("team-filter", "value"),
    Output("player-filter", "options"),
    Output("player-filter", "value"),
    Output("event-type-filter", "options"),
    Output("event-type-filter", "value"),
    Output("minute-range", "value"),
    Input("mode-toggle", "value"),
    Input("match-select", "value"),
    Input("sim-home-team", "value"),
    Input("sim-away-team", "value"),
)
def update_filters(mode: str, match_id: int, sim_home: str, sim_away: str):
    if mode == "historical":
        events = EVENTS.loc[EVENTS["match_id"] == int(match_id)].copy()
    else:
        # For simulator, show teams and players from both selected teams
        events = EVENTS.loc[EVENTS["team_name"].isin([sim_home, sim_away])].copy()

    team_options = [{"label": team, "value": team} for team in sorted(events["team_name"].dropna().unique().tolist())]
    if mode == "simulator":
        team_values = [sim_home, sim_away]
    else:
        team_values = [option["value"] for option in team_options]

    player_options = [{"label": "Todos los jugadores", "value": "All players"}] + [
        {"label": player, "value": player}
        for player in sorted(events["player_name"].dropna().unique().tolist())
    ]
    event_type_options = [
        {"label": event_type, "value": event_type}
        for event_type in sorted(events["event_type"].dropna().unique().tolist())
    ]
    event_type_values = [
        event_type
        for event_type in DEFAULT_EVENT_TYPES
        if event_type in [option["value"] for option in event_type_options]
    ]

    max_minute = int(events["minute"].max()) if not events.empty else 95
    return (
        team_options,
        team_values,
        player_options,
        "All players",
        event_type_options,
        event_type_values,
        [0, max(95, max_minute)],
    )


@callback(
    Output("match-header", "children"),
    Output("prediction-strip", "children"),
    Output("replay-graph", "figure"),
    Output("summary-cards", "children"),
    Output("momentum-bars", "children"),
    Output("recent-events-table", "data"),
    Output("team-summary-table", "data"),
    Input("match-select", "value"),
    Input("team-filter", "value"),
    Input("player-filter", "value"),
    Input("event-type-filter", "value"),
    Input("minute-range", "value"),
    Input("max-steps", "value"),
    Input("trail-length", "value"),
    Input("replay-flags", "value"),
    Input("mode-toggle", "value"),
    Input("sim-home-team", "value"),
    Input("sim-away-team", "value"),
    Input("sim-referee", "value"),
)
def update_dashboard(
    match_id: int,
    team_filter: list[str] | None,
    player_filter: str | None,
    event_type_filter: list[str] | None,
    minute_range: list[int] | None,
    max_steps: int,
    trail_length: int,
    flags: list[str] | None,
    mode: str,
    sim_home: str,
    sim_away: str,
    sim_referee: str,
):
    flags = flags or []
    if mode == "historical":
        raw_events = EVENTS.loc[EVENTS["match_id"] == int(match_id)]
        match_row = MATCHES.loc[MATCHES["id"] == int(match_id)].iloc[0]
        prediction = build_prediction_payload(match_row, raw_events)
        header_score = f"{int(match_row['fthg'])} - {int(match_row['ftag'])}"
        header_teams = (match_row["home_team"], match_row["away_team"])
        season_label = match_row["season_label"]
        referee_label = match_row["referee"]
        result_label = f"Resultado {match_row['ftr']}"
    else:
        raw_events = get_simulated_events(sim_home, sim_away)
        proj = simulate_match_outlook(sim_home, sim_away, sim_referee)
        prediction = {
            "probabilities": proj["probabilities"],
            "likely_winner": proj["likely_winner"],
            "expected_goals": proj["total_goals"],
            "goal_pressure_team": sim_home if proj["total_goals"] > 1.5 else sim_away,
            "sample_size": 1
        }
        header_score = f"VS"
        header_teams = (sim_home, sim_away)
        season_label = "Modo Simulación"
        referee_label = sim_referee
        result_label = "Proyección"

    filtered = filter_match_events(
        df=raw_events,
        teams=team_filter,
        player_name=player_filter,
        event_types=event_type_filter,
        minute_range=minute_range or [0, 95],
        successful_only="successful" in flags,
        include_stops="stops" in flags,
        max_steps=max_steps,
    )

    header = [
        html.Div(
            className="match-core",
            children=[
                html.Div(header_teams[0], className="team-name"),
                html.Div(header_score, className="scoreline"),
                html.Div(header_teams[1], className="team-name"),
            ],
        ),
        html.Div(
            className="match-meta",
            children=[
                html.Span(season_label, className="meta-chip"),
                html.Span(referee_label, className="meta-chip"),
                html.Span(result_label, className="meta-chip"),
                html.Span(f"Acciones de repetición {len(filtered)}", className="meta-chip highlight-chip"),
            ],
        ),
    ]

    prediction_strip = [
        html.Div(
            className="prediction-card winner-card",
            children=[
                html.Div("Perspectiva de Ganador", className="prediction-label"),
                html.Div(prediction["likely_winner"], className="prediction-value"),
                html.Div(
                    f"{max(prediction['probabilities'].values()):.1%} de confianza",
                    className="prediction-subvalue",
                ),
            ],
        ),
        html.Div(
            className="prediction-card goals-card",
            children=[
                html.Div("Goles esperados", className="prediction-label"),
                html.Div(f"{prediction['expected_goals']:.2f}", className="prediction-value"),
                html.Div(
                    f"Basado en {prediction['sample_size']} partidos comparables",
                    className="prediction-subvalue",
                ),
            ],
        ),
        html.Div(
            className="prediction-card prob-card",
            children=[
                html.Div("Probabilidades de resultado", className="prediction-label"),
                html.Div(
                    className="probability-row",
                    children=[
                        html.Span(f"Local {prediction['probabilities']['H']:.0%}", className="prob-pill"),
                        html.Span(f"Empate {prediction['probabilities']['D']:.0%}", className="prob-pill"),
                        html.Span(f"Visitante {prediction['probabilities']['A']:.0%}", className="prob-pill"),
                    ],
                ),
                html.Div(
                    f"Tendencia de presión: {prediction['goal_pressure_team']}",
                    className="prediction-subvalue",
                ),
            ],
        ),
    ]

    summary_cards = [
        html.Div(className="summary-card", children=[html.Div("Acciones", className="summary-label"), html.Div(f"{len(filtered)}", className="summary-value")]),
        html.Div(className="summary-card", children=[html.Div("Tiros", className="summary-label"), html.Div(f"{int(filtered['is_shot'].sum()) if not filtered.empty else 0}", className="summary-value")]),
        html.Div(className="summary-card", children=[html.Div("Goles", className="summary-label"), html.Div(f"{int(filtered['is_goal'].sum()) if not filtered.empty else 0}", className="summary-value")]),
        html.Div(className="summary-card", children=[html.Div("Jugadores", className="summary-label"), html.Div(f"{filtered['player_name'].nunique() if not filtered.empty else 0}", className="summary-value")]),
    ]

    recent = (
        filtered[["time_label", "team_name", "player_name", "event_type", "outcome_label"]]
        .sort_values("time_label", ascending=False)
        .head(14)
        .to_dict("records")
    )
    team_summary = (
        filtered.groupby("team_name")
        .agg(actions=("id", "count"), shots=("is_shot", "sum"), goals=("is_goal", "sum"))
        .sort_values(["goals", "shots", "actions"], ascending=False)
        .reset_index()
        .to_dict("records")
    )

    return (
        header,
        prediction_strip,
        build_replay_figure(filtered, trail_length=trail_length),
        summary_cards,
        build_momentum_bars(filtered),
        recent,
        team_summary,
    )


if __name__ == "__main__":
    app.run(debug=True)
