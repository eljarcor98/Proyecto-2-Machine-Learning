from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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
HOME_FEATURES = ["hs", "hst", "hc", "hf", "hy", "hr"]
AWAY_FEATURES = ["as_", "ast", "ac", "af", "ay", "ar"]


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


@st.cache_resource(show_spinner=False)
def fit_xg_model() -> dict[str, object]:
    shots = load_shot_features().dropna(subset=XG_FEATURES + ["is_goal"]).copy()
    X = shots[XG_FEATURES]
    y = shots["is_goal"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    test_scores = np.clip(model.predict(X_test), 0, 1)
    return {
        "model": model,
        "test_scores": test_scores,
        "y_test": y_test.reset_index(drop=True),
    }


@st.cache_data(show_spinner=False)
def compute_threshold_metrics() -> pd.DataFrame:
    payload = fit_xg_model()
    scores = payload["test_scores"]
    y_test = payload["y_test"]
    rows = []
    for threshold in np.arange(0.05, 0.55, 0.05):
        preds = (scores > threshold).astype(int)
        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        tn = int(((preds == 0) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / len(y_test)
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
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def attach_xg_scores() -> pd.DataFrame:
    shot_map = build_shot_dataset().dropna(subset=XG_FEATURES).copy()
    model = fit_xg_model()["model"]
    shot_map["xg_linear"] = np.clip(model.predict(shot_map[XG_FEATURES]), 0, 1)
    return shot_map


@st.cache_resource(show_spinner=False)
def fit_goal_models() -> dict[str, object]:
    matches = load_matches().copy()

    home_model = LinearRegression()
    away_model = LinearRegression()
    home_model.fit(matches[HOME_FEATURES].astype(float), matches["fthg"].astype(float))
    away_model.fit(matches[AWAY_FEATURES].astype(float), matches["ftag"].astype(float))

    home_pred = home_model.predict(matches[HOME_FEATURES].astype(float))
    away_pred = away_model.predict(matches[AWAY_FEATURES].astype(float))

    return {
        "home_model": home_model,
        "away_model": away_model,
        "home_r2": r2_score(matches["fthg"], home_pred),
        "away_r2": r2_score(matches["ftag"], away_pred),
        "home_rmse": math.sqrt(mean_squared_error(matches["fthg"], home_pred)),
        "away_rmse": math.sqrt(mean_squared_error(matches["ftag"], away_pred)),
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


def poisson_pmf(lmbda: float, goals: int) -> float:
    lmbda = max(lmbda, 0.01)
    return math.exp(-lmbda) * (lmbda**goals) / math.factorial(goals)


def outcome_probabilities(home_xg: float, away_xg: float, max_goals: int = 7) -> dict[str, float]:
    home_probs = [poisson_pmf(home_xg, g) for g in range(max_goals + 1)]
    away_probs = [poisson_pmf(away_xg, g) for g in range(max_goals + 1)]

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    for hg, home_prob in enumerate(home_probs):
        for ag, away_prob in enumerate(away_probs):
            joint = home_prob * away_prob
            if hg > ag:
                p_home += joint
            elif hg == ag:
                p_draw += joint
            else:
                p_away += joint

    total = p_home + p_draw + p_away
    return {
        "H": p_home / total,
        "D": p_draw / total,
        "A": p_away / total,
    }


def build_matchup_features(home_team: str, away_team: str) -> dict[str, pd.Series]:
    profiles = team_profiles()
    home_attack = profiles["home_attack"].loc[home_team]
    home_defense = profiles["home_defense"].loc[home_team]
    away_attack = profiles["away_attack"].loc[away_team]
    away_defense = profiles["away_defense"].loc[away_team]

    home_row = pd.Series(
        {
            feature: float((home_attack[feature] + away_defense[feature]) / 2)
            for feature in HOME_FEATURES
        }
    )
    away_row = pd.Series(
        {
            feature: float((away_attack[feature] + home_defense[feature]) / 2)
            for feature in AWAY_FEATURES
        }
    )
    return {"home": home_row, "away": away_row}


def matchup_projection(home_team: str, away_team: str) -> dict[str, object]:
    models = fit_goal_models()
    features = build_matchup_features(home_team, away_team)
    home_goals = float(
        np.clip(
            models["home_model"].predict(pd.DataFrame([features["home"]], columns=HOME_FEATURES))[0],
            0,
            5,
        )
    )
    away_goals = float(
        np.clip(
            models["away_model"].predict(pd.DataFrame([features["away"]], columns=AWAY_FEATURES))[0],
            0,
            5,
        )
    )
    probs = outcome_probabilities(home_goals, away_goals)
    return {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "probabilities": probs,
        "home_features": features["home"],
        "away_features": features["away"],
    }


def draw_pitch() -> go.Figure:
    fig = go.Figure()
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
        plot_bgcolor="#132033",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-4, 104], visible=False),
        yaxis=dict(range=[-4, 104], visible=False, scaleanchor="x", scaleratio=1),
        height=560,
    )
    return fig


def standings_badge(team_name: str) -> str:
    standings = load_standings().set_index("team_name")
    if team_name not in standings.index:
        return "No standings"
    row = standings.loc[team_name]
    return f"#{int(row['pos'])} | {int(row['pts'])} pts | GD {int(row['gd'])}"


matches = load_matches()
shot_map = attach_xg_scores()
thresholds = compute_threshold_metrics()
models = fit_goal_models()
standings = load_standings()

teams = sorted(set(matches["home_team"]).union(matches["away_team"]))


st.markdown(
    """
    <div class="panel">
        <div class="eyebrow">Arnold Avances</div>
        <div class="hero-title">Premier League Match Lab</div>
        <div class="hero-copy">
            Dashboard local inspirado en la lectura rapida de WhoScored: mucho contexto arriba,
            comparacion rapida entre equipos, shot map y performance de modelos en una sola vista.
            Esta version usa los avances actuales de regresion lineal para goles y xG lineal por tiro.
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
    max_shots = st.slider("Shots shown on map", min_value=50, max_value=1500, value=350, step=50)
    threshold_selected = st.select_slider(
        "Goal threshold",
        options=thresholds["threshold"].round(2).tolist(),
        value=0.20,
    )
    st.caption("The chosen threshold is used to explain the current xG prototype.")


overview_1, overview_2, overview_3, overview_4 = st.columns(4)
overview_1.metric("Matches loaded", f"{len(matches)}")
overview_2.metric("Shots with features", f"{len(load_shot_features()):,}")
overview_3.metric("Home goal R2", f"{models['home_r2']:.3f}")
overview_4.metric("Away goal R2", f"{models['away_r2']:.3f}")

tab_overview, tab_predictor, tab_shots, tab_performance, tab_eda = st.tabs(
    ["Overview", "Match Predictor", "Shot Lab", "Performance", "EDA"]
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
        projection = matchup_projection(home_team, away_team)
        probs = projection["probabilities"]

        st.subheader(f"{home_team} vs {away_team}")
        st.caption(
            f"{home_team}: {standings_badge(home_team)} | "
            f"{away_team}: {standings_badge(away_team)}"
        )

        met_1, met_2, met_3, met_4, met_5 = st.columns(5)
        met_1.metric("Expected home goals", f"{projection['home_goals']:.2f}")
        met_2.metric("Expected away goals", f"{projection['away_goals']:.2f}")
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
                {
                    "home_inputs": projection["home_features"].round(2),
                    "away_inputs": projection["away_features"].round(2),
                }
            ).reset_index(names="feature")
            st.dataframe(feature_table, use_container_width=True, hide_index=True)

        st.caption(
            "Prototype note: this section uses team historical averages to build the feature vector "
            "that feeds the current linear goal models."
        )


with tab_shots:
    st.subheader("Shot map with linear xG color scale")
    shot_view = shot_map.copy()
    if shot_team_filter:
        shot_view = shot_view[shot_view["team_name"].isin(shot_team_filter)]
    shot_view = shot_view.dropna(subset=["x_plot", "y_plot"]).sort_values("xg_linear", ascending=False).head(max_shots)

    pitch = draw_pitch()
    if not shot_view.empty:
        pitch.add_trace(
            go.Scatter(
                x=shot_view["x_plot"],
                y=shot_view["y_plot"],
                mode="markers",
                marker=dict(
                    size=np.where(shot_view["is_goal"] == 1, 13, 8),
                    color=shot_view["xg_linear"],
                    colorscale="YlOrRd",
                    cmin=0,
                    cmax=1,
                    line=dict(width=1, color="#f7f2e8"),
                    symbol=np.where(shot_view["is_goal"] == 1, "star", "circle"),
                    colorbar=dict(title="xG"),
                ),
                customdata=np.stack(
                    [
                        shot_view["player_name"].fillna("Unknown"),
                        shot_view["team_name"].fillna("Unknown"),
                        shot_view["minute"].fillna(0),
                        shot_view["xg_linear"].round(3),
                        shot_view["is_goal"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Team: %{customdata[1]}<br>"
                    "Minute: %{customdata[2]}<br>"
                    "xG: %{customdata[3]}<br>"
                    "Goal: %{customdata[4]}<extra></extra>"
                ),
                name="Shots",
            )
        )
    st.plotly_chart(pitch, use_container_width=True)

    shot_stats_1, shot_stats_2, shot_stats_3, shot_stats_4 = st.columns(4)
    shot_stats_1.metric("Shots shown", f"{len(shot_view)}")
    shot_stats_2.metric("Goals shown", f"{int(shot_view['is_goal'].sum()) if not shot_view.empty else 0}")
    shot_stats_3.metric("Average xG", f"{shot_view['xg_linear'].mean():.3f}" if not shot_view.empty else "0.000")
    shot_stats_4.metric("High-value shots", f"{int((shot_view['xg_linear'] >= 0.20).sum()) if not shot_view.empty else 0}")

    top_shots = (
        shot_view[["player_name", "team_name", "xg_linear", "is_goal", "distance", "angle"]]
        .sort_values("xg_linear", ascending=False)
        .head(15)
        .rename(columns={"xg_linear": "xg"})
    )
    st.dataframe(top_shots, use_container_width=True, hide_index=True)


with tab_performance:
    st.subheader("Current model performance")
    perf_1, perf_2, perf_3, perf_4 = st.columns(4)
    perf_1.metric("Home RMSE", f"{models['home_rmse']:.3f}")
    perf_2.metric("Away RMSE", f"{models['away_rmse']:.3f}")
    perf_3.metric("Home R2", f"{models['home_r2']:.3f}")
    perf_4.metric("Away R2", f"{models['away_r2']:.3f}")

    selected_row = thresholds.loc[thresholds["threshold"] == float(threshold_selected)].iloc[0]
    perf_5, perf_6, perf_7, perf_8 = st.columns(4)
    perf_5.metric("xG accuracy", f"{selected_row['accuracy']:.3f}")
    perf_6.metric("xG precision", f"{selected_row['precision']:.3f}")
    perf_7.metric("xG recall", f"{selected_row['recall']:.3f}")
    perf_8.metric("xG F1", f"{selected_row['f1']:.3f}")

    threshold_fig = px.line(
        thresholds,
        x="threshold",
        y=["precision", "recall", "f1"],
        markers=True,
        labels={"value": "Metric", "threshold": "Threshold", "variable": "Series"},
    )
    threshold_fig.add_vline(x=float(threshold_selected), line_dash="dash", line_color="#a61e2a")
    threshold_fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    st.plotly_chart(threshold_fig, use_container_width=True)

    conf_1, conf_2, conf_3, conf_4 = st.columns(4)
    conf_1.metric("TP", int(selected_row["tp"]))
    conf_2.metric("FP", int(selected_row["fp"]))
    conf_3.metric("TN", int(selected_row["tn"]))
    conf_4.metric("FN", int(selected_row["fn"]))

    st.dataframe(
        thresholds[["threshold", "accuracy", "precision", "recall", "f1"]].round(3),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "The xG prototype uses the current linear regression workflow from this branch. "
        "It is useful for the dashboard narrative now, and can later be swapped for the logistic version."
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
            .agg(avg_xg=("xg_linear", "mean"), goals=("is_goal", "sum"), shots=("id", "count"))
            .reset_index()
        )
        fig_band = px.bar(
            xg_by_band,
            x="distance_band",
            y="avg_xg",
            color="avg_xg",
            color_continuous_scale=["#ffe8d6", "#ffb703", "#d00000"],
            labels={"distance_band": "Distance band", "avg_xg": "Average xG"},
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
