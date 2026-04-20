from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dash_table, dcc, html


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

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
    events = events.sort_values(["match_id", "clock_seconds", "id"]).reset_index(drop=True)
    return events


MATCHES = load_matches()
EVENTS = load_events()


def build_match_options(matches: pd.DataFrame) -> list[dict[str, str | int]]:
    options = []
    for row in matches.sort_values(["date", "time", "id"]).itertuples():
        label = (
            f"{row.home_team} vs {row.away_team} | "
            f"{pd.Timestamp(row.date).strftime('%d %b %Y')} | ID {row.id}"
        )
        options.append({"label": label, "value": int(row.id)})
    return options


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
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis=dict(range=[-4, 104], visible=False),
        yaxis=dict(range=[-4, 104], visible=False, scaleanchor="x", scaleratio=1),
        hovermode="closest",
        height=720,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(15,23,35,0.88)",
            font=dict(color="#f8f4ea"),
        ),
    )
    return fig


def filter_match_events(
    match_id: int,
    teams: list[str] | None,
    player_name: str | None,
    event_types: list[str] | None,
    minute_range: list[int] | tuple[int, int],
    successful_only: bool,
    include_stops: bool,
    max_steps: int,
) -> pd.DataFrame:
    frame = EVENTS.loc[EVENTS["match_id"] == int(match_id)].copy()
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
        customdata=np.stack(
            [
                events["team_name"],
                events["player_name"],
                events["event_type"],
                events["outcome_label"],
                events["time_label"],
            ],
            axis=-1,
        ),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Team: %{customdata[0]}<br>"
            "Action: %{customdata[2]}<br>"
            "Outcome: %{customdata[3]}<br>"
            "Time: %{customdata[4]}<extra></extra>"
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
            name="Trail",
        )

    def action_line(event: pd.Series) -> go.Scatter:
        return go.Scatter(
            x=[event["x_plot"], event["end_x_plot"]],
            y=[event["y_plot"], event["end_y_plot"]],
            mode="lines",
            line=dict(color=team_colors.get(event["team_name"], "#ffd166"), width=6),
            hoverinfo="skip",
            name="Current action",
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
                        "label": "Play",
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
                "currentvalue": {"prefix": "Replay time: ", "font": {"color": "#f8f4ea"}},
                "steps": slider_steps,
            }
        ],
    )
    return fig


def build_momentum_bars(events: pd.DataFrame) -> list[html.Div]:
    if events.empty:
        return [html.Div("No data", className="empty-note")]
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
                    html.Div(f"{int(row['actions'])} acts", className="momentum-value"),
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
                        html.Div("Version 2 | Sportsbook-style replay and match intelligence", className="brand-subtitle"),
                    ],
                ),
                html.Div(
                    className="topbar-badges",
                    children=[
                        html.Span("Dash V2", className="badge"),
                        html.Span("Replay", className="badge"),
                        html.Span("Local data only", className="badge"),
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
                        html.H3("Control room", className="panel-title"),
                        html.Label("Match", className="control-label"),
                        dcc.Dropdown(
                            id="match-select",
                            options=MATCH_OPTIONS,
                            value=MATCH_OPTIONS[0]["value"],
                            clearable=False,
                        ),
                        html.Label("Teams", className="control-label"),
                        dcc.Dropdown(id="team-filter", multi=True),
                        html.Label("Player focus", className="control-label"),
                        dcc.Dropdown(id="player-filter", clearable=False),
                        html.Label("Event types", className="control-label"),
                        dcc.Dropdown(id="event-type-filter", multi=True),
                        html.Label("Minute window", className="control-label"),
                        dcc.RangeSlider(id="minute-range", min=0, max=120, step=1, value=[0, 95]),
                        html.Label("Replay steps", className="control-label"),
                        dcc.Slider(id="max-steps", min=20, max=220, step=10, value=80),
                        html.Label("Trail length", className="control-label"),
                        dcc.Slider(id="trail-length", min=2, max=20, step=1, value=8),
                        html.Div(
                            className="switch-row",
                            children=[
                                dcc.Checklist(
                                    id="replay-flags",
                                    options=[
                                        {"label": "Successful only", "value": "successful"},
                                        {"label": "Include Start/End", "value": "stops"},
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
                                                html.H3("Live replay", className="panel-title"),
                                                html.Div("Event-driven match emulation", className="panel-caption"),
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
                                                html.H3("Momentum", className="panel-title"),
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
                                html.H3("Recent actions", className="panel-title"),
                                dash_table.DataTable(
                                    id="recent-events-table",
                                    columns=[
                                        {"name": "Time", "id": "time_label"},
                                        {"name": "Team", "id": "team_name"},
                                        {"name": "Player", "id": "player_name"},
                                        {"name": "Action", "id": "event_type"},
                                        {"name": "Outcome", "id": "outcome_label"},
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
                                html.H3("Team snapshot", className="panel-title"),
                                dash_table.DataTable(
                                    id="team-summary-table",
                                    columns=[
                                        {"name": "Team", "id": "team_name"},
                                        {"name": "Actions", "id": "actions"},
                                        {"name": "Shots", "id": "shots"},
                                        {"name": "Goals", "id": "goals"},
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
    Output("team-filter", "options"),
    Output("team-filter", "value"),
    Output("player-filter", "options"),
    Output("player-filter", "value"),
    Output("event-type-filter", "options"),
    Output("event-type-filter", "value"),
    Output("minute-range", "value"),
    Input("match-select", "value"),
)
def update_filters(match_id: int):
    events = EVENTS.loc[EVENTS["match_id"] == int(match_id)].copy()
    team_options = [{"label": team, "value": team} for team in sorted(events["team_name"].dropna().unique().tolist())]
    team_values = [option["value"] for option in team_options]
    player_options = [{"label": "All players", "value": "All players"}] + [
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
):
    flags = flags or []
    filtered = filter_match_events(
        match_id=match_id,
        teams=team_filter,
        player_name=player_filter,
        event_types=event_type_filter,
        minute_range=minute_range or [0, 95],
        successful_only="successful" in flags,
        include_stops="stops" in flags,
        max_steps=max_steps,
    )

    match_row = MATCHES.loc[MATCHES["id"] == int(match_id)].iloc[0]
    prediction = build_prediction_payload(match_row, filtered)

    header = [
        html.Div(
            className="match-core",
            children=[
                html.Div(match_row["home_team"], className="team-name"),
                html.Div(f"{int(match_row['fthg'])} - {int(match_row['ftag'])}", className="scoreline"),
                html.Div(match_row["away_team"], className="team-name"),
            ],
        ),
        html.Div(
            className="match-meta",
            children=[
                html.Span(match_row["season_label"], className="meta-chip"),
                html.Span(match_row["referee"], className="meta-chip"),
                html.Span(f"Result {match_row['ftr']}", className="meta-chip"),
                html.Span(f"Replay actions {len(filtered)}", className="meta-chip highlight-chip"),
            ],
        ),
    ]

    prediction_strip = [
        html.Div(
            className="prediction-card winner-card",
            children=[
                html.Div("Winner outlook", className="prediction-label"),
                html.Div(prediction["likely_winner"], className="prediction-value"),
                html.Div(
                    f"{max(prediction['probabilities'].values()):.1%} confidence",
                    className="prediction-subvalue",
                ),
            ],
        ),
        html.Div(
            className="prediction-card goals-card",
            children=[
                html.Div("Expected goals", className="prediction-label"),
                html.Div(f"{prediction['expected_goals']:.2f}", className="prediction-value"),
                html.Div(
                    f"Based on {prediction['sample_size']} comparable matches",
                    className="prediction-subvalue",
                ),
            ],
        ),
        html.Div(
            className="prediction-card prob-card",
            children=[
                html.Div("Outcome probabilities", className="prediction-label"),
                html.Div(
                    className="probability-row",
                    children=[
                        html.Span(f"Home {prediction['probabilities']['H']:.0%}", className="prob-pill"),
                        html.Span(f"Draw {prediction['probabilities']['D']:.0%}", className="prob-pill"),
                        html.Span(f"Away {prediction['probabilities']['A']:.0%}", className="prob-pill"),
                    ],
                ),
                html.Div(
                    f"Goal pressure leaning: {prediction['goal_pressure_team']}",
                    className="prediction-subvalue",
                ),
            ],
        ),
    ]

    summary_cards = [
        html.Div(className="summary-card", children=[html.Div("Actions", className="summary-label"), html.Div(f"{len(filtered)}", className="summary-value")]),
        html.Div(className="summary-card", children=[html.Div("Shots", className="summary-label"), html.Div(f"{int(filtered['is_shot'].sum()) if not filtered.empty else 0}", className="summary-value")]),
        html.Div(className="summary-card", children=[html.Div("Goals", className="summary-label"), html.Div(f"{int(filtered['is_goal'].sum()) if not filtered.empty else 0}", className="summary-value")]),
        html.Div(className="summary-card", children=[html.Div("Players", className="summary-label"), html.Div(f"{filtered['player_name'].nunique() if not filtered.empty else 0}", className="summary-value")]),
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
