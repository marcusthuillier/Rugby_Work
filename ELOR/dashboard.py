"""
dashboard.py
Streamlit dashboard for Rugby ELO analytics.

Run: streamlit run dashboard.py
"""

import os
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASETS = os.path.join("Datasets")
ELO_CSV         = os.path.join(DATASETS, "ELO.csv")
RANKINGS_CSV    = os.path.join(DATASETS, "get_rank.csv")
PERFORMANCE_CSV = os.path.join(DATASETS, "performance.csv")
UPSETS_CSV      = "upsets.csv"
HISTORY_CSV     = "elo_history.csv"
ERA_CSV         = "era_rankings.csv"

HOME_ADVANTAGE = 75


# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_rankings():
    return pd.read_csv(RANKINGS_CSV)

@st.cache_data
def load_performance():
    df = pd.read_csv(PERFORMANCE_CSV)
    df["xWins"] = df["xWins"].round(1)
    df["Wins"] = df["Wins"].round(1)
    df["Performance"] = df["Performance"].round(1)
    return df

@st.cache_data
def load_upsets():
    if not os.path.exists(UPSETS_CSV):
        return None
    return pd.read_csv(UPSETS_CSV)

@st.cache_data
def load_history():
    if not os.path.exists(HISTORY_CSV):
        return None
    df = pd.read_csv(HISTORY_CSV, parse_dates=["Date"])
    return df

@st.cache_data
def load_era():
    if not os.path.exists(ERA_CSV):
        return None
    return pd.read_csv(ERA_CSV)


def probability(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1 + math.pow(10, (rating_b - rating_a) / 400))


# ── App ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rugby ELO Dashboard", page_icon="🏉", layout="wide")
st.title("Rugby ELO Analytics")

tab_rank, tab_time, tab_era, tab_upsets, tab_pred, tab_perf = st.tabs([
    "Rankings",
    "ELO Over Time",
    "Era Dominance",
    "Greatest Upsets",
    "Match Predictor",
    "Performance",
])


# ── Tab 1: Rankings ────────────────────────────────────────────────────────────
with tab_rank:
    st.header("Current ELO Rankings")
    rankings = load_rankings()

    col1, col2 = st.columns([1, 2])
    with col1:
        top_n = st.slider("Show top N teams", 10, len(rankings), 50)
        search = st.text_input("Search team", "")

    df_rank = rankings.copy()
    if search:
        df_rank = df_rank[df_rank["Team"].str.contains(search, case=False, na=False)]
    df_rank = df_rank.head(top_n).reset_index(drop=True)
    df_rank.index += 1

    with col1:
        st.dataframe(df_rank, use_container_width=True)

    with col2:
        fig = px.bar(
            df_rank.head(30), x="ELO", y="Team", orientation="h",
            color="ELO", color_continuous_scale="Blues",
            title="Top 30 Teams by ELO Rating",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: ELO Over Time ───────────────────────────────────────────────────────
with tab_time:
    st.header("ELO Rating Over Time")
    history = load_history()

    if history is None:
        st.warning("Run `python elo_over_time.py` first to generate elo_history.csv.")
    else:
        all_teams = sorted(history["Team"].unique())
        defaults = [t for t in ["New Zealand", "South Africa", "England", "Ireland", "France"]
                    if t in all_teams]
        selected = st.multiselect("Select teams", all_teams, default=defaults)

        year_min = int(history["Date"].dt.year.min())
        year_max = int(history["Date"].dt.year.max())
        year_range = st.slider("Year range", year_min, year_max, (1950, year_max))

        if selected:
            mask = (
                history["Team"].isin(selected) &
                (history["Date"].dt.year >= year_range[0]) &
                (history["Date"].dt.year <= year_range[1])
            )
            plot_df = history[mask]
            fig = px.line(
                plot_df, x="Date", y="ELO", color="Team",
                title="ELO Rating History",
                labels={"ELO": "ELO Rating", "Date": ""},
            )
            fig.update_traces(line={"width": 2})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one team.")


# ── Tab 3: Era Dominance ───────────────────────────────────────────────────────
with tab_era:
    st.header("Era Dominance")
    era_df = load_era()

    if era_df is None:
        st.warning("Run `python era_comparison.py` first to generate era_rankings.csv.")
    else:
        eras = era_df["Era"].unique().tolist()
        selected_era = st.selectbox("Select era", eras)
        top_era = era_df[era_df["Era"] == selected_era].sort_values("Rank")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(top_era[["Rank", "Team", "Avg_ELO"]].reset_index(drop=True),
                         use_container_width=True)
        with col2:
            fig = px.bar(
                top_era, x="Avg_ELO", y="Team", orientation="h",
                color="Avg_ELO", color_continuous_scale="Oranges",
                title=f"Top Teams — {selected_era}",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("All Eras Side-by-Side")
        fig2 = px.bar(
            era_df[era_df["Rank"] <= 5], x="Era", y="Avg_ELO",
            color="Team", barmode="group",
            title="Top 5 Teams per Era (Average ELO)",
            category_orders={"Era": eras},
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Tab 4: Greatest Upsets ─────────────────────────────────────────────────────
with tab_upsets:
    st.header("Greatest Upsets")
    upsets = load_upsets()

    if upsets is None:
        st.warning("Run `python upsets.py` first to generate upsets.csv.")
    else:
        st.caption(
            "Upsets are ranked by how unlikely the winner was to win. "
            "Upset Score = 0.5 - winner's pre-match win probability."
        )
        top_n = st.slider("Show top N upsets", 5, len(upsets), 20)
        show = upsets.head(top_n).copy()
        show["Winner_Prob"] = (show["Winner_Prob"] * 100).round(1).astype(str) + "%"
        show["Upset_Score"] = show["Upset_Score"].round(3)
        st.dataframe(show.reset_index(drop=True), use_container_width=True)

        fig = px.bar(
            upsets.head(top_n),
            x="Upset_Score", y=upsets.head(top_n).apply(
                lambda r: f"{r['Winner']} def. "
                          f"{'Home' if r['Winner'] != r['Home_Team'] else 'Away'} "
                          f"({r['Home_Team']} v {r['Away_Team']}, {r['Date']})", axis=1),
            orientation="h",
            title="Top Upsets by Upset Score",
            labels={"x": "Upset Score", "y": ""},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 5: Match Predictor ─────────────────────────────────────────────────────
with tab_pred:
    st.header("Match Predictor")
    rankings = load_rankings()
    teams = rankings["Team"].tolist()
    ratings = dict(zip(rankings["Team"], rankings["ELO"]))

    col1, col2, col3 = st.columns(3)
    with col1:
        home_team = st.selectbox("Home Team", teams,
                                 index=teams.index("New Zealand") if "New Zealand" in teams else 0)
    with col2:
        away_team = st.selectbox("Away Team", teams,
                                 index=teams.index("South Africa") if "South Africa" in teams else 1)
    with col3:
        neutral = st.checkbox("Neutral venue")
        custom_ha = st.slider("Home advantage (ELO pts)", 0, 200, HOME_ADVANTAGE,
                              disabled=neutral)

    if home_team == away_team:
        st.warning("Select two different teams.")
    else:
        home_elo = ratings[home_team]
        away_elo = ratings[away_team]
        ha = 0 if neutral else custom_ha
        adj_home = home_elo + ha

        home_prob = probability(adj_home, away_elo)
        away_prob = 1.0 - home_prob

        st.subheader("Prediction")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(home_team, f"{home_prob*100:.1f}%", f"ELO {home_elo:.0f}")
        mc2.metric("vs", "")
        mc3.metric(away_team, f"{away_prob*100:.1f}%", f"ELO {away_elo:.0f}")

        fig = go.Figure(go.Bar(
            x=[home_prob * 100, away_prob * 100],
            y=[home_team, away_team],
            orientation="h",
            marker_color=["steelblue", "tomato"],
            text=[f"{home_prob*100:.1f}%", f"{away_prob*100:.1f}%"],
            textposition="inside",
        ))
        fig.update_layout(
            xaxis={"title": "Win Probability (%)", "range": [0, 100]},
            yaxis={"title": ""},
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        elo_diff = home_elo - away_elo
        venue_note = "neutral venue" if neutral else f"{home_team} home (+{custom_ha} ELO)"
        st.caption(
            f"ELO difference: {elo_diff:+.0f} | Venue: {venue_note}"
        )


# ── Tab 6: Performance ─────────────────────────────────────────────────────────
with tab_perf:
    st.header("Over/Under Performers")
    st.caption(
        "Performance = Actual Wins - Expected Wins. "
        "Positive means the team wins more than their ELO suggests."
    )
    perf = load_performance()

    col1, col2 = st.columns(2)
    with col1:
        min_games = st.slider("Minimum expected wins (to filter small samples)", 1, 50, 10)
    with col2:
        top_n = st.slider("Show top/bottom N teams", 5, 30, 15)

    filtered = perf[perf["xWins"] >= min_games].copy()

    over = filtered.nlargest(top_n, "Performance")
    under = filtered.nsmallest(top_n, "Performance")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Over-performers")
        st.dataframe(over.reset_index(drop=True), use_container_width=True)
    with c2:
        st.subheader("Under-performers")
        st.dataframe(under.reset_index(drop=True), use_container_width=True)

    plot_df = pd.concat([over, under]).drop_duplicates().sort_values("Performance", ascending=False)
    fig = px.bar(
        plot_df, x="Performance", y="Team", orientation="h",
        color="Performance", color_continuous_scale="RdYlGn",
        title="Team Performance vs Expectation",
        labels={"Performance": "Actual - Expected Wins"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
