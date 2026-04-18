"""
comparison_dashboard.py
Interactive Streamlit dashboard comparing:
  Real bookmaker odds vs ELO-implied odds vs XGBoost model odds
  — for Super Rugby (2015-2026) using aussportsbetting.com data

Run: streamlit run comparison_dashboard.py
"""

import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

DATA_PATH = "comparison_dataset.csv"

st.set_page_config(page_title="Odds Comparison Dashboard", page_icon="rugby", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.dropna(subset=["Real_Home_Prob", "ELO_Home_Prob"])
    return df


df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
years = sorted(df["Year"].unique())
year_range = st.sidebar.slider("Year range", int(years[0]), int(years[-1]),
                               (int(years[0]), int(years[-1])))

all_teams = sorted(set(df["Home_Team"].unique()) | set(df["Away_Team"].unique()))
team_filter = st.sidebar.multiselect("Filter by team", all_teams,
                                     placeholder="All teams")

show_model = st.sidebar.checkbox("Show XGBoost model", value=True)

mask = (df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])
if team_filter:
    mask &= df["Home_Team"].isin(team_filter) | df["Away_Team"].isin(team_filter)

data = df[mask].copy()
data_model = data.dropna(subset=["Model_Home_Prob"])

st.title("Odds Comparison: Real vs ELO vs Model")
st.caption(f"Super Rugby {year_range[0]}–{year_range[1]} | {len(data):,} games | "
           f"Source: aussportsbetting.com + ELO pipeline")

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab_cal, tab_scatter, tab_spread, tab_value, tab_roi, tab_games = st.tabs([
    "Calibration",
    "Win Prob Scatter",
    "Spread Comparison",
    "Value Bets",
    "ROI Curves",
    "Game Explorer",
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Calibration — do the probs match actual outcomes?
# ══════════════════════════════════════════════════════════════════════════════
with tab_cal:
    st.header("Calibration: Predicted Home Win Probability vs Actual Win Rate")
    st.caption("A well-calibrated model: the 60% bucket should win ~60% of the time.")

    n_bins = st.slider("Number of probability buckets", 5, 20, 10)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centres = (bins[:-1] + bins[1:]) / 2

    def calibration_curve(prob_col: str, label: str, color: str):
        d = data.dropna(subset=[prob_col, "Home_Win"])
        bucket = pd.cut(d[prob_col], bins=bins, labels=bin_centres)
        grp = d.groupby(bucket, observed=True)["Home_Win"].agg(["mean", "count"]).reset_index()
        grp.columns = ["prob_bucket", "actual_rate", "count"]
        grp["prob_bucket"] = grp["prob_bucket"].astype(float)
        return grp, label, color

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line={"dash": "dash", "color": "grey", "width": 1},
                             name="Perfect calibration"))

    sources = [("Real_Home_Prob", "Real odds (no-vig)", "#e63946")]
    sources.append(("ELO_Home_Prob", "ELO model", "#457b9d"))
    if show_model:
        sources.append(("Model_Home_Prob", "XGBoost model", "#2a9d8f"))

    for col, label, color in sources:
        grp, lbl, clr = calibration_curve(col, label, color)
        fig.add_trace(go.Scatter(
            x=grp["prob_bucket"], y=grp["actual_rate"],
            mode="lines+markers",
            marker={"size": grp["count"].clip(5, 30) * 0.6, "sizemode": "area"},
            line={"color": clr, "width": 2},
            name=lbl,
            hovertemplate="%{y:.1%} win rate (%{marker.size:.0f} games)<extra></extra>",
        ))

    fig.update_layout(xaxis_title="Predicted home win probability",
                      yaxis_title="Actual home win rate",
                      xaxis={"tickformat": ".0%"},
                      yaxis={"tickformat": ".0%"},
                      height=480)
    st.plotly_chart(fig, use_container_width=True)

    # Brier scores
    c1, c2, c3 = st.columns(3)
    hw = data["Home_Win"].dropna()
    for col, label, c in [("Real_Home_Prob","Real odds",c1),
                           ("ELO_Home_Prob","ELO",c2),
                           ("Model_Home_Prob","XGBoost",c3)]:
        d = data.dropna(subset=[col,"Home_Win"])
        bs = ((d[col] - d["Home_Win"])**2).mean()
        c.metric(f"Brier Score — {label}", f"{bs:.4f}",
                 help="Lower = better calibrated. 0.25 = random 50/50 baseline.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Win probability scatter — ELO vs Real, Model vs Real
# ══════════════════════════════════════════════════════════════════════════════
with tab_scatter:
    st.header("Win Probability: ELO vs Real Market")

    d = data.dropna(subset=["Real_Home_Prob", "ELO_Home_Prob"]).copy()
    d["Winner"] = d["Result"].map({"H": "Home", "A": "Away", "D": "Draw"})
    d["ELO_Edge"] = d["ELO_Home_Prob"] - d["Real_Home_Prob"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ELO vs Real")
        fig = px.scatter(
            d, x="Real_Home_Prob", y="ELO_Home_Prob",
            color="Winner",
            color_discrete_map={"Home": "#2a9d8f", "Away": "#e63946", "Draw": "#aaa"},
            hover_data=["Date", "Home_Team", "Away_Team", "Home_Score", "Away_Score"],
            labels={"Real_Home_Prob": "Real home prob (no-vig)",
                    "ELO_Home_Prob": "ELO home prob"},
            opacity=0.6,
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line={"dash": "dash", "color": "grey"})
        fig.update_layout(height=420, xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        corr = d["ELO_Home_Prob"].corr(d["Real_Home_Prob"])
        st.caption(f"Pearson r = {corr:.3f}  |  {len(d):,} games  |  "
                   f"Points above diagonal: ELO more bullish on home than market")

    with col2:
        if show_model:
            dm = data.dropna(subset=["Real_Home_Prob","Model_Home_Prob"]).copy()
            dm["Winner"] = dm["Result"].map({"H":"Home","A":"Away","D":"Draw"})
            st.subheader("Model vs Real")
            fig2 = px.scatter(
                dm, x="Real_Home_Prob", y="Model_Home_Prob",
                color="Winner",
                color_discrete_map={"Home":"#2a9d8f","Away":"#e63946","Draw":"#aaa"},
                hover_data=["Date","Home_Team","Away_Team","Home_Score","Away_Score"],
                labels={"Real_Home_Prob":"Real home prob (no-vig)",
                        "Model_Home_Prob":"XGBoost home prob"},
                opacity=0.6,
            )
            fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                           line={"dash":"dash","color":"grey"})
            fig2.update_layout(height=420, xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            st.plotly_chart(fig2, use_container_width=True)
            corr2 = dm["Model_Home_Prob"].corr(dm["Real_Home_Prob"])
            st.caption(f"Pearson r = {corr2:.3f}  |  {len(dm):,} games")
        else:
            st.info("Enable XGBoost model in sidebar to see this chart.")

    # Edge distribution
    st.subheader("ELO Edge over Real Market (distribution)")
    st.caption("Positive = ELO thinks home team is undervalued by market; negative = overvalued.")
    fig3 = px.histogram(d, x="ELO_Edge", nbins=40,
                        color_discrete_sequence=["#457b9d"],
                        labels={"ELO_Edge": "ELO prob - Real prob (home team)"})
    fig3.add_vline(x=0, line_dash="dash", line_color="grey")
    fig3.update_layout(height=320, xaxis_tickformat="+.0%")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Spread comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab_spread:
    st.header("Spread / Handicap Line Comparison")
    st.caption("Real_Spread: bookmaker closing line (negative = home favoured). "
               "ELO_Spread / Model_Spread: our estimated expected margin.")

    ds = data.dropna(subset=["Real_Spread", "ELO_Spread", "Actual_Margin"]).copy()
    if show_model:
        ds = ds.dropna(subset=["Model_Spread"])

    ds["Spread_Error_ELO"]   = ds["ELO_Spread"] - (-ds["Real_Spread"])
    if show_model and "Model_Spread" in ds.columns:
        ds["Spread_Error_Model"] = ds["Model_Spread"] - (-ds["Real_Spread"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Real Spread vs ELO Spread")
        fig = px.scatter(
            ds, x=-ds["Real_Spread"], y="ELO_Spread",
            color="Actual_Margin",
            color_continuous_scale="RdYlGn",
            hover_data=["Date","Home_Team","Away_Team","Actual_Margin"],
            labels={"x":"Real spread (home favoured = positive)",
                    "ELO_Spread":"ELO expected margin"},
            opacity=0.6,
        )
        fig.add_shape(type="line", x0=-60, y0=-60, x1=60, y1=60,
                      line={"dash":"dash","color":"grey"})
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        mae_elo = ds["Spread_Error_ELO"].abs().mean()
        corr_e  = ds["ELO_Spread"].corr(-ds["Real_Spread"])
        st.caption(f"ELO vs Real correlation: {corr_e:.3f}  |  "
                   f"Mean absolute difference: {mae_elo:.1f} pts")

    with col2:
        if show_model and "Model_Spread" in ds.columns:
            st.subheader("Real Spread vs Model Spread")
            fig2 = px.scatter(
                ds, x=-ds["Real_Spread"], y="Model_Spread",
                color="Actual_Margin",
                color_continuous_scale="RdYlGn",
                hover_data=["Date","Home_Team","Away_Team","Actual_Margin"],
                labels={"x":"Real spread (home favoured = positive)",
                        "Model_Spread":"Model expected margin"},
                opacity=0.6,
            )
            fig2.add_shape(type="line", x0=-60, y0=-60, x1=60, y1=60,
                           line={"dash":"dash","color":"grey"})
            fig2.update_layout(height=420)
            st.plotly_chart(fig2, use_container_width=True)
            mae_m  = ds["Spread_Error_Model"].abs().mean()
            corr_m = ds["Model_Spread"].corr(-ds["Real_Spread"])
            st.caption(f"Model vs Real correlation: {corr_m:.3f}  |  "
                       f"Mean absolute difference: {mae_m:.1f} pts")

    # Actual margin vs predicted margin
    st.subheader("Actual Margin vs Predicted Margin (ELO)")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=ds["ELO_Spread"], y=ds["Actual_Margin"],
                              mode="markers", marker={"color":"#457b9d","opacity":0.5},
                              name="ELO spread", text=ds["Home_Team"]+' v '+ds["Away_Team"]))
    if show_model and "Model_Spread" in ds.columns:
        fig3.add_trace(go.Scatter(x=ds["Model_Spread"], y=ds["Actual_Margin"],
                                  mode="markers", marker={"color":"#2a9d8f","opacity":0.5},
                                  name="Model spread"))
    fig3.add_shape(type="line", x0=-80, y0=-80, x1=80, y1=80,
                   line={"dash":"dash","color":"grey"})
    fig3.update_layout(xaxis_title="Predicted margin", yaxis_title="Actual margin",
                       height=400)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Value bets — where does our model most disagree with the market?
# ══════════════════════════════════════════════════════════════════════════════
with tab_value:
    st.header("Identified Value Bets")
    st.caption("Games where ELO or model gave a significantly different probability "
               "to the real market. Positive edge = our model liked home more than market did.")

    source = st.radio("Edge source", ["ELO vs Real", "Model vs Real"], horizontal=True)
    min_edge = st.slider("Minimum edge threshold (pp)", 1, 30, 8) / 100

    if source == "ELO vs Real":
        edge_col = "Home_Edge_ELO_vs_Real"
        prob_col = "ELO_Home_Prob"
    else:
        edge_col = "Home_Edge_Model_vs_Real"
        prob_col = "Model_Home_Prob"

    dv = data.dropna(subset=[edge_col, "Home_Win"]).copy()
    dv["Abs_Edge"] = dv[edge_col].abs()
    value_bets = dv[dv["Abs_Edge"] >= min_edge].sort_values("Abs_Edge", ascending=False)

    col1, col2, col3 = st.columns(3)
    home_value = value_bets[value_bets[edge_col] > 0]
    away_value = value_bets[value_bets[edge_col] < 0]
    col1.metric("Total value spots", len(value_bets))
    col2.metric("Home value bets", len(home_value),
                f"{home_value['Home_Win'].mean()*100:.1f}% won" if len(home_value)>0 else "")
    col3.metric("Away value bets", len(away_value),
                f"{(1-away_value['Home_Win'].dropna()).mean()*100:.1f}% won" if len(away_value)>0 else "")

    show_cols = ["Date","Home_Team","Away_Team","Home_Score","Away_Score",
                 "Real_Home_Prob", prob_col, edge_col, "Home_Win"]
    show_cols = [c for c in show_cols if c in value_bets.columns]

    display = value_bets[show_cols].head(40).copy()
    display["Real_Home_Prob"] = (display["Real_Home_Prob"]*100).round(1).astype(str)+"%"
    display[prob_col]         = (display[prob_col]*100).round(1).astype(str)+"%"
    display[edge_col]         = (display[edge_col]*100).round(1).astype(str)+"pp"
    st.dataframe(display.reset_index(drop=True), use_container_width=True)

    # Edge vs outcome scatter
    fig = px.scatter(
        value_bets, x=edge_col, y="Home_Win",
        trendline="lowess", trendline_color_override="red",
        color=edge_col, color_continuous_scale="RdYlGn",
        hover_data=["Date","Home_Team","Away_Team","Real_Home_Prob",prob_col],
        labels={edge_col: "Edge (model prob - real prob)", "Home_Win": "Home won (1=yes)"},
        opacity=0.6,
    )
    fig.update_layout(height=380, xaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5: ROI curves — cumulative P&L using each source
# ══════════════════════════════════════════════════════════════════════════════
with tab_roi:
    st.header("Simulated ROI Curves")
    st.caption("Flat 1-unit bet on home whenever each source's probability exceeds "
               "its own implied threshold + your selected edge. Uses real bookmaker closing odds.")

    threshold = st.slider("Home win probability threshold to bet", 50, 80, 60) / 100
    min_edge_roi = st.slider("Min edge over real implied odds (pp)", 0, 20, 5) / 100

    d_roi = data.dropna(subset=["Real_Home_Prob","Real_Home_Odds","Home_Win"]).copy()
    d_roi = d_roi.sort_values("Date").reset_index(drop=True)

    def sim_roi(prob_col: str, label: str) -> pd.Series:
        d2 = d_roi.dropna(subset=[prob_col]).copy()
        real_implied = 1 / d2["Real_Home_Odds"]
        edge = d2[prob_col] - real_implied
        bet = (d2[prob_col] >= threshold) & (edge >= min_edge_roi)
        pnl = np.where(bet,
            np.where(d2["Home_Win"] == 1, d2["Real_Home_Odds"] - 1, -1.0), 0.0)
        return pd.Series({"dates": d2.loc[bet.values,"Date"].values,
                          "pnl": pnl[bet.values],
                          "label": label,
                          "n": int(bet.sum()),
                          "total": float(pnl.sum())}), d2.loc[bet.values,"Date"], pnl[bet.values]

    fig = go.Figure()
    sources_roi = [("ELO_Home_Prob", "ELO", "#457b9d")]
    if show_model:
        sources_roi.append(("Model_Home_Prob", "XGBoost", "#2a9d8f"))

    metrics_rows = []
    for prob_col, label, color in sources_roi:
        d2 = d_roi.dropna(subset=[prob_col]).copy().sort_values("Date").reset_index(drop=True)
        real_implied = 1 / d2["Real_Home_Odds"]
        edge = d2[prob_col] - real_implied
        bet  = (d2[prob_col] >= threshold) & (edge >= min_edge_roi)
        pnl  = np.where(bet, np.where(d2["Home_Win"]==1, d2["Real_Home_Odds"]-1, -1.0), 0.0)
        cumulative = np.cumsum(pnl)
        dates = d2["Date"].values
        staked = float(bet.sum())
        total_pnl = float(pnl.sum())
        roi = total_pnl / staked * 100 if staked > 0 else 0

        fig.add_trace(go.Scatter(x=dates, y=cumulative,
                                 mode="lines", name=f"{label} ({roi:+.1f}% ROI)",
                                 line={"color": color, "width": 2}))
        metrics_rows.append({"Source": label, "Bets": int(staked),
                              "P&L (units)": round(total_pnl, 2),
                              "ROI%": round(roi, 2)})

    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative P&L (units)",
                      height=450)
    st.plotly_chart(fig, use_container_width=True)

    if metrics_rows:
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)
    st.caption("Note: Uses actual bookmaker closing odds. Includes bookmaker margin (~5-8%).")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6: Game explorer — pick any game and see all three sources side-by-side
# ══════════════════════════════════════════════════════════════════════════════
with tab_games:
    st.header("Game Explorer")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Home team", all_teams)
    with col2:
        team_b = st.selectbox("Away team", [t for t in all_teams if t != team_a])

    matchups = df[((df["Home_Team"]==team_a) & (df["Away_Team"]==team_b)) |
                  ((df["Home_Team"]==team_b) & (df["Away_Team"]==team_a))].sort_values("Date")

    if matchups.empty:
        st.info("No historical Super Rugby meetings found between these teams.")
    else:
        st.write(f"**{len(matchups)} meetings** between {team_a} and {team_b}")

        for _, row in matchups.tail(15).iterrows():
            home = row["Home_Team"]; away = row["Away_Team"]
            score = f"{int(row['Home_Score']) if pd.notna(row['Home_Score']) else '?'} - {int(row['Away_Score']) if pd.notna(row['Away_Score']) else '?'}"
            result_icon = "✓" if row["Home_Win"] == 1 else ("Draw" if pd.isna(row["Home_Win"]) else "✗")

            with st.expander(f"{row['Date'].date()}  {home} {score} {away}  [{result_icon}]"):
                c1, c2, c3 = st.columns(3)

                def fmt_prob(p): return f"{p*100:.1f}%" if pd.notna(p) else "N/A"
                def fmt_odds(o): return f"{o:.2f}" if pd.notna(o) else "N/A"
                def fmt_spread(s): return f"{s:+.1f}" if pd.notna(s) else "N/A"

                c1.markdown("**Real odds (Pinnacle/bet365)**")
                c1.write(f"Home odds: {fmt_odds(row.get('Real_Home_Odds'))}")
                c1.write(f"Away odds: {fmt_odds(row.get('Real_Away_Odds'))}")
                c1.write(f"Home prob (no-vig): {fmt_prob(row.get('Real_Home_Prob'))}")
                c1.write(f"Spread line: {fmt_spread(row.get('Real_Spread'))}")

                c2.markdown("**ELO model**")
                c2.write(f"Home odds: {fmt_odds(row.get('ELO_Home_Odds'))}")
                c2.write(f"Away odds: {fmt_odds(row.get('ELO_Away_Odds'))}")
                c2.write(f"Home prob: {fmt_prob(row.get('ELO_Home_Prob'))}")
                c2.write(f"ELO spread: {fmt_spread(row.get('ELO_Spread'))}")

                c3.markdown("**XGBoost model**")
                c3.write(f"Home odds: {fmt_odds(row.get('Model_Home_Odds'))}")
                c3.write(f"Away odds: {fmt_odds(row.get('Model_Away_Odds'))}")
                c3.write(f"Home prob: {fmt_prob(row.get('Model_Home_Prob'))}")
                c3.write(f"Model spread: {fmt_spread(row.get('Model_Spread'))}")

                c1.metric("Actual margin", fmt_spread(row.get("Actual_Margin")))
                if pd.notna(row.get("Home_Edge_ELO_vs_Real")):
                    c2.metric("ELO edge", f"{row['Home_Edge_ELO_vs_Real']*100:+.1f}pp")
                if pd.notna(row.get("Home_Edge_Model_vs_Real")):
                    c3.metric("Model edge", f"{row['Home_Edge_Model_vs_Real']*100:+.1f}pp")
