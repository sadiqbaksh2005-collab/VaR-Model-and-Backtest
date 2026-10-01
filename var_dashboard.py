"""
Portfolio VaR & Risk Dashboard (Streamlit)
---------------------------------------------
Builds a multi-asset portfolio from live yfinance prices, computes
Historical / Parametric / Monte Carlo VaR and Expected Shortfall,
and backtests a rolling VaR series with the Kupiec test.

Run:
    streamlit run var_dashboard.py

Requires var_model.py in the same folder.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from var_model import (
    fetch_price_history, compute_log_returns, build_portfolio,
    historical_var_cvar, parametric_var_cvar, monte_carlo_var_cvar,
    rolling_var_backtest, kupiec_pof_test,
)

st.set_page_config(page_title="Portfolio VaR Dashboard", layout="wide")

# ----------------------------------------------------------------------
# Sidebar: build the portfolio
# ----------------------------------------------------------------------
st.sidebar.header("Portfolio")

if "positions" not in st.session_state:
    st.session_state["positions"] = [{"ticker": "AAPL", "shares": 100},
                                      {"ticker": "MSFT", "shares": 50}]

with st.sidebar.form("add_position", clear_on_submit=True):
    c1, c2 = st.columns(2)
    new_ticker = c1.text_input("Ticker")
    new_shares = c2.number_input("Shares (+/-)", value=100, step=10)
    if st.form_submit_button("Add position") and new_ticker:
        st.session_state["positions"].append(
            {"ticker": new_ticker.strip().upper(), "shares": new_shares}
        )

if st.session_state["positions"]:
    st.sidebar.write("**Current positions:**")
    for i, pos in enumerate(st.session_state["positions"]):
        pcol1, pcol2 = st.sidebar.columns([3, 1])
        pcol1.write(f"{pos['ticker']}: {pos['shares']} sh")
        if pcol2.button("✕", key=f"rm_{i}"):
            st.session_state["positions"].pop(i)
            st.rerun()

st.sidebar.divider()
lookback_days = st.sidebar.slider("Lookback window (calendar days)", 100, 1500, 500, 50)
confidence = st.sidebar.select_slider("Confidence level", options=[0.90, 0.95, 0.975, 0.99], value=0.95)
horizon_days = st.sidebar.number_input("Horizon (days)", min_value=1, max_value=20, value=1)
backtest_window = st.sidebar.slider("Backtest rolling window (days)", 100, 750, 250, 25)
run = st.sidebar.button("Fetch & Compute", type="primary")

st.title("📉 Portfolio VaR & Risk Dashboard")
st.caption("Historical, parametric, and Monte Carlo Value-at-Risk with Kupiec backtesting")

if not st.session_state["positions"]:
    st.info("Add at least one position in the sidebar to begin.")
    st.stop()

if not run and "var_results" not in st.session_state:
    st.info("Set up your portfolio in the sidebar, then click **Fetch & Compute**.")
    st.stop()

# ----------------------------------------------------------------------
# Fetch + compute (cached in session_state so tab switches don't re-fetch)
# ----------------------------------------------------------------------
if run:
    tickers = [p["ticker"] for p in st.session_state["positions"]]
    shares = {p["ticker"]: p["shares"] for p in st.session_state["positions"]}
    try:
        with st.spinner("Downloading price history..."):
            prices = fetch_price_history(tickers, lookback_days)
            returns = compute_log_returns(prices)
            port_returns, port_value, weights, values = build_portfolio(shares, prices, returns)

        hist = historical_var_cvar(port_returns, confidence, port_value, horizon_days)
        para = parametric_var_cvar(port_returns, confidence, port_value, horizon_days)
        mc = monte_carlo_var_cvar(returns, weights, confidence, port_value, horizon_days)

        st.session_state["var_results"] = dict(
            prices=prices, returns=returns, port_returns=port_returns,
            port_value=port_value, weights=weights, values=values,
            hist=hist, para=para, mc=mc,
        )
    except Exception as e:
        st.error(f"Couldn't build the portfolio: {e}")
        st.stop()

res = st.session_state["var_results"]

# ----------------------------------------------------------------------
# Portfolio summary
# ----------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Portfolio value", f"${res['port_value']:,.2f}")
c2.metric("Positions", len(res["weights"]))
c3.metric("Confidence / horizon", f"{confidence:.1%} / {horizon_days}d")

weights_df = pd.DataFrame({
    "ticker": list(res["weights"].keys()),
    "value": [res["values"][t] for t in res["weights"]],
    "weight": [res["weights"][t] for t in res["weights"]],
})
st.dataframe(
    weights_df.style.format({"value": "${:,.2f}", "weight": "{:.1%}"}),
    use_container_width=True, hide_index=True,
)

tab_summary, tab_dist, tab_backtest = st.tabs(["VaR / CVaR Summary", "P&L Distribution", "Backtest"])

# ----------------------------------------------------------------------
# Summary tab
# ----------------------------------------------------------------------
with tab_summary:
    methods = [res["hist"], res["para"], res["mc"]]
    cols = st.columns(3)
    for col, m in zip(cols, methods):
        col.subheader(m["method"])
        col.metric(f"VaR ({confidence:.0%})", f"${m['var_dollar']:,.2f}",
                   help=f"As a % of portfolio: {m['var_pct']:.2%}")
        col.metric("Expected Shortfall (CVaR)", f"${m['cvar_dollar']:,.2f}",
                   help=f"As a % of portfolio: {m['cvar_pct']:.2%}")

    st.caption(
        f"VaR is the loss not expected to be exceeded {confidence:.0%} of the time over "
        f"{horizon_days} day(s). CVaR (Expected Shortfall) is the average loss "
        f"*given* that VaR is breached — always larger than VaR, and the number regulators "
        f"increasingly prefer since it captures tail severity, not just the threshold."
    )

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(name="VaR", x=[m["method"] for m in methods],
                                  y=[m["var_dollar"] for m in methods]))
    fig_compare.add_trace(go.Bar(name="CVaR", x=[m["method"] for m in methods],
                                  y=[m["cvar_dollar"] for m in methods]))
    fig_compare.update_layout(barmode="group", title="VaR vs CVaR by method",
                               yaxis_title="Dollar loss", height=400)
    st.plotly_chart(fig_compare, use_container_width=True)

# ----------------------------------------------------------------------
# Distribution tab
# ----------------------------------------------------------------------
with tab_dist:
    port_returns = res["port_returns"]
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=port_returns * res["port_value"], nbinsx=60,
                                     name="Historical daily P&L"))
    fig_hist.add_vline(x=-res["hist"]["var_dollar"], line_color="orange", line_dash="dash",
                        annotation_text="Historical VaR")
    fig_hist.add_vline(x=-res["hist"]["cvar_dollar"], line_color="red", line_dash="dash",
                        annotation_text="Historical CVaR")
    fig_hist.update_layout(title="Distribution of historical daily P&L",
                            xaxis_title="Daily P&L ($)", yaxis_title="Frequency", height=450)
    st.plotly_chart(fig_hist, use_container_width=True)

    sim = res["mc"]["sim_returns"] * res["port_value"]
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(x=sim, nbinsx=80, name="Monte Carlo simulated P&L"))
    fig_mc.add_vline(x=-res["mc"]["var_dollar"], line_color="orange", line_dash="dash",
                      annotation_text="MC VaR")
    fig_mc.add_vline(x=-res["mc"]["cvar_dollar"], line_color="red", line_dash="dash",
                      annotation_text="MC CVaR")
    fig_mc.update_layout(title="Monte Carlo simulated P&L distribution",
                          xaxis_title="Simulated daily P&L ($)", yaxis_title="Frequency", height=450)
    st.plotly_chart(fig_mc, use_container_width=True)

# ----------------------------------------------------------------------
# Backtest tab
# ----------------------------------------------------------------------
with tab_backtest:
    bt_method = st.radio("VaR method to backtest", ["historical", "parametric"], horizontal=True)
    if len(res["port_returns"]) <= backtest_window + 10:
        st.warning("Not enough history for this backtest window — shorten the window or increase lookback.")
    else:
        bt = rolling_var_backtest(res["port_returns"], confidence, backtest_window, bt_method)
        n_obs = len(bt)
        n_breaches = int(bt["breach"].sum())
        kupiec = kupiec_pof_test(n_obs, n_breaches, confidence)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Observations", n_obs)
        k2.metric("Breaches (observed)", n_breaches, f"expected ~{kupiec['expected_breaches']:.1f}")
        k3.metric("Kupiec LR stat", f"{kupiec['lr_stat']:.2f}")
        k4.metric("p-value", f"{kupiec['p_value']:.3f}")

        if kupiec["reject_null"]:
            st.error(
                "Kupiec test rejects the null at 5% — the breach rate is statistically "
                "inconsistent with the model's stated confidence level. The VaR model is "
                "likely miscalibrated (too many or too few breaches for what it claims)."
            )
        else:
            st.success(
                "Kupiec test does not reject the null — the observed breach rate is "
                "statistically consistent with the model's stated confidence level."
            )

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt["date"], y=bt["actual_return"] * res["port_value"],
                                     mode="lines", name="Actual daily P&L", line=dict(width=1)))
        fig_bt.add_trace(go.Scatter(x=bt["date"], y=-bt["var_estimate"] * res["port_value"],
                                     mode="lines", name="Rolling VaR threshold (loss)",
                                     line=dict(color="orange", dash="dash")))
        breaches = bt[bt["breach"]]
        fig_bt.add_trace(go.Scatter(x=breaches["date"], y=breaches["actual_return"] * res["port_value"],
                                     mode="markers", name="Breach",
                                     marker=dict(color="red", size=8, symbol="x")))
        fig_bt.update_layout(title="Rolling VaR backtest", xaxis_title="Date",
                              yaxis_title="Daily P&L ($)", height=450)
        st.plotly_chart(fig_bt, use_container_width=True)