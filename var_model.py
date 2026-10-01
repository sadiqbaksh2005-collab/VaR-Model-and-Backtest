
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, chi2


# Data

def fetch_price_history(tickers: list, lookback_days: int = 500) -> pd.DataFrame:
    """Adjusted close prices, dates x tickers, for the trailing lookback_days calendar days."""
    data = yf.download(tickers, period=f"{lookback_days}d", auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        # single ticker case: yfinance returns a flat frame
        prices = data[["Close"]]
        prices.columns = tickers
    return prices.dropna(how="all").ffill().dropna()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def get_latest_prices(prices: pd.DataFrame) -> dict:
    return prices.iloc[-1].to_dict()

# Portfolio construction


def build_portfolio(shares: dict, prices: pd.DataFrame, returns: pd.DataFrame):
    """
    shares: {ticker: number_of_shares} (negative = short)
    Returns: portfolio_returns (Series), portfolio_value (float), weights (dict)
    """
    tickers = list(shares.keys())
    latest = prices.iloc[-1]
    values = {t: shares[t] * latest[t] for t in tickers}
    total_value = sum(values.values())
    if total_value == 0:
        raise ValueError("Portfolio has zero net value — can't compute weights.")
    weights = {t: values[t] / total_value for t in tickers}

    w_vec = np.array([weights[t] for t in tickers])
    port_returns = returns[tickers].values @ w_vec
    port_returns = pd.Series(port_returns, index=returns.index, name="portfolio_return")

    return port_returns, total_value, weights, values


# VaR / CVaR methods


def historical_var_cvar(port_returns: pd.Series, confidence: float = 0.95,
                         portfolio_value: float = 1.0, horizon_days: int = 1) -> dict:
    scaled = port_returns * np.sqrt(horizon_days)  # simple sqrt-time scaling
    alpha = 1 - confidence
    var_return = -np.percentile(scaled, alpha * 100)
    tail = scaled[scaled <= -var_return]
    cvar_return = -tail.mean() if len(tail) > 0 else var_return
    return dict(
        method="Historical",
        var_pct=var_return, cvar_pct=cvar_return,
        var_dollar=var_return * portfolio_value,
        cvar_dollar=cvar_return * portfolio_value,
    )


def parametric_var_cvar(port_returns: pd.Series, confidence: float = 0.95,
                         portfolio_value: float = 1.0, horizon_days: int = 1) -> dict:
    mu, sigma = port_returns.mean(), port_returns.std()
    mu_h = mu * horizon_days
    sigma_h = sigma * np.sqrt(horizon_days)
    z = norm.ppf(confidence)
    alpha = 1 - confidence

    var_return = z * sigma_h - mu_h
    cvar_return = (norm.pdf(z) / alpha) * sigma_h - mu_h

    return dict(
        method="Parametric (Normal)",
        var_pct=var_return, cvar_pct=cvar_return,
        var_dollar=var_return * portfolio_value,
        cvar_dollar=cvar_return * portfolio_value,
    )


def monte_carlo_var_cvar(returns: pd.DataFrame, weights: dict, confidence: float = 0.95,
                          portfolio_value: float = 1.0, horizon_days: int = 1,
                          n_sims: int = 50_000, seed: int = 42) -> dict:
    tickers = list(weights.keys())
    mu_vec = returns[tickers].mean().values * horizon_days
    cov = returns[tickers].cov().values * horizon_days
    w_vec = np.array([weights[t] for t in tickers])

    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(mu_vec, cov, size=n_sims)
    port_sims = sims @ w_vec

    alpha = 1 - confidence
    var_return = -np.percentile(port_sims, alpha * 100)
    tail = port_sims[port_sims <= -var_return]
    cvar_return = -tail.mean() if len(tail) > 0 else var_return

    return dict(
        method="Monte Carlo",
        var_pct=var_return, cvar_pct=cvar_return,
        var_dollar=var_return * portfolio_value,
        cvar_dollar=cvar_return * portfolio_value,
        sim_returns=port_sims,
    )



# Backtesting


def rolling_var_backtest(port_returns: pd.Series, confidence: float = 0.95,
                          window: int = 250, method: str = "historical") -> pd.DataFrame:
    """
    Walk forward: for each day t, estimate 1-day VaR from the preceding
    `window` days only (no lookahead), then compare to the actual return
    realized on day t. A breach is actual_loss > VaR_estimate.
    """
    rows = []
    values = port_returns.values
    idx = port_returns.index
    alpha = 1 - confidence

    for t in range(window, len(values)):
        train = values[t - window:t]
        actual = values[t]

        if method == "historical":
            var_est = -np.percentile(train, alpha * 100)
        else:  # parametric
            mu, sigma = train.mean(), train.std()
            z = norm.ppf(confidence)
            var_est = z * sigma - mu

        breach = actual < -var_est
        rows.append(dict(date=idx[t], actual_return=actual, var_estimate=var_est, breach=breach))

    return pd.DataFrame(rows)


def kupiec_pof_test(n_obs: int, n_breaches: int, confidence: float = 0.95) -> dict:
    """
    Kupiec (1995) proportion-of-failures likelihood ratio test.
    H0: the observed breach rate matches the expected rate (1 - confidence).
    Returns the LR statistic, p-value, and expected vs observed breach counts.
    """
    p = 1 - confidence
    phat = n_breaches / n_obs if n_obs > 0 else 0.0

    log_num = (n_obs - n_breaches) * np.log(1 - p) + n_breaches * np.log(p) if p not in (0, 1) else 0.0
    # phat terms: guard 0**0 -> handled natively since 0**0 == 1 in Python/NumPy, but log(0) is -inf,
    # so compute via log only on the nonzero pieces.
    log_den = 0.0
    if n_obs - n_breaches > 0:
        log_den += (n_obs - n_breaches) * np.log(1 - phat)
    if n_breaches > 0:
        log_den += n_breaches * np.log(phat)

    lr_stat = -2 * (log_num - log_den)
    p_value = 1 - chi2.cdf(lr_stat, df=1)

    return dict(
        n_obs=n_obs, n_breaches=n_breaches,
        expected_breaches=p * n_obs,
        observed_rate=phat, expected_rate=p,
        lr_stat=lr_stat, p_value=p_value,
        reject_null=p_value < 0.05,  # True => model's breach rate looks wrong
    )