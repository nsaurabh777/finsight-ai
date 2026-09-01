"""
yfinance-backed tools, ported from the prototype notebook
(crewai_multiagents_stocks_fundamentals_technicals.ipynb). The numeric logic
(beta, drawdown, Sharpe/Sortino, RSI, MACD, SMA labels) is unchanged from the
prototype — it already worked there. Changes made while porting:

1. Tools return formatted strings, not raw DataFrames. Current CrewAI passes
   tool output to the LLM as text; a stringified DataFrame is noisier and
   loses column context. A labelled "key: value" block reads better and keeps
   the numbers unambiguous.
2. get_technical_analysis default period is "1y", not "1mo". It computes a
   200-day SMA; with only ~21 trading days of data SMA_50/SMA_200 are all NaN
   and every trend label is wrong. (Latent prototype bug.)
3. fetch_stock_news_raw falls back to Google News RSS when yfinance's .news
   returns nothing or a shape we don't recognise — the .news payload schema
   has changed repeatedly and has no SLA (PRD Section 3).

By the time these run, `ticker` is already a resolved '.NS' symbol
(src/tools/ticker_resolver.py handled that upstream).
"""
from __future__ import annotations

import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from crewai.tools import tool
except ImportError:  # older crewai packaging
    from crewai_tools import tool


# --------------------------------------------------------------------------
# Numeric helpers (verbatim from the prototype)
# --------------------------------------------------------------------------
def _calculate_beta(stock_returns, market_ticker="^NSEI", period="1y"):
    market = yf.Ticker(market_ticker)
    market_history = market.history(period=period)
    market_returns = market_history["Close"].pct_change().dropna()
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    covariance = aligned.cov().iloc[0, 1]
    market_variance = market_returns.var()
    return covariance / market_variance if market_variance else float("nan")


def _calculate_max_drawdown(prices):
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return drawdown.min()


def _calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess = returns - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / excess.std() if excess.std() else float("nan")


def _calculate_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    excess = returns - risk_free_rate / 252
    downside = excess[excess < target_return]
    downside_dev = np.sqrt(np.mean(downside**2)) if len(downside) else float("nan")
    return np.sqrt(252) * excess.mean() / downside_dev if downside_dev else float("nan")


def _calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


def _trend_label(latest):
    if latest["Close"] > latest["SMA_50"] > latest["SMA_200"]:
        return "Bullish"
    if latest["Close"] < latest["SMA_50"] < latest["SMA_200"]:
        return "Bearish"
    return "Neutral"


def _macd_label(latest):
    return "Bullish" if latest["MACD"] > latest["Signal"] else "Bearish"


def _rsi_label(latest):
    if latest["RSI"] > 70:
        return "Overbought"
    if latest["RSI"] < 30:
        return "Oversold"
    return "Neutral"


def _fmt(value, pct=False, money=False):
    """Compact, LLM-friendly formatting that tolerates missing values."""
    if value is None or value == "N/A" or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, (int, float)):
        if pct:
            return f"{value * 100:.2f}%"
        if money:
            return f"{value:,.2f}"
        return f"{value:,.4f}" if abs(value) < 100 else f"{value:,.2f}"
    return str(value)


def _block(title: str, rows: dict) -> str:
    lines = [title]
    for k, v in rows.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CrewAI tools
# --------------------------------------------------------------------------
@tool("get_basic_stock_info")
def get_basic_stock_info(ticker: str) -> str:
    """Basic profile of one stock: name, sector, industry, market cap, current
    price, 52-week range, average volume.

    Params:
    - ticker: a resolved '.NS' ticker (use resolve_ticker first if you only
      have a company name).
    """
    info = yf.Ticker(ticker).info
    return _block(
        f"Basic info for {ticker}",
        {
            "Name": info.get("longName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Market Cap": _fmt(info.get("marketCap"), money=True),
            "Current Price": _fmt(info.get("currentPrice"), money=True),
            "52-Week High": _fmt(info.get("fiftyTwoWeekHigh"), money=True),
            "52-Week Low": _fmt(info.get("fiftyTwoWeekLow"), money=True),
            "Average Volume": _fmt(info.get("averageVolume"), money=True),
        },
    )


@tool("get_fundamental_analysis")
def get_fundamental_analysis(ticker: str, period: str = "1y") -> str:
    """Fundamental analysis for a stock: valuation multiples, margins, growth,
    leverage, returns, plus the price range over `period`.

    Params:
    - ticker: resolved '.NS' ticker.
    - period: history window for the price stats ("1y","2y","5y","10y","ytd","max").
    """
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    info = stock.info
    price_rows = {}
    if not history.empty:
        price_rows = {
            "Price Avg (period)": _fmt(history["Close"].mean(), money=True),
            "Price Max (period)": _fmt(history["Close"].max(), money=True),
            "Price Min (period)": _fmt(history["Close"].min(), money=True),
        }
    return _block(
        f"Fundamental analysis for {ticker} (period={period})",
        {
            "PE (trailing)": _fmt(info.get("trailingPE")),
            "Forward PE": _fmt(info.get("forwardPE")),
            "PEG": _fmt(info.get("pegRatio") or info.get("trailingPegRatio")),
            "Price/Book": _fmt(info.get("priceToBook")),
            "Dividend Yield": _fmt(info.get("dividendYield"), pct=True),
            "EPS (TTM)": _fmt(info.get("trailingEps")),
            "Revenue Growth (YoY)": _fmt(info.get("revenueGrowth"), pct=True),
            "Earnings Growth (YoY)": _fmt(info.get("earningsGrowth"), pct=True),
            "Profit Margin": _fmt(info.get("profitMargins"), pct=True),
            "Operating Margin": _fmt(info.get("operatingMargins"), pct=True),
            "Free Cash Flow": _fmt(info.get("freeCashflow"), money=True),
            "Debt/Equity": _fmt(info.get("debtToEquity")),
            "Return on Equity": _fmt(info.get("returnOnEquity"), pct=True),
            **price_rows,
        },
    )


@tool("get_stock_risk_assessment")
def get_stock_risk_assessment(ticker: str, period: str = "1y") -> str:
    """Risk profile for a stock: annualized volatility, beta vs. Nifty 50,
    95% Value at Risk, maximum drawdown, Sharpe and Sortino ratios.

    Params:
    - ticker: resolved '.NS' ticker.
    - period: history window for the calculation (default "1y").
    """
    history = yf.Ticker(ticker).history(period=period)
    if history.empty:
        return f"No price history available for {ticker} over period={period}."
    returns = history["Close"].pct_change().dropna()
    return _block(
        f"Risk assessment for {ticker} (period={period})",
        {
            "Annualized Volatility": _fmt(returns.std() * np.sqrt(252), pct=True),
            "Beta (vs Nifty 50)": _fmt(_calculate_beta(returns, period=period)),
            "Value at Risk (95%, daily)": _fmt(np.percentile(returns, 5), pct=True),
            "Maximum Drawdown": _fmt(_calculate_max_drawdown(history["Close"]), pct=True),
            "Sharpe Ratio": _fmt(_calculate_sharpe_ratio(returns)),
            "Sortino Ratio": _fmt(_calculate_sortino_ratio(returns)),
        },
    )


@tool("get_technical_analysis")
def get_technical_analysis(ticker: str, period: str = "1y") -> str:
    """Technical analysis for a stock: 50- and 200-day SMA, 14-day RSI, MACD,
    and derived trend / momentum labels.

    Params:
    - ticker: resolved '.NS' ticker.
    - period: history window. Keep this at least "1y" — the 200-day SMA needs
      ~200 trading days of data or it (and the trend label) is meaningless.
    """
    history = yf.Ticker(ticker).history(period=period)
    if history.empty:
        return f"No price history available for {ticker} over period={period}."
    history["SMA_50"] = history["Close"].rolling(window=50).mean()
    history["SMA_200"] = history["Close"].rolling(window=200).mean()
    history["RSI"] = _calculate_rsi(history["Close"])
    history["MACD"], history["Signal"] = _calculate_macd(history["Close"])
    latest = history.iloc[-1]

    enough_history = not (pd.isna(latest["SMA_50"]) or pd.isna(latest["SMA_200"]))
    rows = {
        "Current Price": _fmt(latest["Close"], money=True),
        "50-day SMA": _fmt(latest["SMA_50"], money=True),
        "200-day SMA": _fmt(latest["SMA_200"], money=True),
        "RSI (14)": _fmt(latest["RSI"]),
        "MACD": _fmt(latest["MACD"]),
        "MACD Signal": _fmt(latest["Signal"]),
        "Trend": _trend_label(latest) if enough_history else "N/A (insufficient history)",
        "MACD Direction": _macd_label(latest),
        "RSI Signal": _rsi_label(latest),
    }
    return _block(f"Technical analysis for {ticker} (period={period})", rows)


# --------------------------------------------------------------------------
# News data feed (NOT a CrewAI tool — used by src/rag/ingest_news.py)
# --------------------------------------------------------------------------
def _news_from_yfinance(ticker: str, limit: int) -> list[dict]:
    items = getattr(yf.Ticker(ticker), "news", None) or []
    articles = []
    for item in items[:limit]:
        # Newer yfinance nests everything under "content"; older is flat.
        content = item.get("content", item)
        provider = content.get("provider") or {}
        url = ""
        if isinstance(content.get("clickThroughUrl"), dict):
            url = content["clickThroughUrl"].get("url", "")
        url = url or content.get("link", "") or content.get("canonicalUrl", {}).get("url", "")
        articles.append(
            {
                "ticker": ticker,
                "title": content.get("title", ""),
                "publisher": (
                    provider.get("displayName")
                    if isinstance(provider, dict)
                    else content.get("publisher", "")
                )
                or "",
                "published": str(
                    content.get("pubDate", "") or content.get("providerPublishTime", "")
                ),
                "url": url,
                "summary": content.get("summary", "") or content.get("description", ""),
            }
        )
    return [a for a in articles if a["title"]]


def _news_from_google_rss(ticker: str, limit: int) -> list[dict]:
    """Fallback: Google News RSS. No API key, no signup (PRD Section 3)."""
    try:
        import feedparser
    except ImportError:
        return []
    # Strip the .NS suffix for a cleaner news query.
    term = ticker.replace(".NS", "") + " stock NSE"
    q = urllib.parse.quote(term)
    feed_url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        with urllib.request.urlopen(feed_url, timeout=15) as resp:
            parsed = feedparser.parse(resp.read())
    except Exception:
        return []
    articles = []
    for entry in parsed.entries[:limit]:
        articles.append(
            {
                "ticker": ticker,
                "title": entry.get("title", ""),
                "publisher": (entry.get("source") or {}).get("title", "Google News"),
                "published": entry.get("published", ""),
                "url": entry.get("link", ""),
                "summary": entry.get("summary", ""),
            }
        )
    return [a for a in articles if a["title"]]


def fetch_stock_news_raw(ticker: str, limit: int = 10) -> list[dict]:
    """Data-feed function for src/rag/ingest_news.py. Agents do NOT call this
    directly — they retrieve news via src/tools/knowledge_retriever.py so
    claims are retrieval-grounded rather than summarised from a raw dump.

    Tries yfinance first, falls back to Google News RSS.
    """
    articles = _news_from_yfinance(ticker, limit)
    if not articles:
        articles = _news_from_google_rss(ticker, limit)
    return articles
