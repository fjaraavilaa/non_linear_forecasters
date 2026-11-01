#!/usr/bin/env python3
"""A research-first crypto screen using the repository's Binance collector.

This is an educational ranking tool, not investment advice or an execution bot.
It does not assess custody, smart-contract, regulatory, token-unlock, or
fundamental risks; investigate every candidate separately before investing.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import pstdev

from src.binance_data_collector import BinanceDataCollector


def clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def historical_cvar(returns: list[float], confidence: float) -> float:
    """Return historical expected loss beyond the chosen VaR confidence level."""
    losses = sorted(-return_value for return_value in returns)
    tail_count = max(1, math.ceil(len(losses) * (1 - confidence)))
    return sum(losses[:tail_count]) / tail_count * 100


def historical_var(returns: list[float], confidence: float) -> float:
    """Return the historical loss threshold at the chosen confidence level."""
    losses = sorted(-return_value for return_value in returns)
    var_index = min(len(losses) - 1, math.ceil(len(losses) * confidence) - 1)
    return losses[var_index] * 100


def history_metrics(
    collector: BinanceDataCollector,
    symbol: str,
    days: int,
    cvar_confidence: float,
) -> dict[str, float]:
    """Calculate Binance daily-return metrics for one symbol."""
    data = collector.fetch_crypto_data(symbols=[symbol], interval="1d", days_back=days)
    if data.empty or not {"Date", "Close", "Quote_volume"}.issubset(data.columns):
        raise RuntimeError(f"No usable price history for {symbol}")
    data = data.sort_values("Date")
    prices = [float(price) for price in data["Close"].tolist() if price > 0]
    quote_volumes = [float(volume) for volume in data["Quote_volume"].tolist() if volume >= 0]
    if len(prices) < 10 or not quote_volumes:
        raise RuntimeError(f"Insufficient price history for {symbol}")

    returns = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
    annualized_volatility = pstdev(returns) * math.sqrt(365) * 100
    peak = prices[0]
    max_drawdown = 0.0
    for price in prices:
        peak = max(peak, price)
        max_drawdown = min(max_drawdown, price / peak - 1)
    return {
        "change_30d_pct": (prices[-1] / prices[max(0, len(prices) - 31)] - 1) * 100,
        "volatility_pct": annualized_volatility,
        "max_drawdown_pct": max_drawdown * 100,
        "var_pct": historical_var(returns, cvar_confidence),
        "cvar_pct": historical_cvar(returns, cvar_confidence),
        "average_daily_quote_volume": sum(quote_volumes) / len(quote_volumes),
        "latest_quote_volume": quote_volumes[-1],
    }


def score(metrics: dict[str, float], min_quote_volume: float) -> tuple[float, list[str]]:
    """Score 0–100 using Binance liquidity, momentum, and price-risk metrics."""
    liquidity = clamp(
        math.log10(max(metrics["average_daily_quote_volume"], 1))
        / math.log10(max(min_quote_volume, 1))
        * 100
    )
    momentum = clamp(50 + metrics["change_30d_pct"] * 1.25)
    volatility = clamp(100 - metrics["volatility_pct"] / 2)
    drawdown = clamp(100 + metrics["max_drawdown_pct"] * 1.5)
    result = 0.35 * liquidity + 0.20 * momentum + 0.25 * volatility + 0.20 * drawdown

    flags = []
    if metrics["average_daily_quote_volume"] < min_quote_volume:
        flags.append("low average quote volume")
    if metrics["volatility_pct"] > 150:
        flags.append("very high volatility")
    if metrics["max_drawdown_pct"] < -80:
        flags.append("more than 80% drawdown")
    return result, flags


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank liquid Binance USDT pairs for research; not investment advice.")
    parser.add_argument("--limit", type=int, default=20, help="number of most liquid USDT pairs to inspect (default: 20)")
    parser.add_argument("--days", type=int, default=90, help="price-history window in days (default: 90)")
    parser.add_argument("--cvar-confidence", type=float, default=0.95, help="historical CVaR confidence level (default: 0.95)")
    parser.add_argument("--min-quote-volume", type=float, default=1_000_000, help="minimum average daily USDT volume (default: 1m)")
    parser.add_argument("--data-dir", default="data", help="collector data directory (default: data)")
    parser.add_argument("--out", default="crypto_screen.csv", help="CSV output filename")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100 or args.days < 14:
        parser.error("--limit must be 1–100 and --days must be at least 14")
    if not 0.5 < args.cvar_confidence < 1:
        parser.error("--cvar-confidence must be greater than 0.5 and less than 1")
    if args.min_quote_volume <= 0:
        parser.error("--min-quote-volume must be positive")

    collector = BinanceDataCollector(data_dir=args.data_dir)
    tickers = collector.get_24hr_ticker()
    candidates = sorted(
        (
            ticker for ticker in tickers
            if ticker.get("symbol", "").endswith("USDT")
            and not any(token in ticker["symbol"] for token in ("UP", "DOWN", "BULL", "BEAR"))
            and float(ticker.get("quoteVolume") or 0) > 0
        ),
        key=lambda ticker: float(ticker.get("quoteVolume") or 0),
        reverse=True,
    )[:args.limit]
    rows = []
    for ticker in candidates:
        symbol = ticker["symbol"]
        try:
            metrics = history_metrics(collector, symbol, args.days, args.cvar_confidence)
            research_score, flags = score(metrics, args.min_quote_volume)
            rows.append({
                "rank": "", "symbol": symbol, "quote_asset": "USDT",
                "research_score_0_100": round(research_score, 1),
                "average_daily_quote_volume_usdt": round(metrics["average_daily_quote_volume"], 2),
                "latest_quote_volume_usdt": round(metrics["latest_quote_volume"], 2),
                "change_30d_pct": round(metrics["change_30d_pct"], 2),
                "annualized_volatility_pct": round(metrics["volatility_pct"], 2), "max_drawdown_window_pct": round(metrics["max_drawdown_pct"], 2),
                "historical_var_pct": round(metrics["var_pct"], 2),
                "historical_cvar_pct": round(metrics["cvar_pct"], 2),
                "flags": "; ".join(flags) or "none",
            })
        except RuntimeError as error:
            print(f"Skipping {symbol}: {error}")

    rows.sort(key=lambda row: row["research_score_0_100"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    fields = list(rows[0]) if rows else ["rank", "name", "symbol"]
    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} candidates to {out_path.resolve()}")
    print("Review the flags and perform tokenomics, security, custody, and legal due diligence before any decision.")


if __name__ == "__main__":
    main()
