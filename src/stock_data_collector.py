"""Yahoo Finance stock data collector with the repository's OHLCV schema."""

from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf


class StockDataCollector:
    """Collect historical stock data for use with the project preprocessor.

    The returned columns match ``BinanceDataCollector`` where possible:
    ``Date``, ``Symbol``, ``Open``, ``High``, ``Low``, ``Close``, and ``Volume``.
    Yahoo Finance does not provide Binance-specific fields such as
    ``Quote_volume`` or ``Taker_buy_volume``.
    """

    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_stock_data(
        self,
        symbols: List[str],
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for one or more Yahoo tickers."""
        if start_date is None:
            days_back = 365 if days_back is None else days_back
            start = datetime.now() - timedelta(days=days_back)
        else:
            start = pd.Timestamp(start_date).to_pydatetime()

        end = (
            pd.Timestamp(end_date).to_pydatetime()
            if end_date is not None
            else datetime.now()
        )

        frames = []
        for symbol in symbols:
            frame = self._fetch_one(symbol, interval, start, end)
            if not frame.empty:
                frames.append(frame)

        if not frames:
            raise ValueError("No stock data was returned for the requested symbols.")

        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["Symbol", "Date"])
            .reset_index(drop=True)
        )

    def _fetch_one(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch and normalize one Yahoo Finance ticker."""
        try:
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by="column",
            )
        except Exception as exc:
            logging.error("Failed to fetch %s: %s", symbol, exc)
            return pd.DataFrame()

        if raw.empty:
            logging.warning("No data returned for %s", symbol)
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [column for column in required if column not in raw.columns]
        if missing:
            raise ValueError(f"Missing columns for {symbol}: {missing}")

        frame = raw[required].reset_index()
        date_column = "Datetime" if "Datetime" in frame.columns else "Date"
        frame = frame.rename(columns={date_column: "Date"})
        frame["Date"] = pd.to_datetime(frame["Date"], utc=True).dt.tz_localize(None)
        frame["Symbol"] = symbol

        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        frame[numeric_columns] = frame[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        return frame[["Date", "Symbol", *numeric_columns]].dropna()

    def save_data(self, data: pd.DataFrame, filename: str) -> Path:
        """Save collected stock data as CSV and return its path."""
        path = self.data_dir / f"{filename}.csv"
        data.to_csv(path, index=False)
        return path