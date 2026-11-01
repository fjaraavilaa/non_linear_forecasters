"""
Binance API Data Collection Module for Enhanced Cryptocurrency Volatility Analysis

This module replaces yfinance with the official Binance API for more accurate,
real-time cryptocurrency data collection with higher frequency and better quality.
"""

import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import os
import time
import json
from urllib.parse import urlencode
import hashlib
import hmac

logging.basicConfig(level=logging.INFO)

class BinanceDataCollector:
    """
    Enhanced cryptocurrency data collector using Binance API.
    
    Provides superior data quality, real-time updates, and higher frequency data
    compared to yfinance for cryptocurrency volatility analysis.
    """
    
    def __init__(self, data_dir: str = "../data", api_key: str = None, api_secret: str = None):
        """
        Initialize the Binance data collector.
        
        Args:
            data_dir (str): Directory to save collected data
            api_key (str): Binance API key (optional, for higher rate limits)
            api_secret (str): Binance API secret (optional, for private endpoints)
        """
        self.data_dir = data_dir
        self.base_url = "https://api.binance.com"
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Symbol mappings (Binance uses different format)
        self.symbol_mapping = {
            'BTC-USD': 'BTCUSDT',
            'ETH-USD': 'ETHUSDT', 
            'ADA-USD': 'ADAUSDT',
            'DOT-USD': 'DOTUSDT',
            'LINK-USD': 'LINKUSDT',
            'LTC-USD': 'LTCUSDT',
            'XRP-USD': 'XRPUSDT',
            'SOL-USD': 'SOLUSDT',
            'AVAX-USD': 'AVAXUSDT',
            'MATIC-USD': 'MATICUSDT',
            'UNI-USD': 'UNIUSDT',
            'ATOM-USD': 'ATOMUSDT',
            'ALGO-USD': 'ALGOUSDT',
            'VET-USD': 'VETUSDT',
            'ICP-USD': 'ICPUSDT'
        }
        
        # Available intervals
        self.intervals = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: dict = None, signed: bool = False) -> dict:
        """
        Make a request to the Binance API with proper error handling.
        
        Args:
            endpoint (str): API endpoint
            params (dict): Request parameters
            signed (bool): Whether request needs to be signed
            
        Returns:
            dict: API response
        """
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        headers = {}
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
        
        if signed and self.api_secret:
            if params is None:
                params = {}
            params['timestamp'] = int(time.time() * 1000)
            query_string = urlencode(params)
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            raise
    
    def get_exchange_info(self) -> dict:
        """
        Get exchange information including available trading pairs.
        
        Returns:
            dict: Exchange information
        """
        return self._make_request("/api/v3/exchangeInfo")
    
    def get_available_symbols(self, quote_asset: str = 'USDT') -> List[str]:
        """
        Get all available trading symbols for a quote asset.
        
        Args:
            quote_asset (str): Quote asset (e.g., 'USDT', 'BTC', 'ETH')
            
        Returns:
            List[str]: Available symbols
        """
        exchange_info = self.get_exchange_info()
        symbols = []
        
        for symbol_info in exchange_info['symbols']:
            if (symbol_info['quoteAsset'] == quote_asset and 
                symbol_info['status'] == 'TRADING'):
                symbols.append(symbol_info['symbol'])
        
        return sorted(symbols)

    def get_24hr_ticker(self, symbols: Optional[List[str]] = None) -> List[dict]:
        """Get Binance 24-hour ticker statistics for one or more symbols."""
        if symbols:
            tickers = []
            for symbol in symbols:
                ticker = self._make_request(
                    "/api/v3/ticker/24hr",
                    {'symbol': self.symbol_mapping.get(symbol, symbol)},
                )
                tickers.append(ticker)
            return tickers

        response = self._make_request("/api/v3/ticker/24hr")
        return response if isinstance(response, list) else [response]
    
    def fetch_klines(
        self, 
        symbol: str, 
        interval: str = '1d',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch candlestick/kline data from Binance API.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            interval (str): Time interval ('1m', '5m', '1h', '1d', etc.)
            start_time (datetime): Start time for data
            end_time (datetime): End time for data
            limit (int): Number of data points (max 1000)
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        # Convert yfinance-style symbols if needed
        if symbol in self.symbol_mapping:
            binance_symbol = self.symbol_mapping[symbol]
        else:
            binance_symbol = symbol
        
        #logging.info(f"Fetching {interval} data for {binance_symbol}...")
        
        params = {
            'symbol': binance_symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        try:
            data = self._make_request("/api/v3/klines", params)
            
            if not data:
                logging.info(f"No data returned for {binance_symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_time', 'Quote_volume', 'Count', 'Taker_buy_volume',
                'Taker_buy_quote_volume', 'Ignore'
            ])
            
            # Convert data types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                             'Quote_volume', 'Count', 'Taker_buy_volume', 
                             'Taker_buy_quote_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamps
            df['Date'] = pd.to_datetime(df['Open_time'], unit='ms')
            df['Close_time'] = pd.to_datetime(df['Close_time'], unit='ms')
            
            # Add symbol for identification
            df['Symbol'] = symbol  # Keep original symbol format for consistency
            
            # Reorder columns to match expected format
            df = df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 
                    'Quote_volume', 'Count', 'Taker_buy_volume', 'Taker_buy_quote_volume']]
            
            #logging.info(f"Fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_crypto_data(
        self, 
        symbols: List[str], 
        interval: str = '1d',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: Optional[int] = None,
        limit_per_request: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch cryptocurrency data for multiple symbols using Binance API.
        
        Args:
            symbols (List[str]): List of symbols to fetch
            interval (str): Time interval for data
            start_date (str): Start date, e.g. '2023-01-01'. Takes precedence over days_back.
            end_date (str): End date, e.g. '2023-12-31'. Defaults to now if start_date is given.
            days_back (int): Number of days to fetch, used only if start_date is not given.
            limit_per_request (int): Limit per API request
            
        Returns:
            pd.DataFrame: Combined cryptocurrency data
        """
        # Resolve the time range: explicit dates take precedence over days_back
        if start_date is not None:
            start_time = pd.to_datetime(start_date).to_pydatetime()
            end_time = pd.to_datetime(end_date).to_pydatetime() if end_date is not None else datetime.now()
            #logging.info(f"Fetching Binance data for {len(symbols)} symbols...")
            #logging.info(f"Interval: {interval}, Date range: {start_time.date()} to {end_time.date()}")
        else:
            if days_back is None:
                days_back = 365
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            logging.info(f"Fetching Binance data for {len(symbols)} symbols...")
            logging.info(f"Interval: {interval}, Days back: {days_back}")
        
        all_data = []
        
        for symbol in symbols:
            try:
                # For longer periods, we might need multiple requests
                symbol_data = []
                current_start = start_time
                
                while current_start < end_time:
                    # Calculate end time for this request (max 1000 periods)
                    if interval == '1d':
                        current_end = min(current_start + timedelta(days=limit_per_request), end_time)
                    elif interval == '1h':
                        current_end = min(current_start + timedelta(hours=limit_per_request), end_time)
                    elif interval == '1m':
                        current_end = min(current_start + timedelta(minutes=limit_per_request), end_time)
                    elif interval == '5m':
                        current_end = min(current_start + timedelta(minutes=5*limit_per_request), end_time)
                    elif interval == '15m':
                        current_end = min(current_start + timedelta(minutes=15*limit_per_request), end_time)
                    elif interval == '30m':
                        current_end = min(current_start + timedelta(minutes=30*limit_per_request), end_time)
                    else:
                        current_end = end_time  # For other intervals, fetch in one go
                    
                    batch_data = self.fetch_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=current_start,
                        end_time=current_end,
                        limit=limit_per_request
                    )
                    
                    if not batch_data.empty:
                        symbol_data.append(batch_data)
                    
                    # Move to next batch
                    current_start = current_end
                    
                    if current_end >= end_time or len(symbol_data) == 0:
                        break
                
                # Combine batches for this symbol
                if symbol_data:
                    combined_symbol_data = pd.concat(symbol_data, ignore_index=True)
                    # Remove duplicates that might occur at batch boundaries
                    combined_symbol_data = combined_symbol_data.drop_duplicates(subset=['Date', 'Symbol'])
                    combined_symbol_data = combined_symbol_data.sort_values('Date')
                    all_data.append(combined_symbol_data)
                else:
                    logging.warning(f"No data collected for {symbol}")
                    
            except Exception as e:
                logging.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values(['Symbol', 'Date'])
            logging.info(f"Successfully collected data for {combined_data['Symbol'].nunique()} symbols")
            logging.info(f"Total records: {len(combined_data)}")
            logging.info(f"Date range: {combined_data['Date'].min()} to {combined_data['Date'].max()}")
            return combined_data
        else:
            raise ValueError("No data was successfully collected from Binance API") 
    
    def calculate_enhanced_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced volatility features using Binance-specific data.
        
        Args:
            df (pd.DataFrame): Input OHLCV data
            
        Returns:
            pd.DataFrame: Data with enhanced volatility features
        """
        print("Calculating enhanced volatility features with Binance data...")
        
        data = df.copy()
        
        # Basic returns and volatility (same as before)
        data['Returns'] = data.groupby('Symbol')['Close'].pct_change()
        
        # Enhanced volatility measures using OHLC data
        data['True_Range'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data.groupby('Symbol')['Close'].shift(1)),
                abs(data['Low'] - data.groupby('Symbol')['Close'].shift(1))
            )
        )
        
        # Parkinson volatility estimator (uses High-Low)
        data['Parkinson_Vol'] = (
            data.groupby('Symbol')['True_Range']
            .rolling(window=20, min_periods=1)
            .apply(lambda x: np.sqrt(np.mean(np.log(x)**2) * 252 / (4 * np.log(2))))
            .reset_index(level=0, drop=True)
        )
        
        # Garman-Klass volatility estimator
        ln_hl = np.log(data['High'] / data['Low'])
        ln_co = np.log(data['Close'] / data['Open'])
        
        data['GK_Vol'] = (
            data.groupby('Symbol').apply(
                lambda x: (0.5 * ln_hl.loc[x.index]**2 - 
                          (2*np.log(2) - 1) * ln_co.loc[x.index]**2)
                         .rolling(window=20, min_periods=1)
                         .mean()
                         .apply(lambda y: np.sqrt(y * 252))
            )
            .reset_index(level=0, drop=True)
        )
        
        # Rogers-Satchell volatility estimator
        ln_ho = np.log(data['High'] / data['Open'])
        ln_hc = np.log(data['High'] / data['Close'])
        ln_lo = np.log(data['Low'] / data['Open'])
        ln_lc = np.log(data['Low'] / data['Close'])
        
        rs_vol = ln_ho * ln_hc + ln_lo * ln_lc
        data['RS_Vol'] = (
            data.groupby('Symbol').apply(
                lambda x: rs_vol.loc[x.index]
                         .rolling(window=20, min_periods=1)
                         .mean()
                         .apply(lambda y: np.sqrt(y * 252))
            )
            .reset_index(level=0, drop=True)
        )
        
        # Volume-weighted features (using Binance volume data)
        data['VWAP'] = (
            (data['High'] + data['Low'] + data['Close']) / 3 * data['Volume']
        ).groupby(data['Symbol']).cumsum() / data.groupby('Symbol')['Volume'].cumsum()
        
        data['Volume_MA_10'] = (
            data.groupby('Symbol')['Volume']
            .rolling(window=10, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_10']
        
        # Binance-specific features
        if 'Quote_volume' in data.columns:
            data['Avg_Trade_Size'] = data['Quote_volume'] / data['Count']
            data['Taker_Buy_Ratio'] = data['Taker_buy_volume'] / data['Volume']
            data['Taker_Buy_Quote_Ratio'] = data['Taker_buy_quote_volume'] / data['Quote_volume']
        
        # Standard volatility windows
        windows = [5, 10, 20, 30, 60]
        for window in windows:
            data[f'Volatility_{window}d'] = (
                data.groupby('Symbol')['Returns']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
                * np.sqrt(252)  # Annualized
            )
            
            # Rolling mean return
            data[f'Mean_Return_{window}d'] = (
                data.groupby('Symbol')['Returns']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        
        # Clean up and fill NaN values
        data = data.fillna(method='bfill').fillna(method='ffill')
        
        logging.info("Enhanced volatility features calculated successfully")
        return data
    
    def save_data(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save DataFrame to CSV file.
        
        Args:
            df (pd.DataFrame): Data to save
            filename (str): Name of the file (without extension)
            
        Returns:
            str: Full path of saved file
        """
        filepath = os.path.join(self.data_dir, f"{filename}.csv")
        df.to_csv(filepath, index=False)
        logging.info(f"Data saved to: {filepath}")
        return filepath
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filename (str): Name of the file (without extension)
            
        Returns:
            pd.DataFrame: Loaded data
        """
        filepath = os.path.join(self.data_dir, f"{filename}.csv")
        if os.path.exists(filepath):
            data = pd.read_csv(filepath, parse_dates=['Date'])
            logging.info(f"Data loaded from: {filepath}")
            return data
        else:
            raise FileNotFoundError(f"File not found: {filepath}")


def main():
    """
    Example usage of the Binance API data collector.
    """
    logging.info("Binance API Cryptocurrency Data Collector")
    logging.info("=" * 50)
    
    # Popular cryptocurrency symbols (using yfinance format for compatibility)
    crypto_symbols = [
        'BTC-USD',   # Bitcoin
        'ETH-USD',   # Ethereum
        'ADA-USD',   # Cardano
        'DOT-USD',   # Polkadot
        'LINK-USD',  # Chainlink
        'SOL-USD',   # Solana
        'AVAX-USD',  # Avalanche
        'MATIC-USD', # Polygon
    ]
    
    # Initialize Binance data collector
    collector = BinanceDataCollector(data_dir="data")
    
    # Example 1: Get available symbols
    logging.info("\n1. Getting available USDT pairs...")
    available_symbols = collector.get_available_symbols('USDT')
    logging.info(f"Found {len(available_symbols)} USDT trading pairs")
    logging.info(f"Sample symbols: {available_symbols[:10]}")
    
    # Example 2: Fetch daily data
    #logging.info(f"\n2. Fetching daily data for {len(crypto_symbols)} cryptocurrencies...")
    daily_data = collector.fetch_crypto_data(
        symbols=crypto_symbols,
        interval='1d',
        days_back=730  # 2 years
    )
    
    # Example 3: Calculate enhanced volatility features
    logging.info(f"\n3. Calculating enhanced volatility features...")
    processed_data = collector.calculate_enhanced_volatility_features(daily_data)
    
    # Example 4: Save data
    logging.info(f"\n4. Saving data...")
    collector.save_data(daily_data, "binance_crypto_raw_data")
    collector.save_data(processed_data, "binance_crypto_processed_data")
    
    # Example 5: Get real-time ticker data
    logging.info(f"\n5. Getting 24hr ticker statistics...")
    ticker_data = collector.get_24hr_ticker(crypto_symbols[:5])  # First 5 symbols
    
    logging.info(f"\n Data Collection Summary:")
    logging.info(f"Raw data shape: {daily_data.shape}")
    logging.info(f"Processed data shape: {processed_data.shape}")
    logging.info(f"Date range: {processed_data['Date'].min()} to {processed_data['Date'].max()}")
    logging.info(f"Symbols: {processed_data['Symbol'].unique()}")
    logging.info(f"New volatility features: Parkinson, Garman-Klass, Rogers-Satchell")
    logging.info(f"Binance data collection completed successfully!")

if __name__ == "__main__":
    main()
