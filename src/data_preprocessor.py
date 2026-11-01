"""
Simple DataFrame-based Data Preprocessor for LSTM Cryptocurrency Volatility Analysis

This module takes a DataFrame as input and processes it for LSTM training.
No file I/O - pure DataFrame in, processed data out.

author: @fjaraavila
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Dict
import warnings
import logging
import logging
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO)


class DataPreprocessor:
    """
    Simple DataFrame-based preprocessor for cryptocurrency LSTM models.
    
    Usage:
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.prepare_features(raw_df)
        lstm_data = preprocessor.prepare_lstm_data(processed_data, symbols=['BTC-USD', 'ETH-USD'])
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.feature_columns = []
        self.scalers = {}
    
    def calculate_volatility_from_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility measures from OHLC data.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC columns
            
        Returns:
            pd.DataFrame: DataFrame with volatility features added
        """
        print("Calculating volatility from OHLC data...")
        
        data = df.copy()
        
        # Check required columns
        required = ['Open', 'High', 'Low', 'Close']
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing OHLC columns: {missing}")
        
        # Calculate simple and continuously compounded returns
        data['Returns'] = data.groupby('Symbol')['Close'].diff()
        data['Log_Returns'] = data.groupby('Symbol')['Close'].transform(
            lambda prices: np.log(prices).diff()
        )
        
        # Traditional volatility (rolling std of log returns)
        windows = [5, 10, 20, 30]
        for window in windows:
            data[f'Volatility_{window}d'] = (
                data.groupby('Symbol')['Log_Returns']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            ) * np.sqrt(252)  # Annualized
        
        # Parkinson volatility (High-Low based)
        data['Parkinson_Vol'] = (
            np.sqrt((1/(4*np.log(2))) * (np.log(data['High'] / data['Low']))**2) * np.sqrt(252)
        )
        
        # Garman-Klass volatility
        data['GK_Vol'] = np.sqrt(
            0.5 * (np.log(data['High'] / data['Low']))**2 -
            (2*np.log(2) - 1) * (np.log(data['Close'] / data['Open']))**2
        ) * np.sqrt(252)
        
        # Average True Range
        data['High_Low'] = data['High'] - data['Low']
        previous_close = data.groupby('Symbol')['Close'].shift(1)
        data['High_Close'] = abs(data['High'] - previous_close)
        data['Low_Close'] = abs(data['Low'] - previous_close)
        data['True_Range'] = data[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        
        data['ATR_14'] = (
            data.groupby('Symbol')['True_Range']
            .rolling(window=14, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        
        # Clean up intermediate columns
        data.drop(['High_Low', 'High_Close', 'Low_Close'], axis=1, inplace=True, errors='ignore')
        
        logging.info(f"Added volatility features")
        return data
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators added
        """
        logging.info("Calculating technical indicators...")
        
        data = df.copy()
        
        # RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-8)  # Avoid division by zero
            return 100 - (100 / (1 + rs))
        
        data['RSI'] = data.groupby('Symbol')['Close'].apply(calculate_rsi).reset_index(level=0, drop=True)
        
        # Moving averages
        ma_periods = [5, 10, 20, 50]
        for period in ma_periods:
            data[f'MA_{period}'] = (
                data.groupby('Symbol')['Close']
                .rolling(window=period, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Price ratio to MA
            data[f'Price_MA_{period}_Ratio'] = data['Close'] / data[f'MA_{period}']
        
        # Bollinger Bands (using 20-period MA)
        if 'MA_20' in data.columns:
            bb_std = (
                data.groupby('Symbol')['Close']
                .rolling(window=20, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
            data['BB_Upper'] = data['MA_20'] + (2 * bb_std)
            data['BB_Lower'] = data['MA_20'] - (2 * bb_std)
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        logging.info("Added technical indicators")
        return data
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = None, drop_bad_values:bool = True) -> pd.DataFrame:
        """
        Complete feature preparation from raw DataFrame.
        
        Args:
            df (pd.DataFrame): Raw OHLC DataFrame
            target_col (str): Target column name (auto-detected if None)
            drop_bad_values (bool): Whether to drop rows with NaN values

        Returns:
            pd.DataFrame: DataFrame with all features prepared
        """
        logging.info("Starting feature preparation...")
        
        data = df.copy()
        
        # Ensure we have required columns
        if 'Symbol' not in data.columns:
            raise ValueError("DataFrame must have 'Symbol' column")
        
        if 'Date' not in data.columns:
            # Try common date column names
            date_cols = [col for col in data.columns if col.lower() in ['timestamp', 'datetime', 'time']]
            if date_cols:
                data['Date'] = pd.to_datetime(data[date_cols[0]])
            else:
                raise ValueError("No date column found")
        else:
            data['Date'] = pd.to_datetime(data['Date'])
        
        # Sort by symbol and date
        data = data.sort_values(['Symbol', 'Date'])
        
        # Calculate volatility if we have OHLC data
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            data = self.calculate_volatility_from_ohlc(data)
            data = self.calculate_technical_indicators(data)
        else:
            warnings.warn("No OHLC data found, skipping volatility calculations")
            if 'Close' in data.columns:
                data['Returns'] = data.groupby('Symbol')['Close'].diff()
                data['Log_Returns'] = data.groupby('Symbol')['Close'].transform(
                    lambda prices: np.log(prices).diff()
                )
                data['Volatility_20d'] = (
                    data.groupby('Symbol')['Log_Returns']
                    .rolling(window=20, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                ) * np.sqrt(252)
        
        # Auto-detect target column
        if target_col is None:
            vol_cols = [col for col in data.columns if 'volatility' in col.lower()]
            if 'Volatility_20d' in data.columns:
                target_col = 'Volatility_20d'
            elif vol_cols:
                target_col = vol_cols[0]
            else:
                raise ValueError("No volatility target column found")
        
        logging.info(f"Using target: {target_col}")
        
        # Define feature columns
        exclude_cols = ['Date', 'Symbol', target_col]
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        base_features = [col for col in numeric_cols if col not in exclude_cols]
        
        # Create lagged features
        logging.info("Creating lagged features...")
        lag_periods = [1, 2, 3, 5, 10]
        key_cols = ['Close', 'Returns', 'Log_Returns', target_col]
        
        # Add volume if available
        volume_cols = [col for col in data.columns if 'volume' in col.lower()]
        if volume_cols:
            key_cols.append(volume_cols[0])
        
        for lag in lag_periods:
            for col in key_cols:
                if col in data.columns:
                    lag_col = f"{col}_lag_{lag}"
                    data[lag_col] = data.groupby('Symbol')[col].shift(lag)
                    base_features.append(lag_col)
        
        # Create momentum features
        logging.info("Creating momentum features...")
        if 'Close' in data.columns:
            momentum_periods = [3, 5, 10, 14]
            for period in momentum_periods:
                mom_col = f"Momentum_{period}d"
                data[mom_col] = (data['Close'] / data.groupby('Symbol')['Close'].shift(period)) - 1
                base_features.append(mom_col)
        
        # OHLC ratios
        if all(col in data.columns for col in ['High', 'Low', 'Close']):
            logging.info("Creating OHLC ratios...")
            data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close']
            base_features.append('HL_Ratio')
            
            if 'Open' in data.columns:
                data['OC_Ratio'] = abs(data['Open'] - data['Close']) / data['Close']
                base_features.append('OC_Ratio')
        
        # Clean data
        logging.info("Cleaning data...")
        data = data.replace([np.inf, -np.inf], np.nan)
        initial_rows = len(data)
        if drop_bad_values:
            data = data.dropna()
            cleaned_rows = initial_rows - len(data)
        else:
            cleaned_rows = 0
        
        if cleaned_rows > 0:
            logging.info(f"Removed {cleaned_rows} rows with NaN/inf values")
        elif not drop_bad_values:
            logging.info(" No rows removed, but NaN/inf values remain in the dataset")
        else:
            logging.info("No NaN/inf values found")
        
        # Final feature list
        self.feature_columns = [col for col in base_features if col in data.columns]
        
        logging.info(f"Feature preparation complete. Total features: {len(self.feature_columns)}")
        logging.info(f"Dataset shape: {data.shape}")
        logging.info(f"Features: {len(self.feature_columns)}")
        logging.info(f"Target: {target_col}")
        
        return data

    def create_sequences(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        sequence_length: int = 30,
        target_col: str = 'Volatility_20d',
        prediction_horizon: int = 1,
        input_study_features = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create LSTM sequences for a single symbol.
        
        Args:
            data (pd.DataFrame): Processed data
            symbol (str): Symbol to process
            sequence_length (int): Length of input sequences
            target_col (str): Target column
            prediction_horizon (int): Steps ahead to predict
            input_study_features (List[str], optional): List of feature to include in matrixes
 
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: X sequences, y targets, target dates
        """
        symbol_data = data[data['Symbol'] == symbol].copy().sort_values('Date')
        
        if len(symbol_data) < sequence_length + prediction_horizon:
            return np.array([]), np.array([]), np.array([])
        

        logging.info(f'Using the following {target_col}')
        # Get features, target, and dates
        if input_study_features is None:
            X_data = symbol_data[self.feature_columns].values
        else:
            X_data = symbol_data[input_study_features].values
        y_data = symbol_data[target_col].values
        date_data = symbol_data['Date'].values
        
        sequences = []
        targets = []
        target_dates = []
        
        for i in range(len(X_data) - sequence_length - prediction_horizon + 1):
            seq = X_data[i:i + sequence_length]
            target_idx = i + sequence_length + prediction_horizon - 1
            target = y_data[target_idx]
            
            if not (np.isnan(seq).any() or np.isnan(target)):
                sequences.append(seq)
                targets.append(target)
                # The date the target value corresponds to - used for
                # calendar-based (rather than ratio-based) train/val/test splits
                target_dates.append(date_data[target_idx])
        
        return np.array(sequences), np.array(targets), np.array(target_dates)
        
    
    def scale_features(self, df: pd.DataFrame, scaler: str):

        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

        if scaler == 'minmax':
            self.scaler_used = MinMaxScaler().set_output(transform='pandas')
            returned_df = self.scaler_used.fit_transform(df[numeric_features])
        elif scaler == 'standardization':
            self.scaler_used = StandardScaler().set_output(transform='pandas')
            returned_df = self.scaler_used.fit_transform(df[numeric_features])
        elif scaler == 'robust':
            self.scaler_used = RobustScaler().set_output(transform='pandas')
            returned_df = self.scaler_used.fit_transform(df[numeric_features])
        else:
            raise ValueError(f"Unknown scaler type: {scaler}. Choose from 'minmax', 'standardization', or 'robust'.")
        return returned_df

    def prepare_lstm_data(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        sequence_length: int = 30,
        target_col: str = 'Volatility_20d',
        prediction_horizon: int = 1,
        date_splits: Optional[Dict[str, Dict[str, str]]] = None,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        warn_if_unscaled: bool = True,
        input_study_features: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Transform processed tabular data into LSTM matrix structure.
 
        Splitting can be done in two ways:
          1. Calendar-based (recommended): pass `date_splits`, a dict shaped like
             {"train": {"start": "2025-07-06", "end": "2026-01-05"},
              "validation": {"start": "2026-01-06", "end": "2026-04-06"},
              "test": {"start": "2026-04-07", "end": "2026-07-06"}}
             i.e. exactly the structure produced by your healthy_dates.json.
             Each sequence is assigned to a split based on the date of the
             row it is predicting (the *target* date), so the boundaries you
             specify are respected exactly no matter how many rows get
             dropped upstream for NaNs/lag warm-up.
          2. Ratio-based (legacy/fallback): if `date_splits` is None, falls
             back to the old contiguous test_size/validation_size split.
 
        Feature scaling is NOT performed by this function. If your features
        look unscaled (raw prices, volume, etc. next to ratios/percentages),
        a warning is logged telling you to scale beforehand - use this
        class's `scale_features` method on your DataFrame before calling
        `prepare_lstm_data`, or scale X_train/X_val/X_test yourself after.
        Scaling is intentionally left external so nothing here can leak
        val/test statistics into a scaler without you knowing about it.
 
        Returns a dictionary with:
            - X_train, y_train, dates_train
            - X_val, y_val, dates_val
            - X_test, y_test, dates_test
            - metadata (feature_columns, target_col, sequence_length, prediction_horizon, symbols)
        """
        if date_splits is None:
            if not 0 <= test_size < 1:
                raise ValueError("test_size must be in [0, 1)")
            if not 0 <= validation_size < 1:
                raise ValueError("validation_size must be in [0, 1)")
            if test_size + validation_size >= 1:
                raise ValueError("test_size + validation_size must be < 1")
 
        working_data = data.copy()
        working_data['Date'] = pd.to_datetime(working_data['Date'])
 
        # Ensure feature list exists
        if input_study_features is None:
            if not self.feature_columns:
                numeric_cols = working_data.select_dtypes(include=[np.number]).columns.tolist()
                self.feature_columns = [
                    col for col in numeric_cols
                    if col not in ['Date', 'Symbol', target_col]
                ]
            selected_features = self.feature_columns
        else:
            selected_features = input_study_features
 
        missing_features = [col for col in selected_features if col not in working_data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        if target_col not in working_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
 
        # Build RAW (unscaled) sequences per symbol, keeping the target date
        # of each sequence so we can split on calendar boundaries later.
        X_parts, y_parts, date_parts = [], [], []
        used_symbols = []
 
        for symbol in symbols:
            X_symbol, y_symbol, dates_symbol = self.create_sequences(
                working_data,
                symbol=symbol,
                sequence_length=sequence_length,
                target_col=target_col,
                prediction_horizon=prediction_horizon,
                input_study_features=selected_features
            )
 
            if len(X_symbol) > 0:
                X_parts.append(X_symbol)
                y_parts.append(y_symbol)
                date_parts.append(dates_symbol)
                used_symbols.append(symbol)
 
        if not X_parts:
            raise ValueError(
                "No valid LSTM sequences were generated. "
                "Check sequence_length, prediction_horizon, and available rows per symbol."
            )
 
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        dates = np.concatenate(date_parts, axis=0)
 
        # Sort everything chronologically by target date (symbols get
        # interleaved together, which is fine/expected for LSTM training).
        order = np.argsort(dates)
        X, y, dates = X[order], y[order], dates[order]
 
        if date_splits is not None:
            train_start = pd.Timestamp(date_splits['train']['start'])
            train_end = pd.Timestamp(date_splits['train']['end'])
            val_start = pd.Timestamp(date_splits['validation']['start'])
            val_end = pd.Timestamp(date_splits['validation']['end'])
            test_start = pd.Timestamp(date_splits['test']['start'])
            test_end = pd.Timestamp(date_splits['test']['end'])
 
            dates_ts = pd.to_datetime(dates)
            train_mask = (dates_ts >= train_start) & (dates_ts <= train_end)
            val_mask = (dates_ts >= val_start) & (dates_ts <= val_end)
            test_mask = (dates_ts >= test_start) & (dates_ts <= test_end)
 
            X_train, y_train, dates_train = X[train_mask], y[train_mask], dates[train_mask]
            X_val, y_val, dates_val = X[val_mask], y[val_mask], dates[val_mask]
            X_test, y_test, dates_test = X[test_mask], y[test_mask], dates[test_mask]
 
            if len(X_train) == 0:
                raise ValueError(
                    f"No sequences fall inside the train date range "
                    f"{train_start.date()} - {train_end.date()}. "
                    "sequence_length/prediction_horizon may be eating into it - "
                    "check that your raw data actually starts early enough to "
                    "produce a sequence_length warm-up window before train_start."
                )
        else:
            # Legacy ratio-based, contiguous, no-shuffle split
            n_samples = len(X)
            n_test = int(n_samples * test_size)
            n_val = int(n_samples * validation_size)
            n_train = n_samples - n_test - n_val
 
            if n_train <= 0:
                raise ValueError("Not enough samples left for training after split")
 
            X_train, y_train, dates_train = X[:n_train], y[:n_train], dates[:n_train]
            X_val, y_val, dates_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val], dates[n_train:n_train + n_val]
            X_test, y_test, dates_test = X[n_train + n_val:], y[n_train + n_val:], dates[n_train + n_val:]
 
        # This function no longer scales features itself - it only warns if
        # your data looks unscaled, so you can catch it before training.
        # Use self.scale_features(df, scaler='standardization'/'minmax') on
        # your DataFrame beforehand, or scale X_train/X_val/X_test yourself
        # after this returns (fit on train only to avoid leakage).
        if warn_if_unscaled and len(X_train) > 0:
            n_features = len(selected_features)
            X_train_flat = X_train.reshape(-1, n_features)
            feature_means = np.nanmean(X_train_flat, axis=0)
            feature_stds = np.nanstd(X_train_flat, axis=0)
 
            # Rough heuristic: standardized data has mean~0, std~1 per feature.
            # Flag features that are way outside that range.
            unscaled_mask = (np.abs(feature_means) > 3) | (feature_stds > 10) | (feature_stds < 1e-6)
            if unscaled_mask.any():
                offending = [
                    selected_features[i] for i in np.where(unscaled_mask)[0]
                ][:10]  # cap the list so the warning stays readable
                logging.warning(
                    " X_train does not look scaled (mean/std far from 0/1) "
                    f"for {int(unscaled_mask.sum())} of {n_features} features, "
                    f"e.g. {offending}. LSTMs are sensitive to feature scale - "
                    "consider calling self.scale_features(df, scaler=...) on "
                    "your DataFrame before prepare_lstm_data, or scaling "
                    "X_train/X_val/X_test yourself (fit on X_train only)."
                )

        logging.info(f"Target Column is {target_col}")
        logging.info(" LSTM matrix transformation complete")
        logging.info(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
        if len(dates_train) > 0:
            logging.info(f"   train dates:      {pd.to_datetime(dates_train).min().date()} to {pd.to_datetime(dates_train).max().date()}")
        logging.info(f"   X_val:   {X_val.shape}, y_val:   {y_val.shape}")
        if len(dates_val) > 0:
            logging.info(f"   val dates:        {pd.to_datetime(dates_val).min().date()} to {pd.to_datetime(dates_val).max().date()}")
        logging.info(f"   X_test:  {X_test.shape}, y_test:  {y_test.shape}")
        if len(dates_test) > 0:
            logging.info(f"   test dates:       {pd.to_datetime(dates_test).min().date()} to {pd.to_datetime(dates_test).max().date()}")
 
        return {
            'X_train': X_train,
            'y_train': y_train,
            'dates_train': dates_train,
            'X_val': X_val,
            'y_val': y_val,
            'dates_val': dates_val,
            'X_test': X_test,
            'y_test': y_test,
            'dates_test': dates_test,
            'feature_columns': selected_features,
            'target_col': target_col,
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'symbols': used_symbols,
        }





# Simple usage example
def example_usage():
    """Example of how to use the DataFrame-based preprocessor."""
    
    # Create sample data (replace with your actual DataFrame)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    sample_data = pd.DataFrame({
        'Date': np.tile(dates, 2),
        'Symbol': ['BTC-USD'] * 100 + ['ETH-USD'] * 100,
        'Open': np.random.rand(200) * 50000 + 40000,
        'High': np.random.rand(200) * 52000 + 41000,
        'Low': np.random.rand(200) * 48000 + 39000,
        'Close': np.random.rand(200) * 51000 + 40000,
        'Volume': np.random.rand(200) * 1000000
    })
    
    # Use the preprocessor
    preprocessor = DataPreprocessor()
    
    # Step 1: Prepare features (calculates volatility automatically)
    processed_data = preprocessor.prepare_features(sample_data)
    
    # Step 2: Prepare LSTM data
    lstm_data = preprocessor.prepare_lstm_data(
        processed_data,
        symbols=['BTC-USD', 'ETH-USD'],
        sequence_length=20,
        target_col='Volatility_20d'
    )
    
    logging.info("LSTM data ready!")
    logging.info(f"Training shape: {lstm_data['X_train'].shape}")
    logging.info(f"Target: {lstm_data['target_col']}")
    
    return lstm_data


if __name__ == "__main__":
    example_usage()