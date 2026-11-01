from src.binance_data_collector import BinanceDataCollector
from src.data_preprocessor import DataPreprocessor
from src.stock_data_collector import StockDataCollector
import logging

logging.basicConfig(level=logging.INFO)

def test_fetch():
    # Initialize the data collector
    collector = BinanceDataCollector()
    
    # Fetch historical data for a specific symbol and interval
    symbol = 'BTC-USD'
    interval = '1m'
    start_time = '2024-01-01 00:00:00'
    end_time = '2024-01-15 01:00:00'
    
    data = collector.fetch_crypto_data(symbols = [symbol], interval = interval, days_back=15)
    
    # Check if data is fetched correctly
    assert data is not None, "Data should not be None"
    assert len(data) > 0, "Data should contain at least one entry"
    
    logging.info("Data fetching test passed!")
    return data

test_data = test_fetch()

stupid_preprocessor = DataPreprocessor()
df_test = stupid_preprocessor.prepare_features(test_data, target_col = 'Returns', drop_bad_values=False)

df_test[df_test['Returns'].isna()]

sequences = stupid_preprocessor.create_sequences(df_test, symbol = 'BTC-USD', 
                                                 sequence_length = 5, target_col='Returns', 
                                                 prediction_horizon=1, input_study_features=['Returns', 'Volume'])

sequences_lstm = stupid_preprocessor.prepare_lstm_data(df_test, symbols=['BTC-USD'], sequence_length=5, target_col='Returns')


def test_transform_to_lstm_matrix():
    collector = BinanceDataCollector()
    preprocessor = DataPreprocessor()

    symbol = 'BTC-USD'
    interval = '1m'

    raw_data = collector.fetch_crypto_data(symbols=[symbol], interval=interval, days_back=15)
    processed_data = preprocessor.prepare_features(raw_data, target_col='Returns')

    lstm_data = preprocessor.prepare_lstm_data(
        processed_data,
        symbols=[symbol],
        sequence_length=30,
        target_col='Returns',
        prediction_horizon=1,
        test_size=0.2,
        validation_size=0.1,
        input_study_features=['Returns', 'Volume']
    )

    assert lstm_data['X_train'].ndim == 3, "X_train must be 3D: (samples, timesteps, features)"
    assert lstm_data['y_train'].ndim == 1, "y_train must be 1D: (samples,)"

    logging.info("LSTM matrix transform test passed!")
    logging.info(f"X_train shape: {lstm_data['X_train'].shape}")
    logging.info(f"y_train shape: {lstm_data['y_train'].shape}")
    return lstm_data

test_transform_to_lstm_matrix()

def test_fetch_yfinance():
    collector = StockDataCollector()
    
    symbol = 'AAPL'
    interval = '1d'
    start_time = '2024-01-01'
    end_time = '2024-01-15'

    data = collector.fetch_stock_data(symbols=[symbol], interval=interval, 
                                      start_date=start_time, end_date=end_time)
    
    assert data is not None, "Data should not be None"
    assert len(data) > 0, "Data should contain at least one entry"
    
    logging.info("Yahoo Finance data fetching test passed!")
    return data

test_fetch_yfinance()

