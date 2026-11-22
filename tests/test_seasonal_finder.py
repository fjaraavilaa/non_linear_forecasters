from src.help_functions.seasonal_period_searchers import find_seasonal_period
from tests.test_fetch import test_fetch

def test_find_seasonal_period():
    data = test_fetch()
    df = data[data['Symbol'] == 'BTC-USD']
    period = find_seasonal_period(x = df['Close'], fast = False)
    assert period is not None, "Seasonal period should not be None"
    print(f"Found seasonal period: {period}")
    return period

seasonal_period = test_find_seasonal_period()