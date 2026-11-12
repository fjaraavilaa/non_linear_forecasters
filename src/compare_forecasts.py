"""Compare two native-unit prediction parquet files on common timestamps."""

import argparse
import json

import pandas as pd

from src.forecast_evaluation import compare_forecasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    args = parser.parse_args()
    result = compare_forecasts(pd.read_parquet(args.baseline), pd.read_parquet(args.candidate))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
