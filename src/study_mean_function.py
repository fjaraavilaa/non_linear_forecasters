import optuna


storage_name = "sqlite:///mydb_lstm_5minprediction.db"


study = optuna.create_study(
    directions=["minimize"],
    study_name="lstm_crypto_study_mean_function_minmax_morepatience",
    storage=storage_name,
    load_if_exists=True,
)

