import numpy as np
import timesfm


def _normalize_version(version):
    return str(version).lower().removeprefix('timesfm-')


def set_timefm_model(version, context_length, horizon):
    version = _normalize_version(version)

    if version in {'2.5', '2p5'}:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            'google/timesfm-2.5-200m-pytorch',
            torch_compile=False,
        )

        model.compile(
            timesfm.ForecastConfig(
                max_context=context_length,
                max_horizon=horizon,
                normalize_inputs=False,
                infer_is_positive=False,
                return_backcast=True,
            )
        )
        return model

    if version in {'3', '3.0'}:
        return timesfm.TimesFM3Forecaster.from_pretrained(
            'google/timesfm-3.0-pytorch'
        )

    raise ValueError(f'Unsupported TimesFM version: {version!r}')


def forecast(model, inputs, horizon):
    if hasattr(model, 'predict'):
        return model.predict(inputs, horizon)

    if hasattr(model, 'forecast'):
        return model.forecast(horizon, inputs)

    raise TypeError('Model does not provide a supported forecasting method')


def forecast_with_covariates(model, inputs, dynamic_numerical_covariates):
    if hasattr(model, 'forecast_with_covariates'):
        return model.forecast_with_covariates(
            inputs=inputs,
            dynamic_numerical_covariates={
                name: [np.asarray(values, dtype=np.float32)]
                for name, values in dynamic_numerical_covariates.items()
            },
            xreg_mode='xreg + timesfm',
            normalize_xreg_target_per_input=True,
            force_on_cpu=True,
        )

    if hasattr(model, 'predict'):
        covariates = np.asarray(
            [dynamic_numerical_covariates[name]
             for name in dynamic_numerical_covariates],
            dtype=np.float32,
        )
        return model.predict(
            np.asarray(inputs[0], dtype=np.float32),
            covariates.shape[-1] - len(inputs[0]),
            past_future_covariates=covariates,
        )

    raise TypeError('Model does not support numerical covariates')