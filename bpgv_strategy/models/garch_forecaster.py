"""GARCH(1,1) model for volatility forecasting."""
import pandas as pd
import numpy as np
from arch import arch_model
from typing import Dict, Optional, Tuple, Any
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class GARCHVolatilityForecaster:
    """GARCH(1,1) model for forecasting building permit growth volatility."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GARCH forecaster with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.garch_params = config['garch']
        self.model = None
        self.fitted_model = None
        self.last_fit_date = None
        self.fit_history = []

    def fit(self, returns: pd.Series, rescale: bool = True) -> 'GARCHVolatilityForecaster':
        """
        Fit GARCH(1,1) model to building permit growth data.

        Args:
            returns: Series of returns/growth rates (already in percentage)
            rescale: Whether to apply additional scaling for numerical stability

        Returns:
            Self for method chaining
        """
        try:
            # Clean data
            clean_returns = returns.dropna()

            if len(clean_returns) < self.garch_params['min_window']:
                raise ValueError(f"Insufficient data: {len(clean_returns)} < {self.garch_params['min_window']}")

            # Standardize returns for better GARCH fitting
            self.returns_mean = clean_returns.mean()
            self.returns_std = clean_returns.std()

            # Avoid division by zero
            if self.returns_std == 0:
                self.returns_std = 1.0

            # Standardize the returns
            standardized_returns = (clean_returns - self.returns_mean) / self.returns_std

            # Additional scaling can help with numerical stability if needed
            scale_factor = self.garch_params.get('scaling_factor', 100) if rescale else 1.0
            self.scale_factor = scale_factor

            # Create GARCH(1,1) model with standardized and scaled returns
            self.model = arch_model(
                standardized_returns * scale_factor,
                mean='Zero',  # No mean model for volatility
                vol='Garch',
                p=self.garch_params['p'],
                q=self.garch_params['q'],
                rescale=False  # We're already handling the scaling
            )

            # Fit the model with better optimization bounds
            try:
                self.fitted_model = self.model.fit(
                    disp='off',
                    show_warning=False,
                    options={'maxiter': 1000}
                )

                # Check for convergence
                if self.fitted_model.convergence_flag != 0:
                    logger.warning(f"GARCH model convergence warning: {self.fitted_model.convergence_flag}")

                # Store fitting information
                self.last_fit_date = clean_returns.index[-1]
                self.fit_history.append({
                    'date': self.last_fit_date,
                    'n_obs': len(clean_returns),
                    'log_likelihood': self.fitted_model.loglikelihood,
                    'aic': self.fitted_model.aic,
                    'bic': self.fitted_model.bic
                })

                logger.info(f"GARCH model fitted successfully on {len(clean_returns)} observations")
                logger.info(f"Model parameters: omega={self.fitted_model.params['omega']:.6f}, "
                          f"alpha={self.fitted_model.params['alpha[1]']:.6f}, "
                          f"beta={self.fitted_model.params['beta[1]']:.6f}")

            except Exception as e:
                logger.error(f"Failed to fit GARCH model: {e}")
                # Try with different settings using standardized returns
                self.model = arch_model(
                    standardized_returns * scale_factor,
                    mean='Constant',  # Try with constant mean
                    vol='Garch',
                    p=1,
                    q=1,
                    rescale=False
                )
                self.fitted_model = self.model.fit(disp='off', show_warning=False)
                logger.info("GARCH model fitted with alternative settings")

            return self

        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            raise

    def forecast_volatility(self, horizon: int = 1) -> float:
        """
        Forecast next period's conditional volatility.

        Args:
            horizon: Forecast horizon (periods ahead)

        Returns:
            Forecasted volatility (standard deviation, not variance)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")

        try:
            # Generate forecast
            forecast = self.fitted_model.forecast(horizon=horizon)

            # Extract variance forecast
            variance_forecast = forecast.variance.values[-1, horizon - 1]

            # Convert variance to volatility (standard deviation)
            volatility_forecast = np.sqrt(variance_forecast)

            # Rescale back to original units
            if hasattr(self, 'returns_std') and hasattr(self, 'scale_factor'):
                volatility_forecast = volatility_forecast * self.returns_std / self.scale_factor

            # Handle edge cases
            if np.isnan(volatility_forecast) or np.isinf(volatility_forecast):
                logger.warning("Invalid volatility forecast, using historical average")
                volatility_forecast = np.sqrt(self.fitted_model.conditional_volatility.mean())
                if hasattr(self, 'returns_std') and hasattr(self, 'scale_factor'):
                    volatility_forecast = volatility_forecast * self.returns_std / self.scale_factor

            return float(volatility_forecast)

        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            raise

    def rolling_forecast(self, returns: pd.Series,
                        window: int = 60,
                        refit_freq: int = 12,
                        min_train_size: int = None) -> pd.DataFrame:
        """
        Generate out-of-sample volatility forecasts using walk-forward analysis.

        Args:
            returns: Full series of returns
            window: Maximum lookback window (default 60 months, max 120)
            refit_freq: How often to refit model (default every 12 months)
            min_train_size: Minimum training size (default = window)

        Returns:
            DataFrame with volatility forecasts and related metrics
        """
        # Set parameters
        window = min(window, self.garch_params.get('max_history', 120))
        min_train_size = min_train_size or self.garch_params['min_window']

        if len(returns) < min_train_size:
            raise ValueError(f"Insufficient data for rolling forecast: {len(returns)} < {min_train_size}")

        results = []
        last_refit_idx = -refit_freq  # Force refit on first iteration

        # Walk-forward through the data
        for i in range(min_train_size, len(returns)):
            current_date = returns.index[i]

            # Determine training window
            train_start_idx = max(0, i - window)
            train_data = returns.iloc[train_start_idx:i]

            # Check if we need to refit
            if i - last_refit_idx >= refit_freq or self.fitted_model is None:
                try:
                    logger.info(f"Refitting GARCH at {current_date} with {len(train_data)} observations")
                    self.fit(train_data)
                    last_refit_idx = i
                except Exception as e:
                    logger.warning(f"Failed to refit at {current_date}: {e}")
                    if self.fitted_model is None:
                        continue  # Skip if no model available

            # Generate one-step-ahead forecast
            try:
                # Update model with latest data point for forecasting
                volatility_forecast = self.forecast_volatility(horizon=1)

                # Calculate realized volatility for comparison (if we have future data)
                if i + 12 < len(returns):  # Look ahead 12 months for realized vol
                    future_returns = returns.iloc[i:i+12]
                    realized_vol = future_returns.std()
                else:
                    realized_vol = np.nan

                results.append({
                    'date': current_date,
                    'volatility_forecast': volatility_forecast,
                    'realized_volatility': realized_vol,
                    'train_window_size': len(train_data),
                    'model_refitted': i == last_refit_idx
                })

            except Exception as e:
                logger.warning(f"Failed to forecast at {current_date}: {e}")
                results.append({
                    'date': current_date,
                    'volatility_forecast': np.nan,
                    'realized_volatility': np.nan,
                    'train_window_size': len(train_data),
                    'model_refitted': False
                })

        # Convert to DataFrame
        forecast_df = pd.DataFrame(results)
        if not forecast_df.empty:
            forecast_df.set_index('date', inplace=True)

            # Calculate forecast error metrics
            valid_forecasts = forecast_df.dropna(subset=['volatility_forecast', 'realized_volatility'])
            if len(valid_forecasts) > 0:
                mae = np.abs(valid_forecasts['volatility_forecast'] - valid_forecasts['realized_volatility']).mean()
                rmse = np.sqrt(((valid_forecasts['volatility_forecast'] - valid_forecasts['realized_volatility']) ** 2).mean())
                logger.info(f"Forecast performance - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        logger.info(f"Generated {len(forecast_df)} volatility forecasts")
        return forecast_df

    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about the fitted model.

        Returns:
            Dictionary with model diagnostics
        """
        if self.fitted_model is None:
            return {'error': 'Model not fitted'}

        try:
            diagnostics = {
                'convergence_flag': self.fitted_model.convergence_flag,
                'log_likelihood': float(self.fitted_model.loglikelihood),
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'parameters': {
                    'omega': float(self.fitted_model.params.get('omega', np.nan)),
                    'alpha': float(self.fitted_model.params.get('alpha[1]', np.nan)),
                    'beta': float(self.fitted_model.params.get('beta[1]', np.nan))
                },
                'persistence': float(
                    self.fitted_model.params.get('alpha[1]', 0) +
                    self.fitted_model.params.get('beta[1]', 0)
                ),
                'unconditional_volatility': float(np.sqrt(self.fitted_model.params.get('omega', 0) /
                    (1 - self.fitted_model.params.get('alpha[1]', 0) - self.fitted_model.params.get('beta[1]', 0)))
                    if (self.fitted_model.params.get('alpha[1]', 0) + self.fitted_model.params.get('beta[1]', 0)) < 1
                    else np.nan
                ),
                'last_fit_date': self.last_fit_date.strftime('%Y-%m-%d') if self.last_fit_date else None,
                'n_observations': len(self.fitted_model.model._y) if hasattr(self.fitted_model, 'model') else None
            }

            return diagnostics

        except Exception as e:
            logger.error(f"Error getting model diagnostics: {e}")
            return {'error': str(e)}

    def calculate_volatility_percentiles(self, volatility_forecasts: pd.Series,
                                        lookback: Optional[int] = None) -> pd.Series:
        """
        Calculate expanding or rolling percentile ranks of volatility forecasts.

        Args:
            volatility_forecasts: Series of volatility forecasts
            lookback: Optional lookback window for rolling percentiles (None for expanding)

        Returns:
            Series of percentile ranks (0-100)
        """
        if lookback is None:
            # Expanding window percentiles
            percentiles = volatility_forecasts.expanding(min_periods=1).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
            )
        else:
            # Rolling window percentiles
            percentiles = volatility_forecasts.rolling(window=lookback, min_periods=1).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
            )

        return percentiles