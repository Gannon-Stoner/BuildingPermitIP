"""Performance analytics for BPGV trading strategy."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PerformanceAnalyzer:
    """Analyzes and visualizes backtest performance."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.outputs_dir = Path(config['outputs']['results_dir'])
        self.plots_dir = Path(config['outputs']['plots_dir'])
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Dictionary of performance metrics
        """
        if len(equity_curve) < 2:
            logger.warning("Insufficient data for performance metrics")
            return {}

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        if len(returns) == 0:
            return {}

        # Trading days per year
        trading_days = 252

        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        n_years = len(returns) / trading_days

        # Annual return (CAGR)
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(trading_days)

        # Sharpe Ratio (assume 0% risk-free rate)
        sharpe_ratio = (cagr / annual_vol) if annual_vol > 0 else 0

        # Sortino Ratio (downside volatility)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 0
        sortino_ratio = (cagr / downside_vol) if downside_vol > 0 else 0

        # Maximum Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown_series = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown_series.min()

        # Calmar Ratio (return / max drawdown)
        calmar_ratio = (cagr / abs(max_drawdown)) if max_drawdown != 0 else 0

        # Win Rate and Win/Loss metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0

        # Additional metrics
        metrics = {
            'total_return': total_return * 100,  # Percentage
            'annual_return': cagr * 100,
            'annual_volatility': annual_vol * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate * 100,
            'avg_daily_win': avg_win * 100,
            'avg_daily_loss': avg_loss * 100,
            'profit_factor': abs(positive_returns.sum() / negative_returns.sum()) if negative_returns.sum() != 0 else np.inf if positive_returns.sum() > 0 else 0,
            'total_trading_days': len(returns),
            'best_day': returns.max() * 100,
            'worst_day': returns.min() * 100,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': returns.quantile(0.05) * 100,  # 5% Value at Risk
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean() * 100 if len(returns[returns <= returns.quantile(0.05)]) > 0 else 0  # Conditional VaR
        }

        # Round metrics for display
        metrics = {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in metrics.items()}

        logger.info(f"Performance metrics calculated - Sharpe: {metrics['sharpe_ratio']:.2f}, "
                   f"Max DD: {metrics['max_drawdown']:.1f}%")

        return metrics

    def generate_equity_curve(self, results_df: pd.DataFrame) -> pd.Series:
        """
        Extract equity curve from results DataFrame.

        Args:
            results_df: Backtest results DataFrame

        Returns:
            Series of portfolio values
        """
        if 'portfolio_value' not in results_df.columns:
            raise ValueError("Results DataFrame must contain 'portfolio_value' column")

        return results_df['portfolio_value']

    def calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.Series:
        """
        Calculate monthly returns from equity curve.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Series of monthly returns (only for months with changes)
        """
        # Resample to monthly frequency
        monthly_values = equity_curve.resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()

        # Filter out months with zero or near-zero returns (no active trading)
        # Keep returns that are meaningfully different from zero
        monthly_returns = monthly_returns[abs(monthly_returns) > 1e-6]

        return monthly_returns

    def plot_performance(self, results_df: pd.DataFrame,
                        signals_df: pd.DataFrame,
                        vol_forecasts: pd.DataFrame,
                        save_path: Optional[str] = None) -> None:
        """
        Create comprehensive performance visualization.

        Args:
            results_df: Backtest results
            signals_df: Trading signals
            vol_forecasts: Volatility forecasts
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # Subplot 1: Equity curve with drawdown shading
        ax1 = axes[0]
        equity_curve = results_df['portfolio_value']

        # Plot equity curve (use drawstyle='steps-post' to avoid vertical lines across gaps)
        ax1.plot(equity_curve.index, equity_curve.values, label='Portfolio Value', color='blue', linewidth=1.5, alpha=0.9)

        # Add drawdown shading
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        ax1.fill_between(equity_curve.index,
                         equity_curve.values,
                         running_max.values,
                         where=(drawdown < 0),
                         color='red', alpha=0.3, label='Drawdown')

        ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Subplot 2: Volatility forecast with quartile bands
        ax2 = axes[1]
        if 'volatility_forecast' in vol_forecasts.columns:
            vol_series = vol_forecasts['volatility_forecast'].dropna()

            # Calculate expanding quartiles (these are the actual signal thresholds)
            q25 = vol_series.expanding().quantile(0.25)
            q75 = vol_series.expanding().quantile(0.75)

            # Plot volatility
            ax2.plot(vol_series.index, vol_series.values, label='Volatility Forecast', color='green', linewidth=1.5)

            # Add quartile bands (25-75th percentile range)
            ax2.fill_between(vol_series.index, q25, q75, alpha=0.2, color='green', label='25-75th Percentile Range')

            # Plot dynamic thresholds (these change over time with expanding window)
            ax2.plot(vol_series.index, q25.values, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Long Signal Threshold (25th)')
            ax2.plot(vol_series.index, q75.values, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Short Signal Threshold (75th)')

            ax2.set_title('Building Permit Growth Volatility Forecast', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Volatility (%)', fontsize=12)
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)

        # Subplot 3: Trading signals over time
        ax3 = axes[2]

        # Create signal visualization
        signal_colors = {1: 'green', 0: 'gray', -1: 'red'}
        signal_labels = {1: 'Long', 0: 'Neutral', -1: 'Short'}

        if 'signal' in results_df.columns:
            for signal_val in [-1, 0, 1]:
                mask = results_df['signal'] == signal_val
                if mask.any():
                    ax3.scatter(results_df.index[mask],
                              [signal_val] * mask.sum(),
                              color=signal_colors[signal_val],
                              label=signal_labels[signal_val],
                              alpha=0.6, s=10)

        ax3.set_title('Trading Signals Over Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Signal', fontsize=12)
        ax3.set_yticks([-1, 0, 1])
        ax3.set_yticklabels(['Short', 'Neutral', 'Long'])
        ax3.legend(loc='upper right', ncol=3)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-1.5, 1.5)

        # Subplot 4: Monthly returns distribution
        ax4 = axes[3]

        monthly_returns = self.calculate_monthly_returns(equity_curve)
        if len(monthly_returns) > 0:
            # Create histogram
            n, bins, patches = ax4.hist(monthly_returns * 100, bins=30, alpha=0.7, color='blue', edgecolor='black')

            # Color negative returns red
            for i, patch in enumerate(patches):
                if bins[i] < 0:
                    patch.set_facecolor('red')
                    patch.set_alpha(0.7)

            # Add statistics
            mean_return = monthly_returns.mean() * 100
            ax4.axvline(x=mean_return, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.2f}%')
            ax4.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

            ax4.set_title('Monthly Returns Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Monthly Return (%)', fontsize=12)
            ax4.set_ylabel('Frequency', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')

        # Adjust layout and save
        plt.tight_layout()

        if save_path is None:
            save_path = self.plots_dir / 'performance_report.png'

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Performance plot saved to {save_path}")

        plt.show()

    def create_summary_report(self, metrics: Dict[str, float],
                            trade_stats: Dict[str, Any]) -> str:
        """
        Create formatted summary report.

        Args:
            metrics: Performance metrics
            trade_stats: Trade statistics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("BPGV TRADING STRATEGY - PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")

        # Performance Metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"Total Return:          {metrics.get('total_return', 0):.2f}%")
        report.append(f"Annual Return (CAGR):  {metrics.get('annual_return', 0):.2f}%")
        report.append(f"Annual Volatility:     {metrics.get('annual_volatility', 0):.2f}%")
        report.append(f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Sortino Ratio:         {metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"Maximum Drawdown:      {metrics.get('max_drawdown', 0):.2f}%")
        report.append(f"Calmar Ratio:          {metrics.get('calmar_ratio', 0):.2f}")
        report.append("")

        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-" * 40)
        report.append(f"Value at Risk (95%):   {metrics.get('var_95', 0):.2f}%")
        report.append(f"Conditional VaR (95%): {metrics.get('cvar_95', 0):.2f}%")
        report.append(f"Skewness:              {metrics.get('skewness', 0):.2f}")
        report.append(f"Kurtosis:              {metrics.get('kurtosis', 0):.2f}")
        report.append("")

        # Trading Statistics
        report.append("TRADING STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Trades:          {trade_stats.get('n_trades', 0)}")
        report.append(f"Win Rate:              {metrics.get('win_rate', 0):.2f}%")
        report.append(f"Avg Daily Win:         {metrics.get('avg_daily_win', 0):.2f}%")
        report.append(f"Avg Daily Loss:        {metrics.get('avg_daily_loss', 0):.2f}%")
        report.append(f"Profit Factor:         {metrics.get('profit_factor', 0):.2f}")
        report.append(f"Total Trading Days:    {metrics.get('total_trading_days', 0)}")
        report.append("")

        # Trade Details
        if trade_stats:
            report.append("TRADE DETAILS")
            report.append("-" * 40)
            report.append(f"Long Trades:           {trade_stats.get('n_long_trades', 0)}")
            report.append(f"Short Trades:          {trade_stats.get('n_short_trades', 0)}")
            report.append(f"Average Win:           ${trade_stats.get('avg_win', 0):,.2f}")
            report.append(f"Average Loss:          ${trade_stats.get('avg_loss', 0):,.2f}")
            report.append(f"Largest Win:           ${trade_stats.get('largest_win', 0):,.2f}")
            report.append(f"Largest Loss:          ${trade_stats.get('largest_loss', 0):,.2f}")
            report.append(f"Total Costs:           ${trade_stats.get('total_costs', 0):,.2f}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def save_results(self, results_df: pd.DataFrame,
                    metrics: Dict[str, float],
                    trade_stats: Dict[str, Any]) -> None:
        """
        Save results to files.

        Args:
            results_df: Backtest results DataFrame
            metrics: Performance metrics
            trade_stats: Trade statistics
        """
        # Save results DataFrame
        results_path = self.outputs_dir / 'backtest_results.csv'
        results_df.to_csv(results_path)
        logger.info(f"Results saved to {results_path}")

        # Save metrics
        import json
        metrics_path = self.outputs_dir / 'performance_metrics.json'

        # Combine metrics and trade stats
        all_metrics = {**metrics, **trade_stats}

        # Convert any numpy types to Python types for JSON serialization
        all_metrics = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                      for k, v in all_metrics.items()}

        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")

        # Save text report
        report = self.create_summary_report(metrics, trade_stats)
        report_path = self.outputs_dir / 'report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")