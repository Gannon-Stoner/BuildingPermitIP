import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# Check backtest results
try:
    results = pd.read_csv('outputs/reports/backtest_results.csv', index_col='date', parse_dates=True)
    
    # Find where portfolio jumps
    results['portfolio_change'] = results['portfolio_value'].diff()
    jump_idx = results['portfolio_change'].abs().idxmax()
    
    print(f'Portfolio jump date: {jump_idx}')
    print(f'Portfolio change: ${results.loc[jump_idx, "portfolio_change"]:,.0f}')
    print(f'Position: ${results.loc[jump_idx, "position"]:,.0f}')
    print(f'Signal: {results.loc[jump_idx, "signal"]}')
    print(f'Close price: ${results.loc[jump_idx, "close_price"]:.2f}')
    print(f'Asset volatility: {results.loc[jump_idx, "asset_volatility"]:.4f}')
    
    # Check position sizing
    if results.loc[jump_idx, 'position'] != 0:
        leverage = abs(results.loc[jump_idx, 'position']) / 1000000  
        print(f'Leverage: {leverage:.1f}x')
        
    # Show surrounding days
    print('\nSurrounding days:')
    idx_pos = results.index.get_loc(jump_idx)
    print(results[['portfolio_value', 'position', 'cash', 'signal']].iloc[max(0, idx_pos-2):min(len(results), idx_pos+3)])
except Exception as e:
    print(f'Error: {e}')
