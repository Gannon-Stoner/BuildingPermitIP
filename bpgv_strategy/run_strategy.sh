#!/bin/bash
# Run the BPGV Trading Strategy

echo "========================================"
echo "BPGV Trading Strategy Execution"
echo "========================================"
echo ""
echo "Using local housing permit data from: ~/Housing_Permit_IP"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the strategy
echo "Starting strategy backtest..."
python main.py --start-date 2010-01-01 --end-date 2024-12-31 --log-level INFO

echo ""
echo "========================================"
echo "Strategy execution complete!"
echo "Check the outputs/ directory for results"
echo "========================================"