#!/usr/bin/env python3
"""
BLOB AI Automated Trading System Launcher

This script provides an easy way to start the automated trading system
with different modes and configurations.
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="BLOB AI Automated Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_automated_trading.py --mode single    # Run one trading cycle
  python start_automated_trading.py --mode continuous # Run continuously
  python start_automated_trading.py --demo           # Run in demo mode
  python start_automated_trading.py --help           # Show this help

Safety Features:
  - All trades include stop loss and take profit
  - Position size limited by risk management
  - Maximum open positions enforced
  - Real-time market analysis before each trade
  - Comprehensive logging of all activities

Risk Warning:
  Trading forex involves substantial risk of loss and is not suitable
  for all investors. Past performance is not indicative of future results.
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['single', 'continuous'], 
        default='single',
        help='Trading mode: single cycle or continuous operation (default: single)'
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run in demo mode (paper trading only)'
    )
    
    parser.add_argument(
        '--symbol', 
        default='USDJPY',
        help='Trading symbol (default: USDJPY)'
    )
    
    parser.add_argument(
        '--interval', 
        type=int, 
        default=300,
        help='Analysis interval in seconds for continuous mode (default: 300)'
    )
    
    parser.add_argument(
        '--risk-per-trade', 
        type=float, 
        default=1.0,
        help='Risk per trade as percentage of account (default: 1.0)'
    )
    
    parser.add_argument(
        '--max-positions', 
        type=int, 
        default=3,
        help='Maximum number of open positions (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Display startup banner
    print("\n" + "="*60)
    print("🤖 BLOB AI AUTOMATED TRADING SYSTEM 🤖")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Symbol: {args.symbol}")
    print(f"Demo Mode: {'YES' if args.demo else 'NO'}")
    print(f"Risk per Trade: {args.risk_per_trade}%")
    print(f"Max Positions: {args.max_positions}")
    if args.mode == 'continuous':
        print(f"Analysis Interval: {args.interval} seconds")
    print("="*60)
    
    # Safety confirmation for live trading
    if not args.demo:
        print("\n⚠️  WARNING: You are about to start LIVE TRADING ⚠️")
        print("This will use real money and can result in financial loss.")
        response = input("\nType 'YES' to confirm live trading: ")
        if response.upper() != 'YES':
            print("Trading cancelled. Use --demo flag for paper trading.")
            return
    
    # Build command arguments
    cmd_args = ['python', 'automated_trader.py', '--mode', args.mode]
    
    if args.demo:
        cmd_args.extend(['--demo'])
    
    if args.symbol != 'USDJPY':
        cmd_args.extend(['--symbol', args.symbol])
    
    if args.interval != 300:
        cmd_args.extend(['--interval', str(args.interval)])
    
    if args.risk_per_trade != 1.0:
        cmd_args.extend(['--risk-per-trade', str(args.risk_per_trade)])
    
    if args.max_positions != 3:
        cmd_args.extend(['--max-positions', str(args.max_positions)])
    
    print(f"\n🚀 Starting automated trader...")
    print(f"Command: {' '.join(cmd_args)}")
    print("\nPress Ctrl+C to stop the trading system\n")
    
    # Execute the automated trader
    try:
        os.system(' '.join(cmd_args))
    except KeyboardInterrupt:
        print("\n\n🛑 Trading system stopped by user")
        print("All open positions remain active in MT5")
        print("Check your MT5 terminal for position management")
    except Exception as e:
        print(f"\n❌ Error starting trading system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()