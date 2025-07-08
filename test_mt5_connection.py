#!/usr/bin/env python3
"""
Test MT5 Connection and Trading Permissions
"""

import MetaTrader5 as mt5

def test_mt5_connection():
    print("=== MT5 Connection Test ===")
    
    # Initialize MT5
    init_result = mt5.initialize()
    print(f"Initialize: {init_result}")
    
    if not init_result:
        print("Failed to connect to MT5")
        return False
    
    # Check account info
    account = mt5.account_info()
    if account:
        print(f"Account: {account.login}")
        print(f"Server: {account.server}")
        print(f"Balance: {account.balance}")
        print(f"Equity: {account.equity}")
        print(f"Margin: {account.margin}")
        print(f"Trade allowed: {account.trade_allowed}")
        print(f"Trade expert: {account.trade_expert}")
    else:
        print("Failed to get account info")
    
    # Check terminal info
    terminal = mt5.terminal_info()
    if terminal:
        print(f"Terminal connected: {terminal.connected}")
        print(f"Trade allowed: {terminal.trade_allowed}")
        print(f"Experts enabled: {terminal.experts_enabled}")
        print(f"Auto trading: {terminal.auto_trading}")
    else:
        print("Failed to get terminal info")
    
    # Test symbol info
    symbol = "EURAUD"
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info:
        print(f"Symbol {symbol} info: OK")
        print(f"Trade mode: {symbol_info.trade_mode}")
        print(f"Trade execution: {symbol_info.trade_execution}")
    else:
        print(f"Failed to get {symbol} info")
    
    # Test tick data
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        print(f"Tick data for {symbol}: Bid={tick.bid}, Ask={tick.ask}")
    else:
        print(f"Failed to get tick data for {symbol}")
    
    mt5.shutdown()
    return True

if __name__ == "__main__":
    test_mt5_connection()