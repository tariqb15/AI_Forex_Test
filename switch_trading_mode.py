#!/usr/bin/env python3
"""
Trading Mode Switcher
Switch between DEMO and LIVE trading modes
"""

import sys
from config import ForexEngineConfig, TradingMode

def switch_to_demo():
    """Switch to demo mode"""
    config = ForexEngineConfig()
    config.set_trading_mode(TradingMode.DEMO)
    print("✅ Switched to DEMO mode")
    print("📝 Trades will be simulated (no real MT5 orders)")
    print(f"💰 Demo lot size: {config.trading.lot_size}")
    return config

def switch_to_live():
    """Switch to live mode"""
    config = ForexEngineConfig()
    config.set_trading_mode(TradingMode.LIVE)
    print("⚠️  Switched to LIVE mode")
    print("💸 Trades will be placed on real MT5 account")
    print(f"💰 Live lot size: {config.trading.lot_size}")
    print(f"🎯 Minimum confidence: {config.trading.min_confidence}")
    return config

def show_current_mode():
    """Show current trading mode"""
    config = ForexEngineConfig()
    mode_name = config.trading_mode.value.upper()
    print(f"📊 Current trading mode: {mode_name}")
    
    if config.trading_mode == TradingMode.DEMO:
        print("🔒 Safe mode - trades are simulated")
    elif config.trading_mode == TradingMode.LIVE:
        print("⚠️  Live mode - real money at risk")
    
    print(f"💰 Lot size: {config.trading.lot_size}")
    print(f"🎯 Min confidence: {config.trading.min_confidence}")
    return config

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python switch_trading_mode.py [demo|live|status]")
        print("")
        print("Commands:")
        print("  demo   - Switch to demo mode (safe, simulated trades)")
        print("  live   - Switch to live mode (real trades)")
        print("  status - Show current mode")
        return
    
    command = sys.argv[1].lower()
    
    if command == "demo":
        switch_to_demo()
    elif command == "live":
        print("⚠️  WARNING: This will enable REAL trading!")
        confirm = input("Type 'YES' to confirm live trading: ")
        if confirm == "YES":
            switch_to_live()
        else:
            print("❌ Live mode activation cancelled")
    elif command == "status":
        show_current_mode()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use: demo, live, or status")

if __name__ == "__main__":
    main()