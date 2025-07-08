#!/usr/bin/env python3
"""
Test MT5 tick data retrieval to debug _get_broker_time() issue
"""

import MetaTrader5 as mt5
from datetime import datetime
import pytz

def test_mt5_tick():
    """Test MT5 tick data retrieval"""
    print("🔍 Testing MT5 tick data retrieval...")
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return
    
    print(f"✅ MT5 initialized successfully. Version: {mt5.version()}")
    
    # Test symbol
    symbol = "USDJPY"
    
    # Check symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Symbol {symbol} not found")
        return
    
    print(f"✅ Symbol {symbol} found: {symbol_info.description}")
    print(f"   Selected: {symbol_info.select}")
    print(f"   Visible: {symbol_info.visible}")
    
    # Select symbol if not selected
    if not symbol_info.select:
        print(f"🔄 Selecting symbol {symbol}...")
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Failed to select symbol {symbol}")
            return
        print(f"✅ Symbol {symbol} selected")
    
    # Test tick data retrieval
    print(f"\n🔍 Testing tick data retrieval for {symbol}...")
    
    try:
        tick = mt5.symbol_info_tick(symbol)
        print(f"Tick result: {tick}")
        
        if tick:
            print(f"✅ Tick data received:")
            print(f"   Time: {tick.time}")
            print(f"   Bid: {tick.bid}")
            print(f"   Ask: {tick.ask}")
            print(f"   Last: {tick.last}")
            print(f"   Volume: {tick.volume}")
            
            # Convert time to datetime
            if tick.time:
                broker_time = datetime.fromtimestamp(tick.time)
                system_time = datetime.now()
                utc_time = datetime.utcnow()
                
                print(f"\n🕐 Time comparison:")
                print(f"   Broker time (from tick): {broker_time}")
                print(f"   System time (local):     {system_time}")
                print(f"   UTC time:                {utc_time}")
                
                # Calculate differences
                broker_system_diff = (broker_time - system_time).total_seconds() / 3600
                broker_utc_diff = (broker_time - utc_time).total_seconds() / 3600
                
                print(f"\n📊 Time differences:")
                print(f"   Broker vs System: {broker_system_diff:.2f} hours")
                print(f"   Broker vs UTC:    {broker_utc_diff:.2f} hours")
                
            else:
                print(f"❌ Tick time is None or invalid")
        else:
            print(f"❌ No tick data received")
            last_error = mt5.last_error()
            print(f"   MT5 Error: {last_error}")
            
    except Exception as e:
        print(f"❌ Exception during tick retrieval: {e}")
    
    # Test multiple attempts
    print(f"\n🔄 Testing multiple tick retrieval attempts...")
    for i in range(5):
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick and tick.time:
                broker_time = datetime.fromtimestamp(tick.time)
                print(f"   Attempt {i+1}: {broker_time}")
            else:
                print(f"   Attempt {i+1}: No valid tick data")
        except Exception as e:
            print(f"   Attempt {i+1}: Exception - {e}")
    
    # Cleanup
    mt5.shutdown()
    print(f"\n✅ Test completed")

if __name__ == "__main__":
    test_mt5_tick()