#!/usr/bin/env python3
"""
Simple Market Data Test
Basic test to see what market structures are being detected
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forex_engine import AgenticForexEngine
import MetaTrader5 as mt5
from datetime import datetime

def test_basic_market_detection():
    """Test basic market structure detection"""
    
    print("🔍 Testing Basic Market Structure Detection...")
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    # Test with EURUSD
    symbol = "EURUSD"
    engine = AgenticForexEngine(symbol)
    
    print(f"✅ Testing {symbol}")
    
    try:
        # Get raw market data
        print("\n📊 Getting raw market data...")
        mtf_data = engine.data_collector.get_mtf_data()
        
        if not mtf_data:
            print("❌ No market data available")
            return
        
        print(f"✅ Got data for {len(mtf_data)} timeframes")
        
        # Test each timeframe for basic structures
        for tf, df in mtf_data.items():
            print(f"\n📈 {tf} Timeframe ({len(df)} bars):")
            
            if len(df) < 50:
                print(f"   ⚠️  Insufficient data ({len(df)} bars)")
                continue
            
            # Test basic SMC detection
            try:
                # Structure breaks
                structure_breaks = engine.smc_analyzer.detect_market_structure_break(df)
                print(f"   Structure Breaks: {len(structure_breaks) if structure_breaks else 0}")
                
                # Order blocks
                order_blocks = engine.smc_analyzer.detect_order_blocks(df)
                print(f"   Order Blocks: {len(order_blocks) if order_blocks else 0}")
                
                # Fair Value Gaps
                fvgs = engine.smc_analyzer.detect_fair_value_gaps(df)
                print(f"   Fair Value Gaps: {len(fvgs) if fvgs else 0}")
                
                # Liquidity Sweeps
                sweeps = engine.smc_analyzer.detect_liquidity_sweeps(df)
                print(f"   Liquidity Sweeps: {len(sweeps) if sweeps else 0}")
                
                # Show some details if structures found
                if order_blocks and len(order_blocks) > 0:
                    print(f"   📦 Latest Order Block:")
                    latest_ob = order_blocks[-1]
                    print(f"      Type: {latest_ob.get('type', 'Unknown')}")
                    print(f"      Price: {latest_ob.get('high', 0):.5f} - {latest_ob.get('low', 0):.5f}")
                    print(f"      Strength: {latest_ob.get('strength', 0):.2f}")
                
                if sweeps and len(sweeps) > 0:
                    print(f"   🌊 Latest Liquidity Sweep:")
                    latest_sweep = sweeps[-1]
                    print(f"      Type: {latest_sweep.get('type', 'Unknown')}")
                    print(f"      Price: {latest_sweep.get('price', 0):.5f}")
                    print(f"      Strength: {latest_sweep.get('strength', 0):.2f}")
                
            except Exception as e:
                print(f"   ❌ Error analyzing {tf}: {e}")
        
        # Test current price and basic info
        print(f"\n💰 Current Market Info:")
        current_price = engine.data_collector.get_current_price()
        print(f"   Current Price: {current_price}")
        
        # Test session detection
        current_time = datetime.now()
        session = engine._get_current_session(current_time)
        print(f"   Current Session: {session}")
        
        # Test trend analysis
        if 'H1' in mtf_data:
            trend_context = engine.trend_analyzer.analyze_trend_context(mtf_data['H1'])
            print(f"   Trend Strength: {trend_context.momentum_strength:.2f}")
            print(f"   Trend Age: {trend_context.trend_age_hours:.1f} hours")
        
    except Exception as e:
        print(f"❌ Error in market test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    test_basic_market_detection()
    print("\n✅ Market test completed")