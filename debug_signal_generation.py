#!/usr/bin/env python3
"""
Debug Signal Generation
Test script to identify why no signals are being generated
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forex_engine import AgenticForexEngine
from config import ForexEngineConfig, RiskLevel
from loguru import logger
import MetaTrader5 as mt5

def debug_signal_generation():
    """Debug the signal generation process step by step"""
    
    # Initialize configuration with very low thresholds
    config = ForexEngineConfig()
    config.set_risk_level(RiskLevel.AGGRESSIVE)
    
    # Further lower the thresholds for debugging
    config.trading.min_confidence = 0.1  # Very low threshold
    config.trading.min_risk_reward = 0.5  # Very low threshold
    
    print(f"🔧 Debug Configuration:")
    print(f"   Min Confidence: {config.trading.min_confidence}")
    print(f"   Min Risk/Reward: {config.trading.min_risk_reward}")
    
    # Test with a single pair
    test_symbol = "EURUSD"
    
    try:
        # Initialize engine
        engine = AgenticForexEngine(test_symbol)
        
        print(f"✅ Engine initialized for {test_symbol}")
        
        # Check if MT5 is available
        import MetaTrader5 as mt5
        if not mt5.initialize():
            print("❌ Failed to initialize MT5")
            return
            
        print(f"✅ MT5 initialized")
        
        # Get market data through data collector
        print("📊 Collecting market data...")
        mtf_data = engine.data_collector.get_mtf_data()
        
        if not mtf_data:
            print("❌ Failed to get market data")
            return
            
        print(f"✅ Market data collected: {len(mtf_data)} timeframes")
        for tf in mtf_data.keys():
            print(f"   {tf}: {len(mtf_data[tf])} bars")
        
        # Run analysis
        print("🔍 Running market analysis...")
        analysis = engine.run_full_analysis()
        
        if not analysis:
            print("❌ Analysis failed")
            return
            
        print(f"✅ Analysis completed")
        
        # Check what was found in analysis
        for tf, data in analysis.items():
            if isinstance(data, dict):
                print(f"\n📈 {tf} Analysis:")
                
                # Check structure breaks
                structure_breaks = data.get('structure_breaks', [])
                print(f"   Structure Breaks: {len(structure_breaks)}")
                
                # Check order blocks
                order_blocks = data.get('order_blocks', [])
                print(f"   Order Blocks: {len(order_blocks)}")
                
                # Check FVGs
                fvgs = data.get('fair_value_gaps', [])
                print(f"   Fair Value Gaps: {len(fvgs)}")
                
                # Check liquidity sweeps
                liquidity_sweeps = data.get('liquidity_sweeps', [])
                print(f"   Liquidity Sweeps: {len(liquidity_sweeps)}")
        
        # Generate signals with debug info
        print("\n🎯 Generating signals...")
        signals = engine.generate_signals()
        
        print(f"\n📊 Signal Generation Results:")
        print(f"   Total signals generated: {len(signals) if signals else 0}")
        
        if signals:
            for i, signal in enumerate(signals):
                print(f"\n   Signal {i+1}:")
                print(f"     Type: {signal.get('signal_type', 'Unknown')}")
                print(f"     Direction: {signal.get('direction', 'Unknown')}")
                print(f"     Confidence: {signal.get('confidence', 0):.2f}")
                print(f"     Risk/Reward: {signal.get('risk_reward', 0):.2f}")
                print(f"     Entry: {signal.get('entry_price', 0)}")
        else:
            print("   ❌ No signals generated")
            
            # Try to understand why
            print("\n🔍 Debugging signal generation...")
            
            # Check if we have any raw signals before filtering
            try:
                # Call the internal signal generation method
                raw_signals = engine._generate_enhanced_trading_signals(analysis)
                print(f"   Raw signals before filtering: {len(raw_signals) if raw_signals else 0}")
                
                if raw_signals:
                    print("   Raw signals found but filtered out. Checking filters...")
                    for signal in raw_signals:
                        conf = signal.get('confidence', 0)
                        rr = signal.get('risk_reward', 0)
                        print(f"     Signal: conf={conf:.2f}, rr={rr:.2f}")
                        print(f"     Passes conf filter: {conf >= config.trading.min_confidence}")
                        print(f"     Passes rr filter: {rr >= config.trading.min_risk_reward}")
                        
            except Exception as e:
                print(f"   Error accessing raw signals: {e}")
        
    except Exception as e:
        print(f"❌ Error in debug process: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if mt5.initialize():
            mt5.shutdown()

if __name__ == "__main__":
    print("🚀 Starting Signal Generation Debug...")
    debug_signal_generation()
    print("\n✅ Debug completed")