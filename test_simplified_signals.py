#!/usr/bin/env python3
"""
Test Simplified Signal Generation
Direct test of the new simplified signal generation method
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forex_engine import AgenticForexEngine
import MetaTrader5 as mt5
from datetime import datetime

def test_simplified_signals():
    """Test the new simplified signal generation method"""
    
    print("🧪 Testing Simplified Signal Generation...")
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    try:
        # Test with EURUSD
        symbol = "EURUSD"
        engine = AgenticForexEngine(symbol)
        
        print(f"✅ Testing {symbol}")
        
        # Run full analysis
        print("\n📊 Running full analysis...")
        analysis = engine.run_full_analysis()
        
        if not analysis:
            print("❌ Analysis failed")
            return
        
        print(f"✅ Analysis completed")
        
        # Test the simplified signal generation method
        print("\n🎯 Testing simplified signal generation...")
        
        # Check if the method exists
        if hasattr(engine, '_generate_simplified_trading_signals'):
            print("✅ Simplified signal method found")
            
            try:
                simplified_signals = engine._generate_simplified_trading_signals(analysis)
                print(f"📊 Simplified method generated: {len(simplified_signals)} signals")
                
                for i, signal in enumerate(simplified_signals):
                    print(f"\n   Signal {i+1}:")
                    print(f"     Type: {signal.get('signal_type', 'Unknown')}")
                    print(f"     Direction: {signal.get('direction', 'Unknown')}")
                    print(f"     Entry: {signal.get('entry_price', 0):.5f}")
                    print(f"     Stop Loss: {signal.get('stop_loss', 0):.5f}")
                    print(f"     Take Profit: {signal.get('take_profit', 0):.5f}")
                    print(f"     Confidence: {signal.get('confidence', 0):.2f}")
                    print(f"     Risk/Reward: {signal.get('risk_reward', 0):.2f}")
                    print(f"     Timeframe: {signal.get('timeframe', 'Unknown')}")
                    print(f"     Reason: {signal.get('reason', 'Unknown')}")
                
            except Exception as e:
                print(f"❌ Error in simplified signal generation: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Simplified signal method not found")
        
        # Test the regular signal generation for comparison
        print("\n🔄 Testing regular signal generation...")
        try:
            regular_signals = engine.generate_signals()
            print(f"📊 Regular method generated: {len(regular_signals)} signals")
            
            if regular_signals:
                for i, signal in enumerate(regular_signals):
                    print(f"\n   Regular Signal {i+1}:")
                    if hasattr(signal, 'signal_type'):
                        print(f"     Type: {signal.signal_type}")
                        print(f"     Direction: {signal.direction}")
                        print(f"     Entry: {signal.entry_price:.5f}")
                        print(f"     Confidence: {signal.confidence:.2f}")
                    else:
                        print(f"     Signal: {signal}")
        except Exception as e:
            print(f"❌ Error in regular signal generation: {e}")
            import traceback
            traceback.print_exc()
        
        # Show what market structures were detected
        print("\n📈 Market Structures Detected:")
        tf_analysis = analysis.get('timeframe_analysis', {})
        for tf, data in tf_analysis.items():
            if isinstance(data, dict):
                obs = data.get('order_blocks', [])
                sweeps = data.get('liquidity_sweeps', [])
                fvgs = data.get('fair_value_gaps', [])
                print(f"   {tf}: {len(obs)} OBs, {len(sweeps)} sweeps, {len(fvgs)} FVGs")
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    test_simplified_signals()
    print("\n✅ Test completed")