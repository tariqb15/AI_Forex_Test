#!/usr/bin/env python3
"""
Test Multi-Pair Signal Generation
Direct test of the multi-pair engine with simplified fallback
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_pair_engine import MultiPairAnalysisEngine
import MetaTrader5 as mt5
from datetime import datetime

def test_multi_pair_signals():
    """Test the multi-pair signal generation with fallback"""
    
    print("🧪 Testing Multi-Pair Signal Generation...")
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    try:
        # Test with a few pairs
        test_pairs = ["EURUSD", "GBPUSD", "USDJPY"]
        engine = MultiPairAnalysisEngine(test_pairs)
        
        print(f"✅ Testing {len(test_pairs)} pairs")
        
        # Test each pair individually
        for pair in test_pairs:
            print(f"\n📊 Testing {pair}...")
            
            try:
                result = engine.analyze_single_pair(pair)
                
                if 'error' in result:
                    print(f"❌ {pair}: {result['error']}")
                    continue
                
                signals = result.get('trading_signals', [])
                raw_signals = result.get('raw_signals', [])
                
                print(f"✅ {pair}: {len(signals)} final signals, {len(raw_signals)} raw signals")
                
                # Show signal details
                for i, signal in enumerate(signals):
                    if hasattr(signal, 'signal_type'):
                        print(f"   Signal {i+1}: {signal.signal_type} {signal.direction} @ {signal.entry_price:.5f} (Conf: {signal.confidence:.2f})")
                    else:
                        print(f"   Signal {i+1}: {signal}")
                
                # Check if simplified signals were used
                if len(signals) > 0 and len(raw_signals) == 0:
                    print(f"   🎯 Simplified fallback signals were used for {pair}")
                elif len(raw_signals) > 0:
                    print(f"   📈 Regular signals were generated for {pair}")
                else:
                    print(f"   📭 No signals generated for {pair}")
                    
                # Check timeframe analysis
                tf_analysis = result.get('timeframe_analysis', {})
                print(f"   📊 Analyzed timeframes: {list(tf_analysis.keys())}")
                
                for tf, data in tf_analysis.items():
                    if isinstance(data, dict):
                        obs = data.get('order_blocks', [])
                        fvgs = data.get('fair_value_gaps', [])
                        sweeps = data.get('liquidity_sweeps', [])
                        if isinstance(obs, list) and isinstance(fvgs, list) and isinstance(sweeps, list):
                            print(f"     {tf}: {len(obs)} OBs, {len(fvgs)} FVGs, {len(sweeps)} sweeps")
                
            except Exception as e:
                print(f"❌ Error testing {pair}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n🔄 Testing run_parallel_analysis method...")
        try:
            all_results = engine.run_parallel_analysis(max_workers=3)
            print(f"✅ run_parallel_analysis completed successfully")
            
            # Check the structure of results
            if 'pair_results' in all_results:
                pair_results = all_results['pair_results']
                print(f"   📊 Total results: {len(pair_results)}")
                total_signals = 0
                
                for pair, result in pair_results.items():
                    if 'error' not in result:
                        signals = result.get('trading_signals', [])
                        total_signals += len(signals)
                        print(f"   ✅ {pair}: {len(signals)} signals")
                    else:
                        print(f"   ❌ {pair}: ERROR - {result['error']}")
                
                print(f"\n📈 Total signals across all pairs: {total_signals}")
            else:
                print(f"   📊 Results structure: {list(all_results.keys())}")
            
        except Exception as e:
            print(f"❌ Error in run_parallel_analysis: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    test_multi_pair_signals()
    print("\n✅ Test completed")