#!/usr/bin/env python3
"""
BLOB AI - Multi-Pair Analysis Test
Runs a one-time analysis of all specified currency pairs
"""

import time
from datetime import datetime
from multi_pair_engine import MultiPairAnalysisEngine

def main():
    """Run a one-time multi-pair analysis"""
    
    # Define currency pairs to analyze
    currency_pairs = [
        'EURJPY', 'GBPJPY', 'USDJPY', 'CHFJPY', 'EURAUD', 'AUDJPY', 'EURNZD', 
        'GBPNZD', 'NZDUSD', 'GBPUSD', 'NZDJPY', 'GBPCHF', 'GBPAUD', 'EURCAD', 
        'AUDUSD', 'USDCHF', 'EURCHF', 'AUDCAD'
    ]
    
    print("🚀 BLOB AI Multi-Currency Pair Analysis Test")
    print("=" * 60)
    print(f"📊 Analyzing {len(currency_pairs)} currency pairs:")
    for i, pair in enumerate(currency_pairs, 1):
        print(f"   {i:2d}. {pair}")
    print()
    
    try:
        # Initialize multi-pair engine
        print("🔧 Initializing multi-pair analysis engine...")
        multi_engine = MultiPairAnalysisEngine(currency_pairs)
        
        if not multi_engine.engines:
            print("❌ No engines initialized successfully. Exiting.")
            return
        
        print(f"✅ Initialized {len(multi_engine.engines)} engines successfully\n")
        
        # Run parallel analysis
        print("🔄 Starting parallel analysis...")
        start_time = time.time()
        
        results = multi_engine.run_parallel_analysis(max_workers=6)
        
        end_time = time.time()
        
        print("\n" + "=" * 60)
        print("📊 ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Display summary
        print(f"⏱️  Duration: {results['analysis_duration_seconds']:.2f} seconds")
        print(f"✅ Successful: {results['successful_analyses']}/{results['total_pairs']} pairs")
        print(f"❌ Failed: {results['failed_analyses']} pairs")
        print(f"📈 Total Signals: {results['total_signals']}")
        
        # Show failed pairs if any
        if results['failed_analyses'] > 0:
            print("\n⚠️  FAILED ANALYSES:")
            for pair, result in results['pair_results'].items():
                if 'error' in result:
                    print(f"   {pair}: {result['error']}")
        
        # Show top signals
        top_signals = multi_engine.get_top_signals(min_confidence=0.5, max_signals=10)
        if top_signals:
            print(f"\n🎯 TOP {len(top_signals)} TRADING OPPORTUNITIES:")
            print("-" * 80)
            
            for i, signal in enumerate(top_signals, 1):
                pair = getattr(signal, 'currency_pair', 'Unknown')
                print(f"{i:2d}. {pair} - {signal.direction} {signal.signal_type}")
                print(f"    Confidence: {signal.confidence:.1%} | Risk:Reward = 1:{signal.risk_reward:.1f}")
                print(f"    Entry: {signal.entry_price:.5f} | SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f}")
                print(f"    Reasoning: {signal.reasoning}")
                print()
        else:
            print("\n📊 No signals found across all pairs with minimum confidence threshold")
        
        # Show pair-by-pair summary
        print("\n📈 PAIR-BY-PAIR SUMMARY:")
        print("-" * 60)
        
        successful_pairs = []
        for pair in currency_pairs:
            pair_result = results['pair_results'].get(pair, {})
            
            if 'error' in pair_result:
                print(f"{pair:8s} ❌ Failed: {pair_result['error'][:50]}...")
            else:
                signals = pair_result.get('trading_signals', [])
                current_price = pair_result.get('current_price', {})
                
                status = "✅ Success"
                signal_info = f"{len(signals)} signals"
                
                if current_price:
                    bid = current_price.get('bid', 0)
                    ask = current_price.get('ask', 0)
                    spread = current_price.get('spread', 0)
                    
                    if 'JPY' in pair:
                        spread_pips = spread * 100
                    else:
                        spread_pips = spread * 10000
                    
                    price_info = f"Price: {bid:.5f}/{ask:.5f} (Spread: {spread_pips:.1f} pips)"
                else:
                    price_info = "Price: N/A"
                
                print(f"{pair:8s} {status} | {signal_info:12s} | {price_info}")
                successful_pairs.append(pair)
        
        print(f"\n📊 Summary: {len(successful_pairs)}/{len(currency_pairs)} pairs analyzed successfully")
        
        # Generate and display report
        print("\n📄 Generating comprehensive report...")
        report = multi_engine.generate_multi_pair_report()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"multi_pair_analysis_report_{timestamp}.md"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📁 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
        
        # Save results to JSON
        results_file = multi_engine.save_results(f"multi_pair_results_{timestamp}.json")
        if results_file:
            print(f"📁 Raw results saved to: {results_file}")
        
        print("\n" + "=" * 60)
        print("🎉 Multi-pair analysis test complete!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            if 'multi_engine' in locals():
                multi_engine.disconnect_all()
                print("\n🔌 All MT5 connections closed")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

if __name__ == "__main__":
    main()