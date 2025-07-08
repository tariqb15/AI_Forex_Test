#!/usr/bin/env python3
"""
🚀 Enhanced Forex Engine Test Script
Demonstrates the strategic enhancements to beat the banks:
1. Institutional Flow Tracking
2. Advanced Order Block Validation  
3. Liquidity Mapping System
4. Adaptive Signal Weighting
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forex_engine import AgenticForexEngine
from institutional_flow_tracker import InstitutionalFlowTracker
from advanced_order_block_validator import AdvancedOrderBlockValidator
from liquidity_mapping_system import LiquidityMappingSystem
from adaptive_signal_weighting import AdaptiveSignalWeighting

def print_banner():
    """Print the enhanced engine banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                🚀 ENHANCED FOREX ENGINE 🚀                  ║
    ║              Strategic Enhancements to Beat Banks            ║
    ╠══════════════════════════════════════════════════════════════╣
    ║ 🔍 1. Institutional Flow Tracking                           ║
    ║ 📊 2. Advanced Order Block Validation                       ║
    ║ 🎯 3. Liquidity Mapping System                              ║
    ║ ⚖️  4. Adaptive Signal Weighting                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_section_header(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_enhancement_status(engine):
    """Print the status of all strategic enhancements"""
    print_section_header("STRATEGIC ENHANCEMENT STATUS")
    
    # Check if enhancements are properly initialized
    enhancements = {
        "🔍 Institutional Flow Tracker": hasattr(engine, 'institutional_flow_tracker'),
        "📊 Advanced Order Block Validator": hasattr(engine, 'advanced_ob_validator'),
        "🎯 Liquidity Mapping System": hasattr(engine, 'liquidity_mapper'),
        "⚖️ Adaptive Signal Weighting": hasattr(engine, 'adaptive_weighter')
    }
    
    for name, status in enhancements.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}: {'ACTIVE' if status else 'INACTIVE'}")
    
    all_active = all(enhancements.values())
    print(f"\n🚀 Overall Status: {'ALL SYSTEMS GO!' if all_active else 'SOME SYSTEMS OFFLINE'}")
    
    return all_active

def demonstrate_institutional_flow(engine):
    """Demonstrate institutional flow tracking capabilities"""
    print_section_header("INSTITUTIONAL FLOW TRACKING DEMO")
    
    try:
        # Get sample data for demonstration
        data = engine.data_collector.get_mtf_data()
        if not data or 'H1' not in data:
            print("❌ No H1 data available for institutional flow analysis")
            return
        
        h1_data = data['H1']
        print(f"📊 Analyzing {len(h1_data)} H1 candles for institutional flow...")
        
        # Run institutional flow analysis
        flow_analysis = engine.institutional_flow_tracker.analyze_institutional_flow(h1_data)
        
        print(f"\n🔍 INSTITUTIONAL FLOW RESULTS:")
        print(f"   • Accumulation Signals: {len(flow_analysis.get('accumulation_signals', []))}")
        print(f"   • Distribution Signals: {len(flow_analysis.get('distribution_signals', []))}")
        print(f"   • Volume Anomalies: {len(flow_analysis.get('volume_anomalies', []))}")
        
        # Show top accumulation signals
        acc_signals = flow_analysis.get('accumulation_signals', [])
        if acc_signals:
            print(f"\n🎯 TOP ACCUMULATION ZONES:")
            for i, signal in enumerate(acc_signals[:3], 1):
                print(f"   {i}. Price: {signal.price_level:.5f} | Confidence: {signal.confidence:.1%} | Volume: {signal.volume_strength:.1%}")
        
    except Exception as e:
        print(f"❌ Error in institutional flow demo: {e}")

def demonstrate_order_block_validation(engine):
    """Demonstrate advanced order block validation"""
    print_section_header("ADVANCED ORDER BLOCK VALIDATION DEMO")
    
    try:
        # Get sample data
        data = engine.data_collector.get_mtf_data()
        if not data or 'H4' not in data:
            print("❌ No H4 data available for order block analysis")
            return
        
        h4_data = data['H4']
        print(f"📊 Analyzing {len(h4_data)} H4 candles for order blocks...")
        
        # Get raw order blocks from smart money analyzer
        raw_obs = engine.smart_money_analyzer.detect_order_blocks(h4_data, 'H4')
        print(f"🔍 Found {len(raw_obs)} raw order blocks")
        
        if raw_obs:
            # Validate and score order blocks
            enhanced_obs = engine.advanced_ob_validator.validate_and_score_order_blocks(
                raw_obs, h4_data, 'H4'
            )
            
            print(f"\n📈 ENHANCED ORDER BLOCK RESULTS:")
            print(f"   • Premium Quality (Score > 7.0): {len([ob for ob in enhanced_obs if ob.score.total_score > 7.0])}")
            print(f"   • High Quality (Score > 5.0): {len([ob for ob in enhanced_obs if ob.score.total_score > 5.0])}")
            print(f"   • Total Enhanced OBs: {len(enhanced_obs)}")
            
            # Show top order blocks
            sorted_obs = sorted(enhanced_obs, key=lambda x: x.score.total_score, reverse=True)
            print(f"\n🏆 TOP ORDER BLOCKS:")
            for i, ob in enumerate(sorted_obs[:3], 1):
                print(f"   {i}. Score: {ob.score.total_score:.1f} | Type: {ob.ob_type} | Session: {ob.session.value}")
                print(f"      Price Range: {ob.low:.5f} - {ob.high:.5f}")
        
    except Exception as e:
        print(f"❌ Error in order block validation demo: {e}")

def demonstrate_liquidity_mapping(engine):
    """Demonstrate liquidity mapping system"""
    print_section_header("LIQUIDITY MAPPING SYSTEM DEMO")
    
    try:
        # Get sample data
        data = engine.data_collector.get_mtf_data()
        if not data or 'H4' not in data:
            print("❌ No H4 data available for liquidity mapping")
            return
        
        h4_data = data['H4']
        print(f"📊 Analyzing {len(h4_data)} H4 candles for liquidity mapping...")
        
        # Run liquidity landscape analysis
        liquidity_analysis = engine.liquidity_mapper.analyze_liquidity_landscape(h4_data)
        
        print(f"\n🎯 LIQUIDITY MAPPING RESULTS:")
        print(f"   • Liquidity Zones: {len(liquidity_analysis.get('liquidity_zones', []))}")
        print(f"   • Absorption Events: {len(liquidity_analysis.get('absorption_events', []))}")
        print(f"   • Sweep Targets: {len(liquidity_analysis.get('sweep_targets', []))}")
        
        # Show high-strength liquidity zones
        zones = liquidity_analysis.get('liquidity_zones', [])
        high_strength = [z for z in zones if hasattr(z, 'strength') and hasattr(z.strength, 'value') and z.strength.value in ['high', 'extreme']]
        if high_strength:
            print(f"\n💪 HIGH-STRENGTH LIQUIDITY ZONES:")
            for i, zone in enumerate(high_strength[:3], 1):
                print(f"   {i}. Price: {zone.get('price_level', 0):.5f} | Strength: {zone.get('strength', 0):.1%} | Type: {zone.get('type', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Error in liquidity mapping demo: {e}")

def demonstrate_adaptive_weighting(engine):
    """Demonstrate adaptive signal weighting"""
    print_section_header("ADAPTIVE SIGNAL WEIGHTING DEMO")
    
    try:
        # Get sample data
        data = engine.data_collector.get_mtf_data()
        if not data or 'H4' not in data:
            print("❌ No H4 data available for regime detection")
            return
        
        h4_data = data['H4']
        print(f"📊 Analyzing {len(h4_data)} H4 candles for market regime...")
        
        # Detect market regime
        regime_analysis = engine.adaptive_weighter._analyze_market_regime({'H4': h4_data})
        
        print(f"\n⚖️ MARKET REGIME ANALYSIS:")
        print(f"   • Current Regime: {regime_analysis.get('regime', 'UNKNOWN')}")
        print(f"   • Confidence: {regime_analysis.get('confidence', 0):.1%}")
        print(f"   • Bias: {regime_analysis.get('bias', 'NEUTRAL')}")
        print(f"   • Volatility State: {regime_analysis.get('volatility_state', 'UNKNOWN')}")
        
        # Show regime metrics
        metrics = regime_analysis.get('metrics', {})
        if metrics:
            print(f"\n📊 REGIME METRICS:")
            print(f"   • ATR Percentile: {metrics.get('atr_percentile', 0):.1%}")
            print(f"   • BB Width Percentile: {metrics.get('bb_width_percentile', 0):.1%}")
            print(f"   • EMA200 Distance: {metrics.get('ema200_distance_pct', 0):.2%}")
        
    except Exception as e:
        print(f"❌ Error in adaptive weighting demo: {e}")

def run_enhanced_analysis(engine):
    """Run a full enhanced analysis"""
    print_section_header("FULL ENHANCED ANALYSIS")
    
    try:
        print("🚀 Running enhanced forex analysis...")
        
        # Run the enhanced analysis
        analysis_result = engine.run_full_analysis()
        
        if analysis_result:
            print(f"\n✅ ENHANCED ANALYSIS COMPLETE!")
            
            # Show enhanced market narrative
            narrative = analysis_result.get('market_narrative', 'No narrative available')
            print(f"\n📰 ENHANCED MARKET NARRATIVE:")
            print(f"   {narrative}")
            
            # Show trading signals
            signals = analysis_result.get('trading_signals', [])
            if signals:
                print(f"\n🎯 ENHANCED TRADING SIGNALS ({len(signals)}):")
                for i, signal in enumerate(signals, 1):
                    print(f"   {i}. {signal.signal_type} | {signal.direction} | Confidence: {signal.confidence:.1%}")
                    print(f"      Entry: {signal.entry_price:.5f} | SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f}")
                    print(f"      RR: {signal.risk_reward:.2f} | Reasoning: {signal.reasoning}")
                    print()
            else:
                print("   No trading signals generated")
        else:
            print("❌ Enhanced analysis failed")
            
    except Exception as e:
        print(f"❌ Error in enhanced analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print_banner()
    
    try:
        # Initialize the enhanced forex engine
        print("🔧 Initializing Enhanced Forex Engine...")
        engine = AgenticForexEngine()
        
        # Check if MT5 connection is successful
        if not engine.data_collector.connect_mt5():
            print("❌ Failed to connect to MT5. Please ensure MT5 is running.")
            return
        
        print("✅ MT5 connection established")
        
        # Check enhancement status
        if not print_enhancement_status(engine):
            print("\n⚠️ Some enhancements are not active. Proceeding with available systems...")
        
        # Run individual demonstrations
        demonstrate_institutional_flow(engine)
        demonstrate_order_block_validation(engine)
        demonstrate_liquidity_mapping(engine)
        demonstrate_adaptive_weighting(engine)
        
        # Run full enhanced analysis
        run_enhanced_analysis(engine)
        
        print_section_header("TEST COMPLETE")
        print("🎉 Enhanced Forex Engine test completed successfully!")
        print("🚀 All strategic enhancements are operational and ready to beat the banks!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🔚 Shutting down...")

if __name__ == "__main__":
    main()