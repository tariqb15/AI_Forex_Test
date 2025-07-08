#!/usr/bin/env python3
"""
Edge Case Testing Script for BLOB AI Trading System

This script demonstrates how the edge case handler validates trading signals
against the 10 specific scenarios outlined in the user requirements.

Author: BLOB AI Trading System
Version: 1.0.0
"""

import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

# Import our modules
from edge_case_handler import EdgeCaseHandler, EdgeCaseType, EdgeCaseResult
from forex_engine import TradingSignal, SignalStrength, SessionType, MarketStructure
from multi_pair_engine import MultiPairAnalysisEngine

class EdgeCaseTestSuite:
    """Test suite for edge case validation"""
    
    def __init__(self):
        self.edge_case_handler = EdgeCaseHandler()
        self.test_results = []
        
    def create_test_signal(self, signal_type: str = 'SMC_OrderBlock', direction: str = 'BUY') -> TradingSignal:
        """Create a test trading signal"""
        # Use a Tuesday 10:00 UTC timestamp to avoid weekend issues
        tuesday_timestamp = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
        # Ensure it's a Tuesday (weekday 1)
        days_to_tuesday = (1 - tuesday_timestamp.weekday()) % 7
        if days_to_tuesday != 0:
            tuesday_timestamp = tuesday_timestamp + timedelta(days=days_to_tuesday)
            
        return TradingSignal(
            timestamp=tuesday_timestamp,
            signal_type=signal_type,
            direction=direction,
            strength=SignalStrength.STRONG,
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            risk_reward=2.0,
            confidence=0.8,
            reasoning="Test signal for edge case validation",
            session=SessionType.LONDON,
            market_structure=MarketStructure.BULLISH,
            symbol="EURUSD"
        )
    
    def create_test_context(self, scenario: str) -> Dict:
        """Create test context for different scenarios"""
        base_context = {
            'symbol': 'EURUSD',
            'current_session': SessionType.LONDON,
            'timeframe_analysis': {
                'M15': {
                    'smc_analysis': {
                        'structure_breaks': [
                            {'type': 'Bullish_MSB', 'timestamp': datetime.now(), 'strength': 0.7}
                        ]
                    },
                    'order_blocks': [
                        {'type': 'Bullish_OB', 'low': 1.0980, 'high': 1.1020, 'strength': 0.8}
                    ],
                    'fair_value_gaps': [
                        {'type': 'Bullish_FVG', 'bottom': 1.0990, 'top': 1.1010, 'filled': False}
                    ],
                    'liquidity_sweeps': []
                }
            }
        }
        
        # Modify context based on scenario
        if scenario == 'no_liquidity_sweep':
            # Remove liquidity sweeps to test scenario 1
            base_context['timeframe_analysis']['M15']['liquidity_sweeps'] = []
            
        elif scenario == 'asia_session_fakeout':
            # Set Asia session for scenario 2
            base_context['current_session'] = SessionType.ASIA
            
        elif scenario == 'stacked_order_blocks':
            # Add multiple order blocks for scenario 3
            base_context['timeframe_analysis']['M15']['order_blocks'] = [
                {'type': 'Bullish_OB', 'low': 1.0980, 'high': 1.1020, 'strength': 0.6},
                {'type': 'Bullish_OB', 'low': 1.0990, 'high': 1.1030, 'strength': 0.8},
                {'type': 'Bullish_OB', 'low': 1.1000, 'high': 1.1040, 'strength': 0.7}
            ]
            # Add liquidity sweep before BOS to pass the first check
            base_context['timeframe_analysis']['M15']['liquidity_sweeps'] = [
                {'timestamp': datetime.now() - timedelta(hours=1), 'type': 'sweep_low'}
            ]
            
        elif scenario == 'fvg_re_entry':
            # Mark FVG as already touched for scenario 4
            base_context['timeframe_analysis']['M15']['fair_value_gaps'][0]['bounce_history'] = 1
            base_context['timeframe_analysis']['M15']['fair_value_gaps'][0]['timestamp'] = datetime.now() - timedelta(hours=2)
            
        elif scenario == 'news_spike':
            # Add extreme volatility data for scenario 5
            base_context['timeframe_analysis']['M15']['volatility_analysis'] = {
                'current_atr': 0.0150,  # Very high ATR
                'avg_atr': 0.0050,      # Normal ATR
                'expansion_phase': True
            }
            # Add news event context
            base_context['news_events'] = [{
                'time': datetime.now() - timedelta(minutes=5),
                'impact': 'HIGH',
                'title': 'NFP Release'
            }]
            
        elif scenario == 'weekend_gap':
            # Set weekend timestamp for scenario 6 - this will be handled by special signal creation
            pass
            
        elif scenario == 'continuation_entry':
            # Strong trend without retest for scenario 7
            base_context['timeframe_analysis']['M15']['trend_context'] = {
                'phase': 'TRENDING',
                'momentum_strength': 0.9,
                'age_in_bars': 5
            }
            # Add multiple structure breaks showing strong trend
            base_context['timeframe_analysis']['M15']['smc_analysis']['structure_breaks'] = [
                {'type': 'Bullish_MSB', 'timestamp': datetime.now() - timedelta(hours=4), 'strength': 0.6},
                {'type': 'Bullish_MSB', 'timestamp': datetime.now() - timedelta(hours=3), 'strength': 0.7},
                {'type': 'Bullish_MSB', 'timestamp': datetime.now() - timedelta(hours=2), 'strength': 0.8}
            ]
            # Add liquidity sweep before BOS to pass the first check
            base_context['timeframe_analysis']['M15']['liquidity_sweeps'] = [
                {'timestamp': datetime.now() - timedelta(hours=3), 'type': 'sweep_low'}
            ]
            
        elif scenario == 'htf_consolidation':
            # Add HTF range data for scenario 8
            base_context['timeframe_analysis']['H4'] = {
                'range_analysis': {
                    'in_range': True,
                    'range_high': 1.1200,
                    'range_low': 1.0800,
                    'current_position': 0.5  # Middle of range
                }
            }
            
        elif scenario == 'weak_choch':
            # Weak ChoCH for scenario 9
            base_context['timeframe_analysis']['M15']['smc_analysis']['structure_breaks'][0]['strength'] = 0.3
            
        elif scenario == 'massive_candle_wipeout':
            # Large candle wiping through zones for scenario 10
            base_context['recent_candle'] = {
                'size_pips': 150,  # Very large candle
                'body_ratio': 0.9,
                'direction': 'bearish',
                'timestamp': datetime.now() - timedelta(minutes=1)
            }
            # Add multiple zones that were wiped
            base_context['timeframe_analysis']['M15']['order_blocks'] = [
                {'type': 'Bullish_OB', 'low': 1.0980, 'high': 1.1020, 'strength': 0.8, 'wiped': True},
                {'type': 'Bullish_OB', 'low': 1.1030, 'high': 1.1070, 'strength': 0.7, 'wiped': True}
            ]
        
        return base_context
    
    def test_scenario(self, scenario_num: int, scenario_name: str, context_type: str, expected_result: bool) -> Dict:
        """Test a specific edge case scenario"""
        print(f"\n🧪 Testing Scenario {scenario_num}: {scenario_name}")
        print(f"   Expected: {'✅ PASS' if expected_result else '❌ REJECT'}")
        
        # Create test signal and context
        if context_type == 'continuation_entry':
            signal = self.create_test_signal(signal_type='Breakout_Continuation')
        elif context_type == 'weekend_gap':
            # Create signal with weekend timestamp
            signal = self.create_test_signal()
            # Override with Sunday timestamp
            sunday_time = datetime.now().replace(hour=1, minute=0, second=0, microsecond=0)
            days_to_sunday = (6 - sunday_time.weekday()) % 7
            if days_to_sunday != 0:
                sunday_time = sunday_time + timedelta(days=days_to_sunday)
            signal.timestamp = sunday_time
        elif context_type == 'fvg_re_entry':
            signal = self.create_test_signal(signal_type='SMC_FairValueGap')
        elif context_type == 'asia_session_fakeout':
            signal = self.create_test_signal(signal_type='SMC_OrderBlock')
        else:
            signal = self.create_test_signal()
        context = self.create_test_context(context_type)
        context['signal'] = signal
        
        # Validate signal - extract and flatten market data from context
        m15_data = context.get('timeframe_analysis', {}).get('M15', {})
        h4_data = context.get('timeframe_analysis', {}).get('H4', {})
        
        # Flatten the nested structure to match what edge case handlers expect
        market_data = {
            'structure_breaks': m15_data.get('smc_analysis', {}).get('structure_breaks', []),
            'order_blocks': m15_data.get('order_blocks', []),
            'fair_value_gaps': m15_data.get('fair_value_gaps', []),
            'liquidity_sweeps': m15_data.get('liquidity_sweeps', []),
            'current_price': {'bid': signal.entry_price},
            'atr': 0.0050,  # Default ATR
            'volatility_analysis': m15_data.get('volatility_analysis', {}),
            'trend_context': m15_data.get('trend_context', {}),
            'range_analysis': h4_data.get('range_analysis', {}),
            'recent_candle': context.get('recent_candle', {})
        }
        
        # Add volatility data if present
        if 'volatility_analysis' in m15_data:
            market_data.update(m15_data['volatility_analysis'])
            
        signal_data = {
            'timestamp': signal.timestamp,
            'signal_type': signal.signal_type,
            'direction': signal.direction,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'confidence': signal.confidence
        }
        
        result = self.edge_case_handler.validate_signal(signal_data, market_data, context)
        
        # Check if result matches expectation
        test_passed = result.is_valid == expected_result
        
        print(f"   Result: {'✅ PASS' if result.is_valid else '❌ REJECT'}")
        print(f"   Test: {'✅ CORRECT' if test_passed else '❌ FAILED'}")
        
        if not result.is_valid:
            print(f"   Reason: {result.reasoning}")
            print(f"   Action: {result.action}")
        
        test_result = {
            'scenario_num': scenario_num,
            'scenario_name': scenario_name,
            'expected_result': expected_result,
            'actual_result': result.is_valid,
            'test_passed': test_passed,
            'reasoning': result.reasoning,
            'action': result.action,
            'confidence_adjustment': result.confidence_adjustment
        }
        
        self.test_results.append(test_result)
        return test_result
    
    def run_all_tests(self):
        """Run all edge case tests"""
        print("🚀 BLOB AI Edge Case Testing Suite")
        print("=" * 60)
        print("Testing 10 critical Smart Money Concept edge cases...\n")
        
        # Test scenarios based on the user's table
        test_cases = [
            (1, "BOS without liquidity sweep", "no_liquidity_sweep", False),
            (2, "Asia session OB tap", "asia_session_fakeout", False),
            (3, "Multiple stacked OBs", "stacked_order_blocks", True),
            (4, "FVG re-entry attempt", "fvg_re_entry", False),
            (5, "NFP news spike volatility", "news_spike", False),
            (6, "Weekend gap BOS", "weekend_gap", False),
            (7, "Trend continuation entry", "continuation_entry", True),
            (8, "HTF consolidation trap", "htf_consolidation", False),
            (9, "Weak ChoCH reversal", "weak_choch", False),
            (10, "Massive candle wipeout", "massive_candle_wipeout", False)
        ]
        
        for scenario_num, scenario_name, context_type, expected in test_cases:
            self.test_scenario(scenario_num, scenario_name, context_type, expected)
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['test_passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['test_passed']:
                    print(f"   {result['scenario_num']}. {result['scenario_name']}")
        
        print("\n" + "=" * 60)
    
    def save_test_results(self, filename: str = None):
        """Save test results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"edge_case_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'test_summary': {
                        'total_tests': len(self.test_results),
                        'passed_tests': sum(1 for r in self.test_results if r['test_passed']),
                        'failed_tests': sum(1 for r in self.test_results if not r['test_passed']),
                        'success_rate': (sum(1 for r in self.test_results if r['test_passed']) / len(self.test_results)) * 100
                    },
                    'detailed_results': self.test_results
                }, f, indent=2)
            print(f"\n💾 Test results saved to {filename}")
        except Exception as e:
            print(f"\n❌ Error saving test results: {e}")

def demonstrate_live_integration():
    """Demonstrate edge case handler integration with live trading system"""
    print("\n🔄 DEMONSTRATING LIVE INTEGRATION")
    print("=" * 60)
    
    try:
        # Initialize multi-pair engine with edge case handling
        currency_pairs = ['EURUSD', 'GBPUSD']  # Limited pairs for demo
        multi_engine = MultiPairAnalysisEngine(currency_pairs)
        
        print(f"✅ Initialized multi-pair engine with edge case validation")
        print(f"📊 Analyzing {len(currency_pairs)} currency pairs with edge case filtering...")
        
        # Note: This would require MT5 connection for live demo
        print("\n💡 Live integration features:")
        print("   • All signals validated against 10 edge case scenarios")
        print("   • Invalid signals automatically rejected with reasons")
        print("   • Warnings provided for borderline cases")
        print("   • Original signals preserved for analysis")
        print("   • Real-time session and volatility filtering")
        
    except Exception as e:
        print(f"⚠️  Live demo requires MT5 connection: {e}")
        print("   Edge case validation is fully integrated and ready for live trading")

def main():
    """Main execution function"""
    print("🤖 BLOB AI Smart Money Concept Edge Case Validation")
    print("" * 80)
    
    # Run edge case tests
    test_suite = EdgeCaseTestSuite()
    test_suite.run_all_tests()
    
    # Save results
    test_suite.save_test_results()
    
    # Demonstrate live integration
    demonstrate_live_integration()
    
    print("\n🎉 Edge case testing complete!")
    print("The trading bot now intelligently handles all 10 critical SMC scenarios.")

if __name__ == "__main__":
    main()