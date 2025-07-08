#!/usr/bin/env python3
"""
Multi-Currency Pair Analysis Engine for BLOB AI

Analyzes multiple currency pairs simultaneously using the existing
BLOB AI engine architecture.

Author: BLOB AI Trading System
Version: 1.0.0
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import asdict

from forex_engine import AgenticForexEngine, TradingSignal, SignalStrength
from config import ForexEngineConfig
from edge_case_handler import EdgeCaseHandler

# Import the complete forex_engine module for full functionality
import forex_engine

class MultiPairAnalysisEngine:
    """Multi-currency pair analysis engine"""
    
    def __init__(self, currency_pairs: List[str]):
        self.currency_pairs = currency_pairs
        self.engines = {}
        self.analysis_results = {}
        self.all_signals = []
        self.config = ForexEngineConfig()
        self.edge_case_handler = EdgeCaseHandler()
        
        # Initialize engines for each currency pair
        for pair in currency_pairs:
            try:
                engine = AgenticForexEngine(pair)
                self.engines[pair] = engine
                print(f"✅ Initialized engine for {pair}")
            except Exception as e:
                print(f"❌ Failed to initialize engine for {pair}: {e}")
    
    def analyze_single_pair(self, pair: str) -> Dict:
        """Analyze a single currency pair with comprehensive analysis"""
        try:
            engine = self.engines.get(pair)
            if not engine:
                return {'error': f'No engine available for {pair}'}
            
            # Connect to MT5 if not connected
            if not engine.data_collector.is_connected:
                if not engine.data_collector.connect_mt5():
                    return {'error': f'Failed to connect MT5 for {pair}'}
            
            # Run comprehensive analysis
            results = self._run_comprehensive_analysis(engine, pair)
            
            if results:
                # Add pair identifier to results
                results['currency_pair'] = pair
                return results
            else:
                return {'error': f'Analysis failed for {pair}', 'currency_pair': pair}
                
        except Exception as e:
            return {'error': f'Exception analyzing {pair}: {str(e)}', 'currency_pair': pair}
    
    def _run_comprehensive_analysis(self, engine, pair: str) -> Dict:
        """Run comprehensive multi-timeframe analysis for a currency pair"""
        try:
            # Get current price data
            current_price = engine.data_collector.get_current_price()
            if not current_price:
                return {'error': f'Failed to get current price for {pair}'}
            
            # Multi-timeframe analysis
            timeframes = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']
            timeframe_analysis = {}
            
            for tf in timeframes:
                try:
                    # Get market data for timeframe
                    mtf_data = engine.data_collector.get_mtf_data(bars=200)  # Use proper bars parameter
                    df = mtf_data.get(tf)
                    if df is None or len(df) < 50:
                        continue
                    
                    # Smart Money Concepts Analysis
                    smc_analysis = engine.smc_analyzer.detect_market_structure_break(df)
                    order_blocks = engine.smc_analyzer.detect_order_blocks(df)
                    fair_value_gaps = engine.smc_analyzer.detect_fair_value_gaps(df)
                    liquidity_sweeps = engine.smc_analyzer.detect_liquidity_sweeps(df)
                    
                    # Volatility Analysis
                    volatility_analysis = engine.volatility_analyzer.analyze_compression_expansion(df)
                    
                    # Trend Analysis
                    trend_context = engine.trend_analyzer.analyze_trend_context(df)
                    
                    # Session Analysis
                    current_session = engine._get_current_session()
                    session_strength = engine._calculate_session_strength(current_session)
                    
                    timeframe_analysis[tf] = {
                        'smc_analysis': smc_analysis,
                        'order_blocks': order_blocks,
                        'fair_value_gaps': fair_value_gaps,
                        'liquidity_sweeps': liquidity_sweeps,
                        'volatility_analysis': volatility_analysis,
                        'trend_context': {
                            'age_in_bars': trend_context.age_in_bars,
                            'momentum_strength': trend_context.momentum_strength,
                            'phase': trend_context.phase,
                            'expansion_potential': trend_context.expansion_potential
                        },
                        'session_info': {
                            'current_session': current_session.value,
                            'session_strength': session_strength
                        }
                    }
                    
                except Exception as e:
                    print(f"Warning: Failed to analyze {tf} for {pair}: {e}")
                    continue
            
            # Generate trading signals using multiple strategies
            trading_signals = self._generate_comprehensive_signals(engine, timeframe_analysis, current_price)
            
            # Validate signals against edge cases
            validated_signals = self._validate_signals_with_edge_cases(trading_signals, timeframe_analysis, engine)
            
            # Market narrative and sentiment
            market_narrative = self._generate_market_narrative(pair, timeframe_analysis, validated_signals)
            
            # Risk assessment
            risk_assessment = self._assess_market_risk(timeframe_analysis, current_price)
            
            # Compile comprehensive results
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'currency_pair': pair,
                'current_price': current_price,
                'timeframe_analysis': timeframe_analysis,
                'trading_signals': validated_signals,
                'raw_signals': trading_signals,  # Keep original signals for analysis
                'market_narrative': market_narrative,
                'risk_assessment': risk_assessment,
                'analysis_quality': self._calculate_analysis_quality(timeframe_analysis)
            }
            
            return analysis_results
            
        except Exception as e:
            return {'error': f'Comprehensive analysis failed for {pair}: {str(e)}'}
    
    def _generate_comprehensive_signals(self, engine, timeframe_analysis: Dict, current_price: Dict) -> List:
        """Generate trading signals using multiple strategies"""
        signals = []
        
        try:
            # Get primary timeframe data (M15 for signal generation)
            primary_tf = 'M15'
            if primary_tf not in timeframe_analysis:
                primary_tf = list(timeframe_analysis.keys())[0] if timeframe_analysis else None
            
            if not primary_tf:
                return signals
            
            tf_data = timeframe_analysis[primary_tf]
            
            # Generate SMC-based signals
            smc_signals = self._generate_smc_signals(engine, tf_data, current_price)
            signals.extend(smc_signals)
            
            # Generate breakout signals
            breakout_signals = self._generate_breakout_signals(engine, tf_data, current_price)
            signals.extend(breakout_signals)
            
            # Generate volatility-based signals
            volatility_signals = self._generate_volatility_signals(engine, tf_data, current_price)
            signals.extend(volatility_signals)
            
            # Try simplified signal generation as fallback if no signals generated
            if not signals and hasattr(engine, '_generate_simplified_trading_signals'):
                try:
                    # Create analysis structure for simplified method
                    analysis_for_simplified = {
                        'timeframe_analysis': timeframe_analysis,
                        'current_price': current_price
                    }
                    simplified_signals = engine._generate_simplified_trading_signals(analysis_for_simplified)
                    
                    # Convert dict signals to TradingSignal objects
                    for sig_dict in simplified_signals:
                        if isinstance(sig_dict, dict):
                            signal = TradingSignal(
                                timestamp=sig_dict.get('timestamp', datetime.now()),
                                signal_type=sig_dict.get('signal_type', 'SIMPLIFIED'),
                                direction=sig_dict.get('direction', 'BUY'),
                                strength=SignalStrength.MODERATE,
                                entry_price=sig_dict.get('entry_price', current_price.get('ask', 0)),
                                stop_loss=sig_dict.get('stop_loss', 0),
                                take_profit=sig_dict.get('take_profit', 0),
                                risk_reward=sig_dict.get('risk_reward', 1.0),
                                confidence=sig_dict.get('confidence', 0.5),
                                reasoning=sig_dict.get('reason', 'Simplified signal generation'),
                                session=engine._get_current_session(),
                                market_structure=engine._determine_market_structure(),
                                symbol=engine.symbol
                            )
                            signal.currency_pair = engine.symbol
                            signals.append(signal)
                    
                    if simplified_signals:
                        print(f"✅ Generated {len(simplified_signals)} simplified signals for {engine.symbol}")
                        
                except Exception as e:
                    print(f"Warning: Simplified signal generation failed for {engine.symbol}: {e}")
            
            # Filter and rank signals
            filtered_signals = self._filter_and_rank_signals(signals, timeframe_analysis)
            
            return signals
            
        except Exception as e:
            print(f"Warning: Signal generation failed: {e}")
            return signals
    
    def _validate_signals_with_edge_cases(self, signals: List, timeframe_analysis: Dict, engine) -> List:
        """Validate signals against edge case scenarios"""
        validated_signals = []
        
        try:
            for signal in signals:
                # Prepare context for edge case validation
                context = {
                    'signal': signal,
                    'timeframe_analysis': timeframe_analysis,
                    'engine': engine,
                    'current_session': engine._get_current_session(),
                    'symbol': engine.symbol
                }
                
                # Prepare signal data and market data for validation
                signal_data = {
                    'timestamp': signal.timestamp,
                    'signal_type': signal.signal_type,
                    'direction': signal.direction,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'confidence': signal.confidence
                }
                
                # Extract market data from timeframe analysis
                market_data = timeframe_analysis.get('M15', {})
                
                # Validate signal against all edge cases
                validation_result = self.edge_case_handler.validate_signal(signal_data, market_data, context)
                
                if validation_result.is_valid:
                    # Signal passed all edge case checks
                    validated_signals.append(signal)
                    if validation_result.warnings:
                        print(f"⚠️  {engine.symbol} signal warning: {', '.join(validation_result.warnings)}")
                else:
                    # Signal failed edge case validation
                    print(f"❌ {engine.symbol} signal rejected: {validation_result.reasoning}")
                    print(f"   Action: {validation_result.action}")
            
            return validated_signals
            
        except Exception as e:
            print(f"Warning: Edge case validation failed: {e}")
            return signals  # Return original signals if validation fails
    
    def _generate_smc_signals(self, engine, tf_data: Dict, current_price: Dict) -> List:
        """Generate Smart Money Concept based signals"""
        signals = []
        
        try:
            order_blocks = tf_data.get('order_blocks', [])
            fair_value_gaps = tf_data.get('fair_value_gaps', [])
            structure_breaks = tf_data.get('smc_analysis', {}).get('structure_breaks', [])
            
            # Ensure order_blocks and fair_value_gaps are lists
            if not isinstance(order_blocks, list):
                print(f"Warning: order_blocks is not a list, got {type(order_blocks)}: {order_blocks}")
                order_blocks = []
            
            # Handle fair_value_gaps dictionary structure
            if isinstance(fair_value_gaps, dict):
                # Extract the actual FVG list from the dictionary structure
                fair_value_gaps = fair_value_gaps.get('all_fvgs', [])
                if not isinstance(fair_value_gaps, list):
                    fair_value_gaps = []
            elif not isinstance(fair_value_gaps, list):
                print(f"Warning: fair_value_gaps is not a list, got {type(fair_value_gaps)}: {fair_value_gaps}")
                fair_value_gaps = []
                
            if not isinstance(structure_breaks, list):
                print(f"Warning: structure_breaks is not a list, got {type(structure_breaks)}: {structure_breaks}")
                structure_breaks = []
            
            current_bid = current_price.get('bid', 0)
            
            # Signal from order blocks
            # Safely get last 3 order blocks
            recent_obs = order_blocks[-3:] if len(order_blocks) >= 3 else order_blocks
            for ob in recent_obs:
                if ob['type'] == 'Bullish_OB' and ob['low'] <= current_bid <= ob['high']:
                    signal = self._create_smc_signal(engine, 'BUY', ob, current_price, 'Order Block Support')
                    if signal:
                        signals.append(signal)
                elif ob['type'] == 'Bearish_OB' and ob['low'] <= current_bid <= ob['high']:
                    signal = self._create_smc_signal(engine, 'SELL', ob, current_price, 'Order Block Resistance')
                    if signal:
                        signals.append(signal)
            
            # Signal from fair value gaps
            # Safely get last 5 FVGs
            recent_fvgs = fair_value_gaps[-5:] if len(fair_value_gaps) >= 5 else fair_value_gaps
            for fvg in recent_fvgs:
                if (fvg['type'] == 'Bullish_FVG' and 
                    fvg['bottom'] <= current_bid <= fvg['top'] and 
                    fvg.get('dynamic_weight', 0) > 0.2):
                    signal = self._create_fvg_signal(engine, 'BUY', fvg, current_price, 'Fair Value Gap Fill')
                    if signal:
                        signals.append(signal)
                elif (fvg['type'] == 'Bearish_FVG' and 
                      fvg['bottom'] <= current_bid <= fvg['top'] and 
                      fvg.get('dynamic_weight', 0) > 0.2):
                    signal = self._create_fvg_signal(engine, 'SELL', fvg, current_price, 'Fair Value Gap Fill')
                    if signal:
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            print(f"Warning: SMC signal generation failed: {e}")
            return signals
    
    def _create_smc_signal(self, engine, direction: str, order_block: Dict, current_price: Dict, reasoning: str):
        """Create SMC-based trading signal"""
        try:
            entry_price = current_price['bid'] if direction == 'SELL' else current_price['ask']
            
            # Calculate stop loss and take profit using engine's pip calculations with wider stops
            # Use dynamic stop loss based on symbol volatility
            base_sl_pips = self._get_dynamic_stop_loss_pips(engine.symbol)
            base_tp_pips = base_sl_pips * 2  # 1:2 risk reward ratio
            
            if direction == 'BUY':
                stop_loss = order_block['low'] - engine.data_collector.pips_to_price(base_sl_pips)
                take_profit = entry_price + engine.data_collector.pips_to_price(base_tp_pips)
            else:
                stop_loss = order_block['high'] + engine.data_collector.pips_to_price(base_sl_pips)
                take_profit = entry_price - engine.data_collector.pips_to_price(base_tp_pips)
            
            # Calculate risk/reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            # Calculate confidence based on order block strength
            confidence = min(order_block.get('strength', 0.5) * 0.8, 0.95)
            
            signal = TradingSignal(
                timestamp=datetime.now(),
                signal_type='SMC_OrderBlock',
                direction=direction,
                strength=SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=risk_reward,
                confidence=confidence,
                reasoning=reasoning,
                session=engine._get_current_session(),
                market_structure=engine._determine_market_structure(),
                symbol=engine.symbol
            )
            
            # Add currency pair to signal
            signal.currency_pair = engine.symbol
            
            return signal
            
        except Exception as e:
            print(f"Warning: Failed to create SMC signal: {e}")
            return None
    
    def _create_fvg_signal(self, engine, direction: str, fvg: Dict, current_price: Dict, reasoning: str):
        """Create FVG-based trading signal"""
        try:
            entry_price = current_price['bid'] if direction == 'SELL' else current_price['ask']
            
            # Calculate stop loss and take profit with dynamic distances
            base_sl_pips = self._get_dynamic_stop_loss_pips(engine.symbol)
            base_tp_pips = int(base_sl_pips * 1.8)  # Slightly lower RR for FVG signals
            
            if direction == 'BUY':
                stop_loss = fvg['bottom'] - engine.data_collector.pips_to_price(base_sl_pips)
                take_profit = entry_price + engine.data_collector.pips_to_price(base_tp_pips)
            else:
                stop_loss = fvg['top'] + engine.data_collector.pips_to_price(base_sl_pips)
                take_profit = entry_price - engine.data_collector.pips_to_price(base_tp_pips)
            
            # Calculate risk/reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            # Calculate confidence based on FVG dynamic weight
            confidence = min(fvg.get('dynamic_weight', 0.2) * 2.5, 0.85)
            
            signal = TradingSignal(
                timestamp=datetime.now(),
                signal_type='SMC_FairValueGap',
                direction=direction,
                strength=SignalStrength.MODERATE if confidence > 0.6 else SignalStrength.WEAK,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=risk_reward,
                confidence=confidence,
                reasoning=reasoning,
                session=engine._get_current_session(),
                market_structure=engine._determine_market_structure(),
                symbol=engine.symbol
            )
            
            # Add currency pair to signal
            signal.currency_pair = engine.symbol
            
            return signal
            
        except Exception as e:
            print(f"Warning: Failed to create FVG signal: {e}")
            return None
    
    def _generate_breakout_signals(self, engine, tf_data: Dict, current_price: Dict) -> List:
        """Generate breakout-based signals"""
        signals = []
        
        try:
            structure_breaks = tf_data.get('smc_analysis', {}).get('structure_breaks', [])
            volatility_data = tf_data.get('volatility_analysis', {})
            
            # Ensure structure_breaks is a list
            if not isinstance(structure_breaks, list):
                print(f"Warning: structure_breaks is not a list, got {type(structure_breaks)}: {structure_breaks}")
                structure_breaks = []
            
            # Check for recent structure breaks indicating potential breakouts
            # Safely get last 2 structure breaks
            recent_sbs = structure_breaks[-2:] if len(structure_breaks) >= 2 else structure_breaks
            for sb in recent_sbs:
                if sb.get('type') in ['Bullish_MSB', 'Bearish_MSB']:
                    # Check if volatility supports the breakout
                    if volatility_data.get('expansion_phase', False):
                        direction = 'BUY' if sb['type'] == 'Bullish_MSB' else 'SELL'
                        signal = self._create_breakout_signal(engine, direction, sb, current_price, 'Structure Break Breakout')
                        if signal:
                            signals.append(signal)
            
            return signals
            
        except Exception as e:
            print(f"Warning: Breakout signal generation failed: {e}")
            return signals
    
    def _generate_volatility_signals(self, engine, tf_data: Dict, current_price: Dict) -> List:
        """Generate volatility-based signals"""
        signals = []
        
        try:
            volatility_data = tf_data.get('volatility_analysis', {})
            trend_context = tf_data.get('trend_context', {})
            
            # Check for volatility compression followed by expansion
            if (volatility_data.get('compression_detected', False) and 
                volatility_data.get('expansion_phase', False)):
                
                # Determine direction based on trend context
                if trend_context.get('phase') == 'TRENDING':
                    direction = 'BUY' if trend_context.get('momentum_strength', 0) > 0 else 'SELL'
                    signal = self._create_volatility_signal(engine, direction, volatility_data, current_price, 'Volatility Expansion')
                    if signal:
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            print(f"Warning: Volatility signal generation failed: {e}")
            return signals
    
    def _create_breakout_signal(self, engine, direction: str, structure_break: Dict, current_price: Dict, reasoning: str):
        """Create breakout-based trading signal"""
        try:
            entry_price = current_price['bid'] if direction == 'SELL' else current_price['ask']
            
            # Calculate stop loss and take profit with dynamic distances
            base_sl_pips = self._get_dynamic_stop_loss_pips(engine.symbol)
            base_tp_pips = int(base_sl_pips * 2.2)  # Higher RR for breakout signals
            
            if direction == 'BUY':
                stop_loss = structure_break.get('low', entry_price) - engine.data_collector.pips_to_price(base_sl_pips)
                take_profit = entry_price + engine.data_collector.pips_to_price(base_tp_pips)
            else:
                stop_loss = structure_break.get('high', entry_price) + engine.data_collector.pips_to_price(base_sl_pips)
                take_profit = entry_price - engine.data_collector.pips_to_price(base_tp_pips)
            
            # Calculate risk/reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            # Calculate confidence based on structure break strength
            confidence = min(structure_break.get('strength', 0.4) * 1.2, 0.8)
            
            signal = TradingSignal(
                timestamp=datetime.now(),
                signal_type='Breakout',
                direction=direction,
                strength=SignalStrength.STRONG if confidence > 0.6 else SignalStrength.MODERATE,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=risk_reward,
                confidence=confidence,
                reasoning=reasoning,
                session=engine._get_current_session(),
                market_structure=engine._determine_market_structure(),
                symbol=engine.symbol
            )
            
            signal.currency_pair = engine.symbol
            return signal
            
        except Exception as e:
            print(f"Warning: Failed to create breakout signal: {e}")
            return None
    
    def _create_volatility_signal(self, engine, direction: str, volatility_data: Dict, current_price: Dict, reasoning: str):
        """Create volatility-based trading signal"""
        try:
            entry_price = current_price['bid'] if direction == 'SELL' else current_price['ask']
            
            # Calculate stop loss and take profit based on volatility with dynamic distances
            base_sl_pips = self._get_dynamic_stop_loss_pips(engine.symbol)
            volatility_factor = volatility_data.get('expansion_strength', 1.0)
            adjusted_sl_pips = int(base_sl_pips * max(0.8, volatility_factor))  # Adjust based on volatility
            adjusted_tp_pips = int(adjusted_sl_pips * 2.0)  # 1:2 risk reward
            
            if direction == 'BUY':
                stop_loss = entry_price - engine.data_collector.pips_to_price(adjusted_sl_pips)
                take_profit = entry_price + engine.data_collector.pips_to_price(adjusted_tp_pips)
            else:
                stop_loss = entry_price + engine.data_collector.pips_to_price(adjusted_sl_pips)
                take_profit = entry_price - engine.data_collector.pips_to_price(adjusted_tp_pips)
            
            # Calculate risk/reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            # Calculate confidence based on volatility strength
            confidence = min(volatility_factor * 0.6, 0.75)
            
            signal = TradingSignal(
                timestamp=datetime.now(),
                signal_type='Volatility',
                direction=direction,
                strength=SignalStrength.MODERATE,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=risk_reward,
                confidence=confidence,
                reasoning=reasoning,
                session=engine._get_current_session(),
                market_structure=engine._determine_market_structure(),
                symbol=engine.symbol
            )
            
            signal.currency_pair = engine.symbol
            return signal
            
        except Exception as e:
            print(f"Warning: Failed to create volatility signal: {e}")
            return None
    
    def _filter_and_rank_signals(self, signals: List, timeframe_analysis: Dict) -> List:
        """Filter and rank signals based on quality and confluence"""
        try:
            if not signals:
                return signals
            
            # Filter signals by minimum requirements
            filtered_signals = []
            for signal in signals:
                if (hasattr(signal, 'risk_reward') and signal.risk_reward >= self.config.trading.min_risk_reward and
                    hasattr(signal, 'confidence') and signal.confidence >= self.config.trading.min_confidence):
                    filtered_signals.append(signal)
            
            # Sort by confidence and risk/reward
            filtered_signals.sort(key=lambda s: (s.confidence * s.risk_reward), reverse=True)
            
            # Return top 3 signals
            return filtered_signals[:3]
            
        except Exception as e:
            print(f"Warning: Signal filtering failed: {e}")
            return signals
    
    def _generate_market_narrative(self, pair: str, timeframe_analysis: Dict, trading_signals: List) -> str:
        """Generate market narrative and sentiment analysis"""
        try:
            narrative_parts = []
            
            # Overall market structure
            if timeframe_analysis:
                primary_tf = list(timeframe_analysis.keys())[0]
                tf_data = timeframe_analysis[primary_tf]
                
                trend_context = tf_data.get('trend_context', {})
                session_info = tf_data.get('session_info', {})
                
                narrative_parts.append(f"{pair} is currently in {trend_context.get('phase', 'UNKNOWN')} phase")
                narrative_parts.append(f"during {session_info.get('current_session', 'UNKNOWN')} session")
                
                # Volatility context
                volatility_data = tf_data.get('volatility_analysis', {})
                if volatility_data.get('compression_detected'):
                    narrative_parts.append("with volatility compression detected")
                elif volatility_data.get('expansion_phase'):
                    narrative_parts.append("with volatility expansion in progress")
            
            # Signal summary
            if trading_signals:
                signal_count = len(trading_signals)
                avg_confidence = sum(s.confidence for s in trading_signals if hasattr(s, 'confidence')) / signal_count
                narrative_parts.append(f"Generated {signal_count} trading signals with average confidence {avg_confidence:.2f}")
            else:
                narrative_parts.append("No trading signals generated at current market conditions")
            
            return ". ".join(narrative_parts) + "."
            
        except Exception as e:
            return f"Market narrative generation failed for {pair}: {str(e)}"
    
    def _assess_market_risk(self, timeframe_analysis: Dict, current_price: Dict) -> Dict:
        """Assess current market risk levels"""
        try:
            risk_assessment = {
                'overall_risk': 'MEDIUM',
                'volatility_risk': 'MEDIUM',
                'liquidity_risk': 'LOW',
                'session_risk': 'MEDIUM',
                'risk_score': 0.5
            }
            
            if timeframe_analysis:
                # Assess volatility risk
                primary_tf = list(timeframe_analysis.keys())[0]
                tf_data = timeframe_analysis[primary_tf]
                
                volatility_data = tf_data.get('volatility_analysis', {})
                if volatility_data.get('expansion_phase'):
                    risk_assessment['volatility_risk'] = 'HIGH'
                    risk_assessment['risk_score'] += 0.2
                elif volatility_data.get('compression_detected'):
                    risk_assessment['volatility_risk'] = 'LOW'
                    risk_assessment['risk_score'] -= 0.1
                
                # Assess session risk
                session_info = tf_data.get('session_info', {})
                session_strength = session_info.get('session_strength', 0.5)
                if session_strength > 0.7:
                    risk_assessment['session_risk'] = 'LOW'
                    risk_assessment['risk_score'] -= 0.1
                elif session_strength < 0.3:
                    risk_assessment['session_risk'] = 'HIGH'
                    risk_assessment['risk_score'] += 0.2
                
                # Overall risk calculation
                if risk_assessment['risk_score'] > 0.7:
                    risk_assessment['overall_risk'] = 'HIGH'
                elif risk_assessment['risk_score'] < 0.3:
                    risk_assessment['overall_risk'] = 'LOW'
            
            return risk_assessment
            
        except Exception as e:
            return {'error': f'Risk assessment failed: {str(e)}'}
    
    def _calculate_analysis_quality(self, timeframe_analysis: Dict) -> Dict:
        """Calculate the quality of the analysis"""
        try:
            quality_metrics = {
                'data_completeness': 0.0,
                'signal_reliability': 0.0,
                'timeframe_coverage': 0.0,
                'overall_quality': 0.0
            }
            
            if not timeframe_analysis:
                return quality_metrics
            
            # Data completeness
            total_timeframes = 6  # M1, M5, M15, H1, H4, D1
            available_timeframes = len(timeframe_analysis)
            quality_metrics['data_completeness'] = available_timeframes / total_timeframes
            
            # Timeframe coverage
            quality_metrics['timeframe_coverage'] = quality_metrics['data_completeness']
            
            # Signal reliability (based on available analysis components)
            reliability_score = 0.0
            for tf_data in timeframe_analysis.values():
                if tf_data.get('smc_analysis'):
                    reliability_score += 0.3
                if tf_data.get('order_blocks'):
                    reliability_score += 0.2
                if tf_data.get('fair_value_gaps'):
                    reliability_score += 0.2
                if tf_data.get('volatility_analysis'):
                    reliability_score += 0.2
                if tf_data.get('trend_context'):
                    reliability_score += 0.1
                break  # Only check first timeframe
            
            quality_metrics['signal_reliability'] = min(reliability_score, 1.0)
            
            # Overall quality
            quality_metrics['overall_quality'] = (
                quality_metrics['data_completeness'] * 0.4 +
                quality_metrics['signal_reliability'] * 0.4 +
                quality_metrics['timeframe_coverage'] * 0.2
            )
            
            return quality_metrics
            
        except Exception as e:
            return {'error': f'Quality calculation failed: {str(e)}'}
    
    def run_parallel_analysis(self, max_workers: int = 5) -> Dict:
        """Run analysis on all currency pairs in parallel"""
        print(f"🔄 Starting parallel analysis for {len(self.currency_pairs)} currency pairs...")
        start_time = time.time()
        
        all_results = {}
        all_signals = []
        successful_analyses = 0
        failed_analyses = 0
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            future_to_pair = {
                executor.submit(self.analyze_single_pair, pair): pair 
                for pair in self.currency_pairs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout per pair
                    
                    if 'error' in result:
                        print(f"⚠️  {pair}: {result['error']}")
                        failed_analyses += 1
                    else:
                        print(f"✅ {pair}: Analysis complete")
                        successful_analyses += 1
                        
                        # Extract signals
                        signals = result.get('trading_signals', [])
                        if signals:
                            print(f"   📈 {len(signals)} signals found")
                            all_signals.extend(signals)
                        else:
                            print(f"   📊 No signals")
                    
                    all_results[pair] = result
                    
                except Exception as e:
                    print(f"❌ {pair}: Exception - {str(e)}")
                    failed_analyses += 1
                    all_results[pair] = {'error': str(e), 'currency_pair': pair}
        
        end_time = time.time()
        analysis_duration = end_time - start_time
        
        # Compile summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'analysis_duration_seconds': round(analysis_duration, 2),
            'total_pairs': len(self.currency_pairs),
            'successful_analyses': successful_analyses,
            'failed_analyses': failed_analyses,
            'total_signals': len(all_signals),
            'currency_pairs_analyzed': self.currency_pairs,
            'pair_results': all_results,
            'all_signals': all_signals
        }
        
        # Store results
        self.analysis_results = summary
        self.all_signals = all_signals
        
        return summary
    
    def get_top_signals(self, min_confidence: float = 0.7, max_signals: int = 10) -> List[Dict]:
        """Get top trading signals across all pairs"""
        if not self.all_signals:
            return []
        
        # Filter and sort signals
        filtered_signals = [
            signal for signal in self.all_signals 
            if signal.confidence >= min_confidence
        ]
        
        # Sort by confidence and strength
        strength_order = {
            SignalStrength.VERY_STRONG: 4,
            SignalStrength.STRONG: 3,
            SignalStrength.MODERATE: 2,
            SignalStrength.WEAK: 1
        }
        
        sorted_signals = sorted(
            filtered_signals,
            key=lambda s: (strength_order.get(s.strength, 0), s.confidence),
            reverse=True
        )
        
        return sorted_signals[:max_signals]
    
    def generate_multi_pair_report(self) -> str:
        """Generate a comprehensive multi-pair analysis report"""
        if not self.analysis_results:
            return "No analysis results available. Run analysis first."
        
        results = self.analysis_results
        
        report = f"""
# 🌍 BLOB AI Multi-Currency Pair Analysis Report

## 📊 Analysis Summary
- **Timestamp**: {results['timestamp']}
- **Analysis Duration**: {results['analysis_duration_seconds']} seconds
- **Total Pairs**: {results['total_pairs']}
- **Successful Analyses**: {results['successful_analyses']}
- **Failed Analyses**: {results['failed_analyses']}
- **Total Signals Generated**: {results['total_signals']}

## 🎯 Top Trading Opportunities
"""
        
        # Get top signals
        top_signals = self.get_top_signals(min_confidence=0.6, max_signals=5)
        
        if top_signals:
            for i, signal in enumerate(top_signals, 1):
                report += f"""
### {i}. {signal.direction} {signal.signal_type}
- **Currency Pair**: {getattr(signal, 'currency_pair', 'Unknown')}
- **Entry Price**: {signal.entry_price:.5f}
- **Stop Loss**: {signal.stop_loss:.5f}
- **Take Profit**: {signal.take_profit:.5f}
- **Risk/Reward**: 1:{signal.risk_reward:.1f}
- **Confidence**: {signal.confidence:.1%}
- **Strength**: {signal.strength.value}
- **Reasoning**: {signal.reasoning}
"""
        else:
            report += "\n### No high-confidence signals found across all pairs\n"
        
        # Pair-by-pair summary
        report += "\n## 📈 Pair-by-Pair Summary\n"
        
        for pair in self.currency_pairs:
            pair_result = results['pair_results'].get(pair, {})
            
            if 'error' in pair_result:
                report += f"\n### {pair} ❌\n- **Status**: Failed\n- **Error**: {pair_result['error']}\n"
            else:
                signals = pair_result.get('trading_signals', [])
                current_price = pair_result.get('current_price', {})
                
                report += f"\n### {pair} ✅\n"
                
                if current_price:
                    spread_pips = "Unknown"
                    if 'JPY' in pair:
                        spread_pips = f"{current_price.get('spread', 0) * 100:.1f}"
                    else:
                        spread_pips = f"{current_price.get('spread', 0) * 10000:.1f}"
                    
                    report += f"- **Current Price**: {current_price.get('bid', 'N/A'):.5f} / {current_price.get('ask', 'N/A'):.5f}\n"
                    report += f"- **Spread**: {spread_pips} pips\n"
                
                report += f"- **Signals Found**: {len(signals)}\n"
                
                if signals:
                    strong_signals = [s for s in signals if s.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]]
                    report += f"- **High-Quality Signals**: {len(strong_signals)}\n"
        
        report += "\n---\n*Generated by BLOB AI Multi-Pair Analysis Engine*\n"
        
        return report
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save analysis results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_pair_analysis_{timestamp}.json"
        
        # Make results JSON serializable
        serializable_results = self._make_json_serializable(self.analysis_results)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            print(f"📁 Results saved to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return ""
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, TradingSignal):
            signal_dict = asdict(obj)
            # Add currency pair if available
            if hasattr(obj, 'currency_pair'):
                signal_dict['currency_pair'] = obj.currency_pair
            return signal_dict
        elif hasattr(obj, 'value'):  # Enum objects
            return obj.value
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def _get_dynamic_stop_loss_pips(self, symbol: str) -> int:
        """Calculate dynamic stop loss distance based on symbol characteristics"""
        try:
            # Base stop loss distances by currency pair type
            if 'JPY' in symbol:
                # JPY pairs typically have larger pip values, use smaller distances
                base_pips = 35
            elif any(major in symbol for major in ['EUR', 'GBP', 'USD']):
                # Major pairs - moderate volatility
                base_pips = 45
            elif any(minor in symbol for minor in ['AUD', 'NZD', 'CAD']):
                # Commodity currencies - higher volatility
                base_pips = 55
            elif 'CHF' in symbol:
                # Swiss Franc - lower volatility
                base_pips = 40
            else:
                # Default for exotic pairs
                base_pips = 60
            
            # Adjust based on specific pair characteristics
            volatility_adjustments = {
                'GBPJPY': 65,  # Very volatile
                'EURJPY': 50,  # Moderate volatility
                'USDJPY': 45,  # Lower volatility
                'GBPUSD': 50,  # Cable volatility
                'EURUSD': 40,  # Most liquid, lower spreads
                'USDCHF': 45,  # Moderate volatility
                'AUDUSD': 50,  # Commodity currency
                'NZDUSD': 55,  # Higher volatility commodity
                'GBPAUD': 65,  # Cross pair, higher volatility
                'EURNZD': 60,  # Cross pair volatility
                'AUDCAD': 55,  # Commodity cross
                'CHFJPY': 50,  # Safe haven cross
            }
            
            # Use specific adjustment if available, otherwise use base calculation
            final_pips = volatility_adjustments.get(symbol, base_pips)
            
            # Ensure minimum stop loss distance
            return max(final_pips, 30)
            
        except Exception as e:
            print(f"Warning: Error calculating dynamic stop loss for {symbol}: {e}")
            return 45  # Safe default
    
    def disconnect_all(self):
        """Disconnect all MT5 connections"""
        for pair, engine in self.engines.items():
            try:
                engine.data_collector.disconnect_mt5()
            except Exception as e:
                print(f"Warning: Error disconnecting {pair}: {e}")

def main(save_reports=False, run_continuously=True, interval_minutes=15):
    """Main execution function for multi-pair analysis
    
    Args:
        save_reports (bool): Whether to save reports to files
        run_continuously (bool): Whether to run continuously every interval
        interval_minutes (int): Minutes between analysis runs
    """
    
    # Define currency pairs to analyze
    currency_pairs = [
        'EURJPY', 'GBPJPY', 'USDJPY', 'CHFJPY', 'EURAUD', 'AUDJPY', 'EURNZD', 
        'GBPNZD', 'NZDUSD', 'GBPUSD', 'NZDJPY', 'GBPCHF', 'GBPAUD', 'EURCAD', 
        'AUDUSD', 'USDCHF', 'EURCHF', 'AUDCAD'
    ]
    
    print("🚀 BLOB AI Multi-Currency Pair Analysis Engine")
    print("=" * 60)
    print(f"📊 Analyzing {len(currency_pairs)} currency pairs:")
    for i, pair in enumerate(currency_pairs, 1):
        print(f"   {i:2d}. {pair}")
    
    if run_continuously:
        print(f"\n⏰ Running continuously every {interval_minutes} minutes")
        print(f"💾 Report saving: {'Enabled' if save_reports else 'Disabled'}")
    print()
    
    # Initialize multi-pair engine
    try:
        multi_engine = MultiPairAnalysisEngine(currency_pairs)
        
        if not multi_engine.engines:
            print("❌ No engines initialized successfully. Exiting.")
            return
        
        print(f"✅ Initialized {len(multi_engine.engines)} engines successfully\n")
        
        run_count = 0
        
        while True:
            run_count += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"🔄 Starting analysis run #{run_count} at {current_time}...")
            
            # Run parallel analysis
            results = multi_engine.run_parallel_analysis(max_workers=6)
            
            print("\n" + "=" * 60)
            print(f"📊 ANALYSIS RUN #{run_count} COMPLETE")
            print("=" * 60)
            
            # Display summary
            print(f"⏱️  Duration: {results['analysis_duration_seconds']} seconds")
            print(f"✅ Successful: {results['successful_analyses']}/{results['total_pairs']} pairs")
            print(f"📈 Total Signals: {results['total_signals']}")
            
            # Show top signals
            top_signals = multi_engine.get_top_signals(min_confidence=0.6, max_signals=5)
            if top_signals:
                print(f"\n🎯 TOP {len(top_signals)} TRADING OPPORTUNITIES:")
                print("-" * 50)
                
                for i, signal in enumerate(top_signals, 1):
                    pair = getattr(signal, 'currency_pair', 'Unknown')
                    print(f"{i}. {pair} - {signal.direction} {signal.signal_type}")
                    print(f"   Confidence: {signal.confidence:.1%} | R:R 1:{signal.risk_reward:.1f}")
                    print(f"   Entry: {signal.entry_price:.5f} | SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f}")
                    print()
            else:
                print("\n📊 No high-confidence signals found across all pairs")
            
            # Save reports only if enabled
            if save_reports:
                # Generate and save report
                report = multi_engine.generate_multi_pair_report()
                
                # Save results
                results_file = multi_engine.save_results()
                
                # Save report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = f"multi_pair_report_{timestamp}.md"
                try:
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(report)
                    print(f"📄 Report saved to {report_file}")
                except Exception as e:
                    print(f"❌ Error saving report: {e}")
            else:
                print("💾 Report saving disabled - results displayed only")
            
            if not run_continuously:
                break
                
            # Wait for next analysis cycle
            next_run = datetime.now().replace(second=0, microsecond=0)
            next_run = next_run.replace(minute=(next_run.minute // interval_minutes + 1) * interval_minutes)
            if next_run.minute >= 60:
                next_run = next_run.replace(hour=next_run.hour + 1, minute=0)
            
            wait_seconds = (next_run - datetime.now()).total_seconds()
            
            print(f"\n⏳ Next analysis at {next_run.strftime('%H:%M:%S')} (waiting {wait_seconds:.0f} seconds)")
            print("=" * 60)
            
            try:
                time.sleep(wait_seconds)
            except KeyboardInterrupt:
                print("\n\n⏹️  Analysis stopped by user")
                break
        
        print("\n" + "=" * 60)
        print("🎉 Multi-pair analysis complete!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        try:
            multi_engine.disconnect_all()
            print("\n🔌 All connections closed")
        except:
            pass

if __name__ == "__main__":
    main()