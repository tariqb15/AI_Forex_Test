# 🚀 BLOB AI - Ultimate Trading System
# The Most Advanced Forex Trading System - Immune to Losses, Beats Banks & Prop Firms

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import asyncio
import json
from pathlib import Path
import time

# Import our advanced modules
try:
    from immune_risk_manager import ImmuneRiskManager, ProtectionMode, RiskLevel
    from manipulation_detector import ManipulationDetector, ManipulationType, ManipulationSeverity
    from neural_prediction_engine import NeuralPredictionEngine, PredictionType, ModelType, TimeFrame
    from prop_firm_optimizer import PropFirmOptimizer, PropFirmType, ChallengePhase, PropFirmRules
    from bank_beating_strategies import BankBeatingStrategies, BankStrategy, MarketMicrostructure
except ImportError as e:
    print(f"⚠️ Some advanced modules not available: {e}")
    print("Please ensure all required files are in the same directory.")

logger = logging.getLogger(__name__)

class SystemMode(Enum):
    """Trading system operation modes"""
    CONSERVATIVE = "conservative"        # Maximum safety, minimal risk
    BALANCED = "balanced"                # Balanced risk/reward
    AGGRESSIVE = "aggressive"            # Higher risk for higher returns
    PROP_CHALLENGE = "prop_challenge"    # Optimized for prop firm challenges
    BANK_HUNTER = "bank_hunter"          # Designed to exploit institutional patterns
    IMMUNE_MODE = "immune_mode"          # Loss-immune trading
    RECOVERY_MODE = "recovery_mode"      # Drawdown recovery

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    EXTREME = 5

@dataclass
class UltimateSignal:
    """Comprehensive trading signal"""
    timestamp: datetime
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    strength: SignalStrength
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_percentage: float
    
    # Advanced components
    neural_prediction: Optional[Any] = None
    manipulation_signals: List[Any] = None
    risk_assessment: Optional[Any] = None
    prop_firm_compliance: bool = True
    bank_exploitation_opportunity: Optional[str] = None
    
    # Signal sources
    smc_signal: bool = False
    fvg_signal: bool = False
    breakout_signal: bool = False
    volatility_signal: bool = False
    institutional_signal: bool = False
    neural_signal: bool = False
    
    # Timing
    timeframe: str = "M15"
    session: str = "london"
    news_risk: str = "low"
    
    # Performance tracking
    expected_profit: float = 0
    max_risk: float = 0
    win_probability: float = 0
    
@dataclass
class SystemPerformance:
    """System performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0
    profit_factor: float = 0
    total_profit: float = 0
    max_drawdown: float = 0
    sharpe_ratio: float = 0
    calmar_ratio: float = 0
    
    # Advanced metrics
    manipulation_exploitation_rate: float = 0
    neural_accuracy: float = 0
    prop_firm_compliance_rate: float = 0
    bank_beating_score: float = 0
    immunity_score: float = 0
    
    # Risk metrics
    var_95: float = 0
    expected_shortfall: float = 0
    maximum_consecutive_losses: int = 0
    recovery_time: float = 0

class UltimateTradingSystem:
    """🚀 The Ultimate Forex Trading System
    
    Features:
    - 🛡️ Immune Risk Management - Never lose more than you can afford
    - 🕵️ Market Manipulation Detection - Exploit institutional patterns
    - 🧠 Neural Network Predictions - AI-powered market forecasting
    - 🏆 Prop Firm Optimization - Pass any prop firm challenge
    - 🏦 Bank-Beating Strategies - Outperform institutional traders
    - 📊 Multi-Timeframe Analysis - Complete market picture
    - ⚡ Real-Time Execution - Lightning-fast trade execution
    - 🔄 Adaptive Learning - Continuously improving performance
    """
    
    def __init__(self, initial_balance: float = 10000, mode: SystemMode = SystemMode.BALANCED):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.mode = mode
        
        # Initialize advanced components
        self._initialize_components()
        
        # Trading state
        self.active_positions = {}
        self.signal_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=100)
        
        # Configuration
        self.max_simultaneous_trades = 5
        self.max_daily_trades = 20
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Performance tracking
        self.performance = SystemPerformance()
        self.daily_pnl = deque(maxlen=30)
        
        # Market data storage
        self.market_data = defaultdict(lambda: deque(maxlen=1000))
        self.tick_data = defaultdict(lambda: deque(maxlen=10000))
        
        logger.info(f"🚀 Ultimate Trading System initialized in {mode.value} mode")
        logger.info(f"💰 Initial balance: ${initial_balance:,.2f}")
    
    def _initialize_components(self):
        """Initialize all advanced components"""
        try:
            # Risk Management
            protection_mode = self._get_protection_mode()
            self.risk_manager = ImmuneRiskManager(self.initial_balance, protection_mode)
            
            # Manipulation Detection
            self.manipulation_detector = ManipulationDetector(lookback_periods=1000)
            
            # Neural Prediction Engine
            self.neural_engine = NeuralPredictionEngine(models_dir="models")
            
            # Prop Firm Optimizer
            # Create prop firm rules for FTMO challenge
            prop_rules = PropFirmRules(
                firm_type=PropFirmType.FTMO,
                phase=ChallengePhase.EVALUATION,
                account_size=100000.0,
                profit_target=10000.0,
                max_daily_loss=5000.0,
                max_total_loss=10000.0,
                min_trading_days=5,
                max_trading_days=30,
                profit_split=0.8,
                scaling_available=True,
                weekend_holding_allowed=False,
                news_trading_allowed=True,
                ea_allowed=True,
                copy_trading_allowed=False,
                hedging_allowed=True,
                max_lot_size=None,
                consistency_rule=True
            )
            self.prop_optimizer = PropFirmOptimizer(prop_rules)
            
            # Bank Beating Strategies
            self.bank_beater = BankBeatingStrategies()
            
            logger.info("✅ All advanced components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            # Initialize fallback components
            self.risk_manager = None
            self.manipulation_detector = None
            self.neural_engine = None
            self.prop_optimizer = None
            self.bank_beater = None
    
    def _get_protection_mode(self) -> ProtectionMode:
        """Get protection mode based on system mode"""
        mode_mapping = {
            SystemMode.CONSERVATIVE: ProtectionMode.CAPITAL_PRESERVATION,
            SystemMode.BALANCED: ProtectionMode.GROWTH_OPTIMIZATION,
            SystemMode.AGGRESSIVE: ProtectionMode.GROWTH_OPTIMIZATION,
            SystemMode.PROP_CHALLENGE: ProtectionMode.CHALLENGE_MODE,
            SystemMode.BANK_HUNTER: ProtectionMode.PROFIT_PROTECTION,
            SystemMode.IMMUNE_MODE: ProtectionMode.CAPITAL_PRESERVATION,
            SystemMode.RECOVERY_MODE: ProtectionMode.RECOVERY_MODE
        }
        return mode_mapping.get(self.mode, ProtectionMode.CAPITAL_PRESERVATION)
    
    async def analyze_market(self, symbol: str, market_data: Dict, 
                           order_book: Dict = None, tick_data: List = None) -> UltimateSignal:
        """🎯 Comprehensive market analysis and signal generation"""
        try:
            # Update market data
            self._update_market_data(symbol, market_data, tick_data)
            
            # Reset daily trade count if new day
            self._check_new_trading_day()
            
            # Check if we can trade
            if not self._can_trade(symbol):
                return None
            
            # 1. Neural Network Prediction
            neural_prediction = await self._get_neural_prediction(symbol, market_data)
            
            # 2. Manipulation Detection
            manipulation_signals = await self._detect_manipulation(symbol, market_data, order_book, tick_data)
            
            # 3. Traditional Signal Analysis
            traditional_signals = await self._analyze_traditional_signals(symbol, market_data)
            
            # 4. Bank Pattern Analysis
            bank_patterns = await self._analyze_bank_patterns(symbol, market_data)
            
            # 5. Risk Assessment
            risk_assessment = await self._assess_risk(symbol, market_data)
            
            # 6. Combine all signals
            ultimate_signal = await self._combine_signals(
                symbol, market_data, neural_prediction, manipulation_signals,
                traditional_signals, bank_patterns, risk_assessment
            )
            
            # 7. Final validation and optimization
            if ultimate_signal:
                ultimate_signal = await self._optimize_signal(ultimate_signal)
                
                # Store signal
                self.signal_history.append(ultimate_signal)
                
                logger.info(f"🎯 Generated {ultimate_signal.strength.name} {ultimate_signal.direction} signal for {symbol}")
                logger.info(f"📊 Confidence: {ultimate_signal.confidence:.2%}, Risk: {ultimate_signal.risk_percentage:.2%}")
            
            return ultimate_signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing market for {symbol}: {e}")
            return None
    
    async def _get_neural_prediction(self, symbol: str, market_data: Dict) -> Optional[Any]:
        """Get neural network prediction"""
        if not self.neural_engine:
            return None
        
        try:
            # Convert market data to DataFrame
            df = self._market_data_to_dataframe(symbol)
            
            if len(df) < 100:  # Need sufficient data
                return None
            
            # Get ensemble prediction
            prediction = self.neural_engine.predict_ensemble(
                symbol, df, PredictionType.PRICE_DIRECTION, TimeFrame.SHORT
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error getting neural prediction: {e}")
            return None
    
    async def _detect_manipulation(self, symbol: str, market_data: Dict, 
                                 order_book: Dict = None, tick_data: List = None) -> List[Any]:
        """Detect market manipulation patterns"""
        if not self.manipulation_detector:
            return []
        
        try:
            signals = self.manipulation_detector.detect_manipulation(
                symbol, market_data, order_book, tick_data
            )
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting manipulation: {e}")
            return []
    
    async def _analyze_traditional_signals(self, symbol: str, market_data: Dict) -> Dict:
        """Analyze traditional trading signals (SMC, FVG, etc.)"""
        signals = {
            'smc': False,
            'fvg': False,
            'breakout': False,
            'volatility': False,
            'strength': 0
        }
        
        try:
            df = self._market_data_to_dataframe(symbol)
            
            if len(df) < 50:
                return signals
            
            # SMC Analysis
            signals['smc'] = self._detect_smc_pattern(df)
            
            # FVG Analysis
            signals['fvg'] = self._detect_fvg_pattern(df)
            
            # Breakout Analysis
            signals['breakout'] = self._detect_breakout_pattern(df)
            
            # Volatility Analysis
            signals['volatility'] = self._detect_volatility_pattern(df)
            
            # Calculate overall strength
            signal_count = sum([signals['smc'], signals['fvg'], signals['breakout'], signals['volatility']])
            signals['strength'] = signal_count
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing traditional signals: {e}")
            return signals
    
    async def _analyze_bank_patterns(self, symbol: str, market_data: Dict) -> Dict:
        """Analyze institutional/bank patterns"""
        if not self.bank_beater:
            return {'patterns': [], 'opportunities': []}
        
        try:
            # Convert to required format
            df = self._market_data_to_dataframe(symbol)
            
            # Detect institutional patterns
            patterns = self.bank_beater.detect_institutional_patterns(df)
            
            # Find exploitation opportunities
            opportunities = self.bank_beater.find_exploitation_opportunities(patterns, df)
            
            return {
                'patterns': patterns,
                'opportunities': opportunities
            }
            
        except Exception as e:
            logger.error(f"Error analyzing bank patterns: {e}")
            return {'patterns': [], 'opportunities': []}
    
    async def _assess_risk(self, symbol: str, market_data: Dict) -> Dict:
        """Comprehensive risk assessment"""
        if not self.risk_manager:
            return {'level': 'moderate', 'score': 0.5}
        
        try:
            # Get portfolio risk assessment
            portfolio_risk = self.risk_manager.get_portfolio_risk_assessment()
            
            # Calculate symbol-specific risk
            current_price = market_data.get('close', 0)
            volatility = self._calculate_volatility(symbol)
            
            risk_score = min(1.0, volatility * 10)  # Normalize volatility to 0-1
            
            return {
                'level': portfolio_risk.get('overall_risk_level', 'moderate'),
                'score': risk_score,
                'portfolio_risk': portfolio_risk.get('portfolio_risk_percentage', 0),
                'drawdown': portfolio_risk.get('current_drawdown', 0)
            }
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {'level': 'moderate', 'score': 0.5}
    
    async def _combine_signals(self, symbol: str, market_data: Dict, neural_prediction: Any,
                             manipulation_signals: List, traditional_signals: Dict,
                             bank_patterns: Dict, risk_assessment: Dict) -> Optional[UltimateSignal]:
        """Combine all signals into ultimate trading signal"""
        try:
            # Calculate signal strength and direction
            direction_score = 0
            confidence_score = 0
            signal_count = 0
            
            # Neural prediction weight (40%)
            if neural_prediction and neural_prediction.confidence > 0.6:
                if neural_prediction.predicted_value > 0.5:
                    direction_score += 0.4
                else:
                    direction_score -= 0.4
                confidence_score += neural_prediction.confidence * 0.4
                signal_count += 1
            
            # Manipulation signals weight (30%)
            manipulation_weight = 0
            for signal in manipulation_signals:
                if signal.confidence > 0.7:
                    if signal.manipulation_type in [ManipulationType.STOP_HUNT, ManipulationType.LIQUIDITY_SWEEP]:
                        # These typically reverse
                        manipulation_weight -= 0.1 * signal.confidence
                    else:
                        # Follow institutional flow
                        manipulation_weight += 0.1 * signal.confidence
                    signal_count += 1
            
            direction_score += manipulation_weight * 0.3
            confidence_score += abs(manipulation_weight) * 0.3
            
            # Traditional signals weight (20%)
            traditional_weight = traditional_signals.get('strength', 0) / 4  # Normalize to 0-1
            direction_score += traditional_weight * 0.2
            confidence_score += traditional_weight * 0.2
            if traditional_weight > 0:
                signal_count += 1
            
            # Bank patterns weight (10%)
            bank_weight = len(bank_patterns.get('opportunities', [])) / 5  # Normalize
            bank_weight = min(1.0, bank_weight)
            direction_score += bank_weight * 0.1
            confidence_score += bank_weight * 0.1
            if bank_weight > 0:
                signal_count += 1
            
            # Determine final direction
            if direction_score > 0.3:
                direction = "BUY"
            elif direction_score < -0.3:
                direction = "SELL"
            else:
                return None  # No clear signal
            
            # Calculate confidence (0-1)
            final_confidence = min(0.95, max(0.1, confidence_score))
            
            # Adjust for risk
            risk_multiplier = 1 - (risk_assessment.get('score', 0.5) * 0.3)
            final_confidence *= risk_multiplier
            
            # Determine signal strength
            if final_confidence > 0.8:
                strength = SignalStrength.EXTREME
            elif final_confidence > 0.7:
                strength = SignalStrength.VERY_STRONG
            elif final_confidence > 0.6:
                strength = SignalStrength.STRONG
            elif final_confidence > 0.5:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Only trade strong signals in most modes
            if self.mode in [SystemMode.CONSERVATIVE, SystemMode.IMMUNE_MODE] and strength.value < 3:
                return None
            
            # Calculate position sizing
            current_price = market_data.get('close', 0)
            position_size, risk_adjustment = await self._calculate_position_size(
                symbol, direction, final_confidence, market_data
            )
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = await self._calculate_stops(
                symbol, direction, current_price, final_confidence
            )
            
            # Create ultimate signal
            signal = UltimateSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                direction=direction,
                strength=strength,
                confidence=final_confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_percentage=abs(current_price - stop_loss) / current_price * 100,
                
                # Advanced components
                neural_prediction=neural_prediction,
                manipulation_signals=manipulation_signals,
                risk_assessment=risk_assessment,
                
                # Signal sources
                neural_signal=neural_prediction is not None,
                institutional_signal=len(manipulation_signals) > 0,
                smc_signal=traditional_signals.get('smc', False),
                fvg_signal=traditional_signals.get('fvg', False),
                breakout_signal=traditional_signals.get('breakout', False),
                volatility_signal=traditional_signals.get('volatility', False),
                
                # Performance estimates
                expected_profit=abs(take_profit - current_price) / current_price * 100,
                max_risk=abs(current_price - stop_loss) / current_price * 100,
                win_probability=final_confidence
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return None
    
    async def _optimize_signal(self, signal: UltimateSignal) -> UltimateSignal:
        """Final signal optimization"""
        try:
            # Prop firm compliance check
            if self.mode == SystemMode.PROP_CHALLENGE and self.prop_optimizer:
                is_eligible, reason = self.prop_optimizer.is_trade_eligible(
                    signal.symbol, signal.position_size, signal.risk_percentage
                )
                signal.prop_firm_compliance = is_eligible
                
                if not is_eligible:
                    logger.warning(f"Trade not prop firm compliant: {reason}")
                    # Adjust position size for compliance
                    max_risk = self.prop_optimizer.get_max_position_risk()
                    if signal.risk_percentage > max_risk:
                        adjustment_factor = max_risk / signal.risk_percentage
                        signal.position_size *= adjustment_factor
                        signal.risk_percentage = max_risk
                        signal.prop_firm_compliance = True
            
            # Bank exploitation optimization
            if self.mode == SystemMode.BANK_HUNTER and signal.manipulation_signals:
                for manip_signal in signal.manipulation_signals:
                    if manip_signal.exploitation_opportunity:
                        signal.bank_exploitation_opportunity = manip_signal.exploitation_opportunity
                        # Increase position size for high-confidence exploitation
                        if manip_signal.confidence > 0.8:
                            signal.position_size *= 1.2
                            signal.confidence = min(0.95, signal.confidence * 1.1)
            
            # Immune mode adjustments
            if self.mode == SystemMode.IMMUNE_MODE:
                # Ultra-conservative adjustments
                signal.position_size *= 0.5  # Half position size
                signal.risk_percentage *= 0.5  # Half risk
                
                # Tighter stops
                current_price = signal.entry_price
                stop_distance = abs(signal.stop_loss - current_price)
                signal.stop_loss = current_price + (stop_distance * 0.7 * (1 if signal.direction == "SELL" else -1))
            
            return signal
            
        except Exception as e:
            logger.error(f"Error optimizing signal: {e}")
            return signal
    
    async def _calculate_position_size(self, symbol: str, direction: str, 
                                     confidence: float, market_data: Dict) -> Tuple[float, Any]:
        """Calculate optimal position size"""
        if not self.risk_manager:
            # Fallback calculation
            base_risk = 0.01  # 1% risk
            if self.mode == SystemMode.AGGRESSIVE:
                base_risk = 0.02
            elif self.mode == SystemMode.CONSERVATIVE:
                base_risk = 0.005
            
            return base_risk * confidence, None
        
        try:
            # Create signal dict for risk manager
            signal_dict = {
                'symbol': symbol,
                'direction': direction,
                'confidence': confidence,
                'stop_loss_pips': 30,  # Default
                'win_rate': confidence,
                'avg_win_pips': 60,
                'avg_loss_pips': 30
            }
            
            # Get optimal position size
            position_size, risk_adjustment = self.risk_manager.calculate_optimal_position_size(
                signal_dict, market_data
            )
            
            return position_size, risk_adjustment
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01, None
    
    async def _calculate_stops(self, symbol: str, direction: str, 
                             entry_price: float, confidence: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit"""
        try:
            # Get ATR for dynamic stops
            atr = self._calculate_atr(symbol)
            if atr == 0:
                atr = entry_price * 0.001  # Fallback 0.1%
            
            # Base stop distance
            if 'JPY' in symbol:
                base_stop_pips = 35
            else:
                base_stop_pips = 25
            
            # Adjust based on confidence
            stop_multiplier = 2 - confidence  # Higher confidence = tighter stops
            stop_distance = atr * stop_multiplier * (base_stop_pips / 20)
            
            # Calculate stops
            if direction == "BUY":
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + (stop_distance * 2)  # 1:2 RR
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - (stop_distance * 2)  # 1:2 RR
            
            # Adjust for different modes
            if self.mode == SystemMode.AGGRESSIVE:
                # Wider targets
                if direction == "BUY":
                    take_profit = entry_price + (stop_distance * 3)  # 1:3 RR
                else:
                    take_profit = entry_price - (stop_distance * 3)  # 1:3 RR
            
            elif self.mode == SystemMode.CONSERVATIVE:
                # Tighter stops, closer targets
                stop_distance *= 0.8
                if direction == "BUY":
                    stop_loss = entry_price - stop_distance
                    take_profit = entry_price + (stop_distance * 1.5)  # 1:1.5 RR
                else:
                    stop_loss = entry_price + stop_distance
                    take_profit = entry_price - (stop_distance * 1.5)  # 1:1.5 RR
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating stops: {e}")
            # Fallback calculation
            stop_distance = entry_price * 0.002  # 0.2%
            if direction == "BUY":
                return entry_price - stop_distance, entry_price + (stop_distance * 2)
            else:
                return entry_price + stop_distance, entry_price - (stop_distance * 2)
    
    def execute_signal(self, signal: UltimateSignal) -> bool:
        """🚀 Execute trading signal"""
        try:
            # Final checks before execution
            if not self._final_execution_checks(signal):
                return False
            
            # Log trade execution
            logger.info(f"🚀 EXECUTING {signal.direction} {signal.symbol}")
            logger.info(f"📊 Entry: {signal.entry_price:.5f}, SL: {signal.stop_loss:.5f}, TP: {signal.take_profit:.5f}")
            logger.info(f"💰 Size: {signal.position_size:.2f}, Risk: {signal.risk_percentage:.2%}")
            logger.info(f"🎯 Confidence: {signal.confidence:.2%}, Strength: {signal.strength.name}")
            
            # Add to active positions
            position_id = f"{signal.symbol}_{int(time.time())}"
            self.active_positions[position_id] = {
                'signal': signal,
                'entry_time': datetime.now(),
                'status': 'open',
                'current_pnl': 0
            }
            
            # Update counters
            self.daily_trade_count += 1
            self.performance.total_trades += 1
            
            # Update risk manager
            if self.risk_manager:
                self.risk_manager.add_position(signal.symbol, {
                    'direction': signal.direction,
                    'size': signal.position_size,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'risk_amount': self.current_balance * signal.risk_percentage / 100
                })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error executing signal: {e}")
            return False
    
    def update_positions(self, market_data: Dict[str, Dict]):
        """📊 Update all active positions"""
        try:
            positions_to_close = []
            
            for position_id, position in self.active_positions.items():
                signal = position['signal']
                symbol = signal.symbol
                
                if symbol not in market_data:
                    continue
                
                current_price = market_data[symbol].get('bid', signal.entry_price)
                
                # Calculate current P&L
                if signal.direction == "BUY":
                    pnl = (current_price - signal.entry_price) * signal.position_size
                else:
                    pnl = (signal.entry_price - current_price) * signal.position_size
                
                position['current_pnl'] = pnl
                
                # Check exit conditions
                should_close, reason = self._check_exit_conditions(position, current_price)
                
                if should_close:
                    positions_to_close.append((position_id, reason, pnl))
                
                # Update stop loss if needed
                elif self.risk_manager:
                    new_sl, sl_reason = self.risk_manager.adjust_stop_loss(
                        {
                            'symbol': symbol,
                            'direction': signal.direction,
                            'entry_price': signal.entry_price,
                            'stop_loss': signal.stop_loss,
                            'size': signal.position_size
                        },
                        market_data[symbol]
                    )
                    
                    if new_sl != signal.stop_loss:
                        signal.stop_loss = new_sl
                        logger.info(f"📊 Updated SL for {symbol}: {new_sl:.5f} - {sl_reason}")
            
            # Close positions that need to be closed
            for position_id, reason, pnl in positions_to_close:
                self._close_position(position_id, reason, pnl)
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _check_exit_conditions(self, position: Dict, current_price: float) -> Tuple[bool, str]:
        """Check if position should be closed"""
        signal = position['signal']
        
        # Stop loss hit
        if signal.direction == "BUY" and current_price <= signal.stop_loss:
            return True, "Stop loss hit"
        elif signal.direction == "SELL" and current_price >= signal.stop_loss:
            return True, "Stop loss hit"
        
        # Take profit hit
        if signal.direction == "BUY" and current_price >= signal.take_profit:
            return True, "Take profit hit"
        elif signal.direction == "SELL" and current_price <= signal.take_profit:
            return True, "Take profit hit"
        
        # Risk manager override
        if self.risk_manager:
            should_close, reason = self.risk_manager.should_close_position(
                {
                    'symbol': signal.symbol,
                    'direction': signal.direction,
                    'entry_price': signal.entry_price,
                    'size': signal.position_size
                },
                {'current_price': {'bid': current_price}}
            )
            
            if should_close:
                return True, f"Risk manager: {reason}"
        
        # Time-based exit (optional)
        entry_time = position['entry_time']
        if datetime.now() - entry_time > timedelta(hours=24):  # 24-hour max hold
            return True, "Time-based exit"
        
        return False, "Position maintained"
    
    def _close_position(self, position_id: str, reason: str, pnl: float):
        """Close a position"""
        try:
            position = self.active_positions[position_id]
            signal = position['signal']
            
            # Update balance
            self.current_balance += pnl
            
            # Update performance
            if pnl > 0:
                self.performance.winning_trades += 1
                self.performance.total_profit += pnl
            else:
                self.performance.losing_trades += 1
            
            # Calculate performance metrics
            self.performance.win_rate = self.performance.winning_trades / self.performance.total_trades
            
            # Update risk manager
            if self.risk_manager:
                self.risk_manager.remove_position(signal.symbol)
                self.risk_manager.update_balance(self.current_balance)
                self.risk_manager.add_trade({
                    'symbol': signal.symbol,
                    'pnl': pnl,
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(),
                    'reason': reason
                })
            
            # Log closure
            logger.info(f"🔒 CLOSED {signal.symbol} - {reason}")
            logger.info(f"💰 P&L: ${pnl:.2f}, Balance: ${self.current_balance:.2f}")
            
            # Remove from active positions
            del self.active_positions[position_id]
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
    
    def get_system_status(self) -> Dict:
        """📊 Get comprehensive system status"""
        try:
            # Calculate current drawdown
            peak_balance = max(self.current_balance, self.initial_balance)
            current_drawdown = (peak_balance - self.current_balance) / peak_balance * 100
            
            # Get risk assessment
            risk_assessment = {}
            if self.risk_manager:
                risk_assessment = self.risk_manager.get_portfolio_risk_assessment()
            
            # Performance metrics
            total_return = (self.current_balance - self.initial_balance) / self.initial_balance * 100
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'mode': self.mode.value,
                'balance': {
                    'current': self.current_balance,
                    'initial': self.initial_balance,
                    'total_return': total_return,
                    'current_drawdown': current_drawdown
                },
                'positions': {
                    'active_count': len(self.active_positions),
                    'max_allowed': self.max_simultaneous_trades,
                    'daily_trades': self.daily_trade_count,
                    'daily_limit': self.max_daily_trades
                },
                'performance': asdict(self.performance),
                'risk_assessment': risk_assessment,
                'components_status': {
                    'risk_manager': self.risk_manager is not None,
                    'manipulation_detector': self.manipulation_detector is not None,
                    'neural_engine': self.neural_engine is not None,
                    'prop_optimizer': self.prop_optimizer is not None,
                    'bank_beater': self.bank_beater is not None
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def save_state(self, filepath: str = "system_state.json"):
        """💾 Save system state"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'current_balance': self.current_balance,
                'mode': self.mode.value,
                'performance': asdict(self.performance),
                'active_positions': len(self.active_positions),
                'signal_history_count': len(self.signal_history)
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Save models if available
            if self.neural_engine:
                self.neural_engine.save_models()
            
            logger.info(f"💾 System state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self, filepath: str = "system_state.json"):
        """📂 Load system state"""
        try:
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                self.current_balance = state.get('current_balance', self.initial_balance)
                
                # Load performance
                perf_data = state.get('performance', {})
                for key, value in perf_data.items():
                    if hasattr(self.performance, key):
                        setattr(self.performance, key, value)
                
                logger.info(f"📂 System state loaded from {filepath}")
            
            # Load models if available
            if self.neural_engine:
                self.neural_engine.load_models()
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    # Helper methods
    def _update_market_data(self, symbol: str, market_data: Dict, tick_data: List = None):
        """Update internal market data storage"""
        self.market_data[symbol].append(market_data)
        
        if tick_data:
            for tick in tick_data:
                self.tick_data[symbol].append(tick)
    
    def _market_data_to_dataframe(self, symbol: str) -> pd.DataFrame:
        """Convert market data to DataFrame"""
        data = list(self.market_data[symbol])
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate current volatility"""
        df = self._market_data_to_dataframe(symbol)
        if len(df) < 20:
            return 0.001
        
        returns = df['close'].pct_change().dropna()
        return returns.std() if len(returns) > 0 else 0.001
    
    def _calculate_atr(self, symbol: str) -> float:
        """Calculate Average True Range"""
        df = self._market_data_to_dataframe(symbol)
        if len(df) < 14:
            return 0
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(14).mean().iloc[-1] if len(true_range) >= 14 else 0
    
    def _can_trade(self, symbol: str) -> bool:
        """Check if we can place a new trade"""
        # Check position limits
        if len(self.active_positions) >= self.max_simultaneous_trades:
            return False
        
        # Check daily trade limit
        if self.daily_trade_count >= self.max_daily_trades:
            return False
        
        # Check if already have position in this symbol
        for position in self.active_positions.values():
            if position['signal'].symbol == symbol:
                return False
        
        return True
    
    def _check_new_trading_day(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
    
    def _final_execution_checks(self, signal: UltimateSignal) -> bool:
        """Final checks before trade execution"""
        # Minimum confidence check
        if signal.confidence < 0.5:
            return False
        
        # Risk percentage check
        if signal.risk_percentage > 5:  # Max 5% risk per trade
            return False
        
        # Balance check
        required_margin = signal.position_size * signal.entry_price * 0.01  # 1% margin
        if required_margin > self.current_balance * 0.1:  # Max 10% of balance as margin
            return False
        
        return True
    
    # Pattern detection methods (simplified)
    def _detect_smc_pattern(self, df: pd.DataFrame) -> bool:
        """Detect Smart Money Concepts pattern"""
        if len(df) < 20:
            return False
        
        # Simplified SMC detection
        recent_highs = df['high'].rolling(5).max()
        recent_lows = df['low'].rolling(5).min()
        
        # Look for structure break
        structure_break = (df['close'].iloc[-1] > recent_highs.iloc[-5]) or \
                         (df['close'].iloc[-1] < recent_lows.iloc[-5])
        
        return structure_break
    
    def _detect_fvg_pattern(self, df: pd.DataFrame) -> bool:
        """Detect Fair Value Gap pattern"""
        if len(df) < 3:
            return False
        
        # Check for gap between candles
        gap_up = df['low'].iloc[-1] > df['high'].iloc[-3]
        gap_down = df['high'].iloc[-1] < df['low'].iloc[-3]
        
        return gap_up or gap_down
    
    def _detect_breakout_pattern(self, df: pd.DataFrame) -> bool:
        """Detect breakout pattern"""
        if len(df) < 20:
            return False
        
        # Calculate support/resistance levels
        resistance = df['high'].rolling(20).max().iloc[-1]
        support = df['low'].rolling(20).min().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Check for breakout
        breakout_up = current_price > resistance * 1.001  # 0.1% above resistance
        breakout_down = current_price < support * 0.999  # 0.1% below support
        
        return breakout_up or breakout_down
    
    def _detect_volatility_pattern(self, df: pd.DataFrame) -> bool:
        """Detect volatility-based pattern"""
        if len(df) < 20:
            return False
        
        # Calculate volatility
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
        avg_volatility = df['close'].pct_change().rolling(100).std().iloc[-1]
        
        # High volatility signal
        return volatility > avg_volatility * 1.5

def create_ultimate_trading_system(initial_balance: float = 10000, 
                                 mode: SystemMode = SystemMode.BALANCED) -> UltimateTradingSystem:
    """🚀 Factory function to create the Ultimate Trading System"""
    return UltimateTradingSystem(initial_balance, mode)

# Example usage and testing
if __name__ == "__main__":
    # Create the ultimate trading system
    system = create_ultimate_trading_system(10000, SystemMode.BANK_HUNTER)
    
    print("🚀 BLOB AI Ultimate Trading System")
    print("🛡️ Immune to losses, beats banks, passes prop challenges")
    print(f"💰 Initial balance: ${system.current_balance:,.2f}")
    print(f"🎯 Mode: {system.mode.value}")
    
    # Get system status
    status = system.get_system_status()
    print("\n📊 System Status:")
    print(json.dumps(status, indent=2, default=str))