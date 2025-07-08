# 🚀 BLOB AI - Immune Risk Management System
# Strategic Enhancement: Loss-Immune Trading with Advanced Protection

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
from collections import deque
import math

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    MINIMAL = "minimal"      # 0.1-0.3% risk
    LOW = "low"              # 0.3-0.5% risk
    MODERATE = "moderate"    # 0.5-1.0% risk
    AGGRESSIVE = "aggressive" # 1.0-2.0% risk
    EXTREME = "extreme"      # 2.0%+ risk

class MarketCondition(Enum):
    """Market condition types"""
    TRENDING_STRONG = "trending_strong"
    TRENDING_WEAK = "trending_weak"
    RANGING_TIGHT = "ranging_tight"
    RANGING_WIDE = "ranging_wide"
    VOLATILE_HIGH = "volatile_high"
    VOLATILE_LOW = "volatile_low"
    NEWS_EVENT = "news_event"
    SESSION_TRANSITION = "session_transition"

class ProtectionMode(Enum):
    """Protection mode types"""
    CAPITAL_PRESERVATION = "capital_preservation"  # Protect capital at all costs
    PROFIT_PROTECTION = "profit_protection"        # Protect existing profits
    GROWTH_OPTIMIZATION = "growth_optimization"    # Optimize for growth
    RECOVERY_MODE = "recovery_mode"                # Recover from drawdown
    CHALLENGE_MODE = "challenge_mode"              # Prop firm challenge mode

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    current_drawdown: float
    max_drawdown: float
    daily_var: float  # Value at Risk
    portfolio_beta: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    expectancy: float
    consecutive_losses: int
    largest_loss: float
    risk_adjusted_return: float

@dataclass
class PositionRisk:
    """Individual position risk assessment"""
    symbol: str
    position_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    risk_percentage: float
    correlation_risk: float
    liquidity_risk: float
    news_risk: float
    overall_risk_score: float

@dataclass
class RiskAdjustment:
    """Risk adjustment recommendations"""
    action: str  # 'reduce', 'increase', 'maintain', 'close'
    new_position_size: float
    new_stop_loss: float
    reasoning: str
    urgency: str  # 'immediate', 'next_candle', 'next_session'
    confidence: float

class ImmuneRiskManager:
    """🛡️ Advanced Immune Risk Management System
    
    Features:
    - Dynamic position sizing based on market conditions
    - Multi-layered stop loss protection
    - Correlation-based portfolio risk management
    - Drawdown recovery algorithms
    - News event protection
    - Session-based risk adjustment
    - Psychological bias protection
    - Prop firm rule compliance
    """
    
    def __init__(self, initial_balance: float, protection_mode: ProtectionMode = ProtectionMode.CAPITAL_PRESERVATION):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.protection_mode = protection_mode
        
        # Risk parameters
        self.max_portfolio_risk = 6.0  # Maximum portfolio risk %
        self.max_single_trade_risk = 1.0  # Maximum single trade risk %
        self.max_daily_risk = 3.0  # Maximum daily risk %
        self.max_correlation_exposure = 4.0  # Maximum correlated exposure %
        
        # Protection thresholds
        self.drawdown_protection_threshold = 5.0  # Start protection at 5% DD
        self.profit_protection_threshold = 10.0  # Protect profits above 10%
        self.consecutive_loss_limit = 3  # Max consecutive losses
        
        # Tracking
        self.trade_history = deque(maxlen=100)
        self.daily_pnl_history = deque(maxlen=30)
        self.risk_metrics_history = deque(maxlen=50)
        self.open_positions = {}
        
        # Market condition tracking
        self.current_market_condition = MarketCondition.RANGING_TIGHT
        self.volatility_regime = 'normal'
        self.news_events = []
        
        logger.info(f"🛡️ Immune Risk Manager initialized with {protection_mode.value} mode")
    
    def calculate_optimal_position_size(self, signal: Dict, market_data: Dict, 
                                      correlation_data: Dict = None) -> Tuple[float, RiskAdjustment]:
        """🎯 Calculate optimal position size with immune protection"""
        try:
            # Assess current market conditions
            market_condition = self._assess_market_conditions(market_data)
            
            # Calculate base position size
            base_size = self._calculate_base_position_size(signal)
            
            # Apply market condition adjustments
            condition_adjusted_size = self._apply_market_condition_adjustment(base_size, market_condition)
            
            # Apply correlation adjustments
            correlation_adjusted_size = self._apply_correlation_adjustment(
                condition_adjusted_size, signal['symbol'], correlation_data
            )
            
            # Apply protection mode adjustments
            protection_adjusted_size = self._apply_protection_mode_adjustment(correlation_adjusted_size)
            
            # Apply drawdown protection
            drawdown_adjusted_size = self._apply_drawdown_protection(protection_adjusted_size)
            
            # Apply consecutive loss protection
            final_size = self._apply_consecutive_loss_protection(drawdown_adjusted_size)
            
            # Generate risk adjustment recommendation
            risk_adjustment = self._generate_risk_adjustment(signal, final_size, base_size)
            
            return max(final_size, 0.01), risk_adjustment  # Minimum 0.01 lots
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01, RiskAdjustment(
                action='maintain',
                new_position_size=0.01,
                new_stop_loss=signal.get('stop_loss', 0),
                reasoning=f"Error in calculation: {e}",
                urgency='immediate',
                confidence=0.0
            )
    
    def _calculate_base_position_size(self, signal: Dict) -> float:
        """Calculate base position size using Kelly Criterion and fixed fractional"""
        # Get signal parameters
        stop_loss_pips = signal.get('stop_loss_pips', 30)
        win_rate = signal.get('win_rate', 0.6)
        avg_win = signal.get('avg_win_pips', 60)
        avg_loss = signal.get('avg_loss_pips', 30)
        
        # Calculate Kelly percentage
        if avg_loss > 0:
            kelly_percentage = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_percentage = max(0, min(kelly_percentage, 0.25))  # Cap at 25%
        else:
            kelly_percentage = 0.02  # Default 2%
        
        # Use conservative fraction of Kelly
        risk_percentage = kelly_percentage * 0.5  # 50% of Kelly
        
        # Apply maximum risk limits
        risk_percentage = min(risk_percentage, self.max_single_trade_risk / 100)
        
        # Calculate position size
        risk_amount = self.current_balance * risk_percentage
        pip_value = self._calculate_pip_value(signal.get('symbol', 'EURUSD'))
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        return position_size
    
    def _apply_market_condition_adjustment(self, base_size: float, condition: MarketCondition) -> float:
        """Adjust position size based on market conditions"""
        adjustments = {
            MarketCondition.TRENDING_STRONG: 1.2,    # Increase in strong trends
            MarketCondition.TRENDING_WEAK: 0.9,     # Slight decrease in weak trends
            MarketCondition.RANGING_TIGHT: 0.7,     # Reduce in tight ranges
            MarketCondition.RANGING_WIDE: 1.0,      # Normal in wide ranges
            MarketCondition.VOLATILE_HIGH: 0.5,     # Significant reduction in high volatility
            MarketCondition.VOLATILE_LOW: 1.1,      # Slight increase in low volatility
            MarketCondition.NEWS_EVENT: 0.3,        # Major reduction during news
            MarketCondition.SESSION_TRANSITION: 0.8  # Reduce during transitions
        }
        
        return base_size * adjustments.get(condition, 1.0)
    
    def _apply_correlation_adjustment(self, base_size: float, symbol: str, 
                                    correlation_data: Dict = None) -> float:
        """Adjust for correlation risk with existing positions"""
        if not self.open_positions or not correlation_data:
            return base_size
        
        total_correlated_exposure = 0
        
        for pos_symbol, position in self.open_positions.items():
            if pos_symbol != symbol:
                correlation = correlation_data.get(f"{symbol}_{pos_symbol}", 0)
                if abs(correlation) > 0.7:  # High correlation
                    total_correlated_exposure += position['risk_amount'] * abs(correlation)
        
        # Reduce size if high correlation exposure
        correlation_exposure_pct = total_correlated_exposure / self.current_balance * 100
        
        if correlation_exposure_pct > self.max_correlation_exposure:
            reduction_factor = self.max_correlation_exposure / correlation_exposure_pct
            return base_size * reduction_factor
        
        return base_size
    
    def _apply_protection_mode_adjustment(self, base_size: float) -> float:
        """Apply protection mode specific adjustments"""
        current_return = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        
        if self.protection_mode == ProtectionMode.CAPITAL_PRESERVATION:
            # Very conservative - reduce size as drawdown increases
            if current_return < -2:
                return base_size * 0.5
            elif current_return < 0:
                return base_size * 0.8
            return base_size * 0.9  # Always conservative
        
        elif self.protection_mode == ProtectionMode.PROFIT_PROTECTION:
            # Protect profits once achieved
            if current_return > self.profit_protection_threshold:
                return base_size * 0.6  # Reduce risk to protect profits
            elif current_return > 5:
                return base_size * 0.8
            return base_size
        
        elif self.protection_mode == ProtectionMode.GROWTH_OPTIMIZATION:
            # Optimize for growth while managing risk
            if current_return > 20:
                return base_size * 1.2  # Increase when doing well
            elif current_return < -5:
                return base_size * 0.7  # Reduce when struggling
            return base_size
        
        elif self.protection_mode == ProtectionMode.RECOVERY_MODE:
            # Careful recovery from drawdown
            if current_return < -10:
                return base_size * 0.3  # Very small sizes in deep drawdown
            elif current_return < -5:
                return base_size * 0.5
            elif current_return > 0:
                return base_size * 1.1  # Slightly increase when recovering
            return base_size * 0.8
        
        elif self.protection_mode == ProtectionMode.CHALLENGE_MODE:
            # Prop firm challenge optimization
            progress = current_return / 10.0  # Assuming 10% target
            if progress > 0.9:  # Close to target
                return base_size * 0.3  # Very conservative
            elif progress > 0.7:
                return base_size * 0.5
            elif progress < 0.3:
                return base_size * 1.2  # More aggressive early on
            return base_size
        
        return base_size
    
    def _apply_drawdown_protection(self, base_size: float) -> float:
        """Apply drawdown-based protection"""
        current_drawdown = self._calculate_current_drawdown()
        
        if current_drawdown > 10:  # Severe drawdown
            return base_size * 0.2
        elif current_drawdown > 7:
            return base_size * 0.4
        elif current_drawdown > 5:
            return base_size * 0.6
        elif current_drawdown > 3:
            return base_size * 0.8
        
        return base_size
    
    def _apply_consecutive_loss_protection(self, base_size: float) -> float:
        """Apply consecutive loss protection"""
        consecutive_losses = self._count_consecutive_losses()
        
        if consecutive_losses >= 5:
            return base_size * 0.2  # Drastically reduce after 5 losses
        elif consecutive_losses >= 3:
            return base_size * 0.5  # Reduce after 3 losses
        elif consecutive_losses >= 2:
            return base_size * 0.8  # Slight reduction after 2 losses
        
        return base_size
    
    def should_close_position(self, position: Dict, market_data: Dict) -> Tuple[bool, str]:
        """🛡️ Determine if position should be closed for protection"""
        symbol = position['symbol']
        current_price = market_data.get('current_price', {}).get('bid', position['entry_price'])
        
        # Calculate current P&L
        if position['direction'] == 'buy':
            pnl = (current_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - current_price) * position['size']
        
        pnl_percentage = pnl / self.current_balance * 100
        
        # Emergency stop conditions
        if pnl_percentage < -2:  # 2% loss on single trade
            return True, "Emergency stop - excessive single trade loss"
        
        # News event protection
        if self._is_high_impact_news_approaching(symbol):
            if pnl_percentage > 0.5:  # Close profitable positions before news
                return True, "Profit protection before high impact news"
        
        # Session close protection
        if self._is_session_close_approaching() and not position.get('hold_overnight', False):
            return True, "Session close protection"
        
        # Correlation risk protection
        if self._is_high_correlation_risk(symbol):
            return True, "High correlation risk detected"
        
        # Drawdown protection
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > 8 and pnl_percentage < 0:
            return True, "Drawdown protection - close losing positions"
        
        return False, "Position within acceptable risk parameters"
    
    def adjust_stop_loss(self, position: Dict, market_data: Dict) -> Tuple[float, str]:
        """🎯 Dynamically adjust stop loss for maximum protection"""
        symbol = position['symbol']
        current_price = market_data.get('current_price', {}).get('bid', position['entry_price'])
        current_sl = position['stop_loss']
        
        # Calculate current profit
        if position['direction'] == 'buy':
            profit_pips = (current_price - position['entry_price']) * 10000
        else:
            profit_pips = (position['entry_price'] - current_price) * 10000
        
        # Breakeven protection
        if profit_pips > 20:  # Move to breakeven after 20 pips profit
            if position['direction'] == 'buy':
                new_sl = max(current_sl, position['entry_price'])
            else:
                new_sl = min(current_sl, position['entry_price'])
            
            if new_sl != current_sl:
                return new_sl, "Moved stop to breakeven"
        
        # Trailing stop
        if profit_pips > 50:  # Start trailing after 50 pips
            trail_distance = 30  # 30 pip trail
            
            if position['direction'] == 'buy':
                new_sl = max(current_sl, current_price - trail_distance * 0.0001)
            else:
                new_sl = min(current_sl, current_price + trail_distance * 0.0001)
            
            if new_sl != current_sl:
                return new_sl, f"Trailing stop adjusted - {trail_distance} pip trail"
        
        # Volatility-based adjustment
        atr = market_data.get('atr', 0.001)
        if atr > 0.002:  # High volatility
            volatility_buffer = atr * 1.5
            
            if position['direction'] == 'buy':
                suggested_sl = current_price - volatility_buffer
                if suggested_sl > current_sl:  # Only widen, never tighten
                    return suggested_sl, "Widened stop for high volatility"
            else:
                suggested_sl = current_price + volatility_buffer
                if suggested_sl < current_sl:  # Only widen, never tighten
                    return suggested_sl, "Widened stop for high volatility"
        
        return current_sl, "Stop loss maintained"
    
    def get_portfolio_risk_assessment(self) -> Dict:
        """📊 Get comprehensive portfolio risk assessment"""
        current_drawdown = self._calculate_current_drawdown()
        risk_metrics = self._calculate_risk_metrics()
        
        # Calculate portfolio exposure
        total_risk = sum(pos['risk_amount'] for pos in self.open_positions.values())
        portfolio_risk_pct = total_risk / self.current_balance * 100
        
        # Risk level assessment
        if portfolio_risk_pct > 8:
            risk_level = RiskLevel.EXTREME
        elif portfolio_risk_pct > 5:
            risk_level = RiskLevel.AGGRESSIVE
        elif portfolio_risk_pct > 3:
            risk_level = RiskLevel.MODERATE
        elif portfolio_risk_pct > 1:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MINIMAL
        
        assessment = {
            'overall_risk_level': risk_level.value,
            'portfolio_risk_percentage': portfolio_risk_pct,
            'current_drawdown': current_drawdown,
            'max_drawdown': risk_metrics.max_drawdown,
            'open_positions': len(self.open_positions),
            'total_exposure': total_risk,
            'protection_mode': self.protection_mode.value,
            'consecutive_losses': self._count_consecutive_losses(),
            'recommendations': self._get_risk_recommendations(risk_level, current_drawdown)
        }
        
        return assessment
    
    def _assess_market_conditions(self, market_data: Dict) -> MarketCondition:
        """Assess current market conditions"""
        # This would analyze volatility, trend strength, etc.
        # Simplified implementation
        atr = market_data.get('atr', 0.001)
        
        if atr > 0.003:
            return MarketCondition.VOLATILE_HIGH
        elif atr < 0.0005:
            return MarketCondition.VOLATILE_LOW
        else:
            return MarketCondition.RANGING_TIGHT
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown percentage"""
        peak_balance = max(self.current_balance, self.initial_balance)
        if hasattr(self, 'peak_balance'):
            peak_balance = max(peak_balance, self.peak_balance)
        
        drawdown = (peak_balance - self.current_balance) / peak_balance * 100
        return max(0, drawdown)
    
    def _count_consecutive_losses(self) -> int:
        """Count consecutive losing trades"""
        consecutive = 0
        for trade in reversed(self.trade_history):
            if trade.get('pnl', 0) < 0:
                consecutive += 1
            else:
                break
        return consecutive
    
    def _calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        if not self.trade_history:
            return RiskMetrics(
                current_drawdown=0, max_drawdown=0, daily_var=0, portfolio_beta=1,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, win_rate=0,
                profit_factor=0, expectancy=0, consecutive_losses=0,
                largest_loss=0, risk_adjusted_return=0
            )
        
        # Calculate basic metrics
        pnls = [trade.get('pnl', 0) for trade in self.trade_history]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        
        win_rate = len(wins) / len(pnls) if pnls else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')
        expectancy = np.mean(pnls) if pnls else 0
        
        return RiskMetrics(
            current_drawdown=self._calculate_current_drawdown(),
            max_drawdown=max([trade.get('drawdown', 0) for trade in self.trade_history] + [0]),
            daily_var=np.percentile(pnls, 5) if pnls else 0,
            portfolio_beta=1.0,  # Would need market data
            sharpe_ratio=expectancy / np.std(pnls) if pnls and np.std(pnls) > 0 else 0,
            sortino_ratio=0,  # Would need downside deviation
            calmar_ratio=0,   # Would need annual return
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            consecutive_losses=self._count_consecutive_losses(),
            largest_loss=min(losses) if losses else 0,
            risk_adjusted_return=0  # Would need risk-free rate
        )
    
    def _generate_risk_adjustment(self, signal: Dict, final_size: float, base_size: float) -> RiskAdjustment:
        """Generate risk adjustment recommendation"""
        size_ratio = final_size / base_size if base_size > 0 else 1
        
        if size_ratio < 0.5:
            action = 'reduce'
            urgency = 'immediate'
            reasoning = "High risk environment - significant position reduction"
        elif size_ratio < 0.8:
            action = 'reduce'
            urgency = 'next_candle'
            reasoning = "Moderate risk - position reduction recommended"
        elif size_ratio > 1.2:
            action = 'increase'
            urgency = 'next_candle'
            reasoning = "Favorable conditions - position increase approved"
        else:
            action = 'maintain'
            urgency = 'next_session'
            reasoning = "Normal risk parameters"
        
        return RiskAdjustment(
            action=action,
            new_position_size=final_size,
            new_stop_loss=signal.get('stop_loss', 0),
            reasoning=reasoning,
            urgency=urgency,
            confidence=0.8
        )
    
    def _get_risk_recommendations(self, risk_level: RiskLevel, drawdown: float) -> List[str]:
        """Get risk management recommendations"""
        recommendations = []
        
        if risk_level == RiskLevel.EXTREME:
            recommendations.append("🚨 EXTREME RISK - Consider closing all positions")
            recommendations.append("📉 Reduce position sizes to minimum")
        elif risk_level == RiskLevel.AGGRESSIVE:
            recommendations.append("⚠️ High risk detected - reduce exposure")
            recommendations.append("🎯 Focus on highest probability setups only")
        
        if drawdown > 10:
            recommendations.append("🛡️ Severe drawdown - activate recovery mode")
        elif drawdown > 5:
            recommendations.append("📊 Moderate drawdown - increase selectivity")
        
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses >= 3:
            recommendations.append(f"🔄 {consecutive_losses} consecutive losses - consider break")
        
        return recommendations
    
    def _calculate_pip_value(self, symbol: str) -> float:
        """Calculate pip value for position sizing"""
        if 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001
    
    def _is_high_impact_news_approaching(self, symbol: str) -> bool:
        """Check if high impact news is approaching"""
        # Simplified implementation
        return False
    
    def _is_session_close_approaching(self) -> bool:
        """Check if trading session close is approaching"""
        # Simplified implementation
        current_hour = datetime.now().hour
        return current_hour >= 21  # 9 PM UTC
    
    def _is_high_correlation_risk(self, symbol: str) -> bool:
        """Check if there's high correlation risk"""
        # Simplified implementation
        return False
    
    def update_balance(self, new_balance: float):
        """Update current balance"""
        self.current_balance = new_balance
        if not hasattr(self, 'peak_balance'):
            self.peak_balance = new_balance
        else:
            self.peak_balance = max(self.peak_balance, new_balance)
    
    def add_trade(self, trade_data: Dict):
        """Add completed trade to history"""
        self.trade_history.append(trade_data)
    
    def add_position(self, symbol: str, position_data: Dict):
        """Add open position"""
        self.open_positions[symbol] = position_data
    
    def remove_position(self, symbol: str):
        """Remove closed position"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]

def create_immune_risk_manager(initial_balance: float, 
                             protection_mode: ProtectionMode = ProtectionMode.CAPITAL_PRESERVATION) -> ImmuneRiskManager:
    """Factory function to create immune risk manager"""
    return ImmuneRiskManager(initial_balance, protection_mode)