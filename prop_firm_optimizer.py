# 🚀 BLOB AI - Prop Firm Challenge Optimizer
# Strategic Enhancement: Beat Prop Challenges with Precision

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PropFirmType(Enum):
    """Different prop firm challenge types"""
    FTMO = "ftmo"
    FUNDED_TRADER = "funded_trader"
    MY_FOREX_FUNDS = "my_forex_funds"
    THE5ERS = "the5ers"
    TOPSTEP = "topstep"
    APEX = "apex"
    SURGE_TRADER = "surge_trader"

class ChallengePhase(Enum):
    """Challenge phases"""
    EVALUATION = "evaluation"
    VERIFICATION = "verification"
    FUNDED = "funded"

@dataclass
class PropFirmRules:
    """Prop firm specific rules and constraints"""
    firm_type: PropFirmType
    phase: ChallengePhase
    account_size: float
    profit_target: float  # %
    max_daily_loss: float  # %
    max_total_loss: float  # %
    min_trading_days: int
    max_lot_size: float
    news_trading_allowed: bool
    weekend_holding_allowed: bool
    ea_allowed: bool
    max_positions: int
    consistency_rule: bool  # Best day can't exceed X% of total profit
    consistency_percentage: float  # Usually 50%

@dataclass
class RiskMetrics:
    """Real-time risk tracking"""
    current_drawdown: float
    daily_pnl: float
    total_pnl: float
    profit_target_progress: float  # %
    days_traded: int
    largest_win_day: float
    consistency_ratio: float
    risk_score: float  # 0-100

class PropFirmOptimizer:
    """🎯 Prop Firm Challenge Optimizer
    
    Features:
    - Dynamic risk adjustment based on challenge progress
    - Firm-specific rule compliance
    - Consistency optimization
    - News avoidance strategies
    - Drawdown protection
    - Profit target achievement optimization
    """
    
    def __init__(self, firm_rules: PropFirmRules):
        self.firm_rules = firm_rules
        self.daily_pnl_history = []
        self.trade_history = []
        self.current_metrics = None
        
        # Risk parameters based on firm type
        self.risk_params = self._get_firm_risk_params()
        
        logger.info(f"🎯 Prop Firm Optimizer initialized for {firm_rules.firm_type.value}")
    
    def _get_firm_risk_params(self) -> Dict:
        """Get firm-specific risk parameters"""
        base_params = {
            'max_risk_per_trade': 0.5,  # % of account
            'max_daily_risk': 2.0,      # % of account
            'profit_protection_threshold': 80,  # % of target
            'consistency_buffer': 0.1,   # Buffer for consistency rule
            'news_avoidance_hours': 2,   # Hours before/after news
            'weekend_close_threshold': 0.8  # Close positions if this much profit
        }
        
        # Firm-specific adjustments
        if self.firm_rules.firm_type == PropFirmType.FTMO:
            base_params.update({
                'max_risk_per_trade': 0.3,
                'consistency_buffer': 0.15
            })
        elif self.firm_rules.firm_type == PropFirmType.THE5ERS:
            base_params.update({
                'max_risk_per_trade': 0.4,
                'max_daily_risk': 1.5
            })
        
        return base_params
    
    def calculate_optimal_position_size(self, signal: Dict, current_balance: float, 
                                      current_drawdown: float) -> float:
        """🎯 Calculate optimal position size for prop firm rules"""
        try:
            # Update current metrics
            self._update_risk_metrics(current_balance, current_drawdown)
            
            # Base position size calculation
            stop_loss_pips = signal.get('stop_loss_pips', 30)
            pip_value = self._calculate_pip_value(signal.get('symbol', 'EURUSD'))
            
            # Maximum risk per trade (account %)
            max_risk_amount = current_balance * (self.risk_params['max_risk_per_trade'] / 100)
            
            # Adjust risk based on challenge progress
            risk_adjustment = self._calculate_risk_adjustment()
            adjusted_risk = max_risk_amount * risk_adjustment
            
            # Calculate position size
            position_size = adjusted_risk / (stop_loss_pips * pip_value)
            
            # Apply firm-specific constraints
            position_size = self._apply_firm_constraints(position_size)
            
            # Daily risk check
            if not self._check_daily_risk_limit(position_size, stop_loss_pips, pip_value):
                logger.warning("Position size exceeds daily risk limit")
                return 0.0
            
            return round(position_size, 2)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _calculate_risk_adjustment(self) -> float:
        """Calculate risk adjustment based on challenge progress"""
        if not self.current_metrics:
            return 1.0
        
        progress = self.current_metrics.profit_target_progress
        drawdown = abs(self.current_metrics.current_drawdown)
        
        # Reduce risk as we approach profit target
        if progress > 80:
            risk_multiplier = 0.3  # Very conservative near target
        elif progress > 60:
            risk_multiplier = 0.5  # Moderate risk
        elif progress > 40:
            risk_multiplier = 0.8  # Standard risk
        else:
            risk_multiplier = 1.0  # Full risk early in challenge
        
        # Reduce risk if significant drawdown
        if drawdown > 3:
            risk_multiplier *= 0.5
        elif drawdown > 2:
            risk_multiplier *= 0.7
        
        # Consistency rule adjustment
        if self.firm_rules.consistency_rule:
            consistency_adjustment = self._calculate_consistency_adjustment()
            risk_multiplier *= consistency_adjustment
        
        return max(risk_multiplier, 0.1)  # Minimum 10% of normal risk
    
    def _calculate_consistency_adjustment(self) -> float:
        """Adjust risk to maintain consistency rule compliance"""
        if not self.daily_pnl_history or len(self.daily_pnl_history) < 2:
            return 1.0
        
        total_profit = sum(pnl for pnl in self.daily_pnl_history if pnl > 0)
        largest_day = max(self.daily_pnl_history) if self.daily_pnl_history else 0
        
        if total_profit <= 0:
            return 1.0
        
        current_ratio = largest_day / total_profit
        max_allowed_ratio = self.firm_rules.consistency_percentage / 100
        
        # If we're close to violating consistency rule, reduce risk
        if current_ratio > (max_allowed_ratio - self.risk_params['consistency_buffer']):
            return 0.2  # Very low risk to avoid violation
        elif current_ratio > (max_allowed_ratio - 0.2):
            return 0.5  # Moderate risk
        
        return 1.0
    
    def should_take_trade(self, signal: Dict, market_context: Dict) -> Tuple[bool, str]:
        """🎯 Determine if trade should be taken based on prop firm rules"""
        try:
            # Check basic rule compliance
            compliance_check = self._check_rule_compliance(signal, market_context)
            if not compliance_check[0]:
                return compliance_check
            
            # Check news events
            if not self.firm_rules.news_trading_allowed:
                news_check = self._check_news_events(market_context)
                if not news_check[0]:
                    return news_check
            
            # Check weekend holding
            if not self.firm_rules.weekend_holding_allowed:
                weekend_check = self._check_weekend_holding()
                if not weekend_check[0]:
                    return weekend_check
            
            # Check consistency rule
            if self.firm_rules.consistency_rule:
                consistency_check = self._check_consistency_rule()
                if not consistency_check[0]:
                    return consistency_check
            
            # Check if we're close to profit target (be more selective)
            if self.current_metrics and self.current_metrics.profit_target_progress > 90:
                return False, "Too close to profit target - avoiding unnecessary risk"
            
            return True, "Trade approved for prop firm rules"
            
        except Exception as e:
            logger.error(f"Error in trade decision: {e}")
            return False, f"Error in trade validation: {e}"
    
    def _check_rule_compliance(self, signal: Dict, market_context: Dict) -> Tuple[bool, str]:
        """Check basic prop firm rule compliance"""
        # Check daily loss limit
        if self.current_metrics and abs(self.current_metrics.daily_pnl) > self.firm_rules.max_daily_loss:
            return False, "Daily loss limit reached"
        
        # Check total loss limit
        if self.current_metrics and abs(self.current_metrics.current_drawdown) > self.firm_rules.max_total_loss:
            return False, "Total loss limit reached"
        
        # Check maximum positions
        open_positions = market_context.get('open_positions', 0)
        if open_positions >= self.firm_rules.max_positions:
            return False, "Maximum positions limit reached"
        
        return True, "Basic rules compliant"
    
    def _check_news_events(self, market_context: Dict) -> Tuple[bool, str]:
        """Check for upcoming high-impact news events"""
        news_events = market_context.get('upcoming_news', [])
        current_time = datetime.now()
        
        for event in news_events:
            event_time = event.get('time')
            impact = event.get('impact', 'low')
            
            if impact.lower() in ['high', 'medium']:
                time_diff = abs((event_time - current_time).total_seconds() / 3600)
                if time_diff <= self.risk_params['news_avoidance_hours']:
                    return False, f"High impact news event within {self.risk_params['news_avoidance_hours']} hours"
        
        return True, "No conflicting news events"
    
    def _check_weekend_holding(self) -> Tuple[bool, str]:
        """Check weekend holding restrictions"""
        current_time = datetime.now()
        
        # Check if it's Friday after certain time (depends on broker timezone)
        if current_time.weekday() == 4 and current_time.hour >= 20:  # Friday 8 PM
            if self.current_metrics and self.current_metrics.profit_target_progress > self.risk_params['weekend_close_threshold'] * 100:
                return False, "Close to profit target - avoid weekend risk"
        
        return True, "Weekend holding acceptable"
    
    def _check_consistency_rule(self) -> Tuple[bool, str]:
        """Check consistency rule compliance"""
        if not self.daily_pnl_history:
            return True, "No trading history yet"
        
        total_profit = sum(pnl for pnl in self.daily_pnl_history if pnl > 0)
        today_pnl = self.current_metrics.daily_pnl if self.current_metrics else 0
        
        if total_profit <= 0:
            return True, "No profits yet"
        
        # Project what today's ratio would be if we add more profit
        projected_total = total_profit + max(today_pnl, 0)
        max_allowed_day_profit = projected_total * (self.firm_rules.consistency_percentage / 100)
        
        if today_pnl > max_allowed_day_profit * 0.8:  # 80% of max allowed
            return False, "Approaching consistency rule limit for today"
        
        return True, "Consistency rule compliant"
    
    def _apply_firm_constraints(self, position_size: float) -> float:
        """Apply firm-specific position size constraints"""
        # Maximum lot size constraint
        if position_size > self.firm_rules.max_lot_size:
            position_size = self.firm_rules.max_lot_size
        
        # Minimum position size (0.01 lots typically)
        if position_size < 0.01:
            position_size = 0.01
        
        return position_size
    
    def _check_daily_risk_limit(self, position_size: float, stop_loss_pips: float, pip_value: float) -> bool:
        """Check if position would exceed daily risk limit"""
        trade_risk = position_size * stop_loss_pips * pip_value
        current_daily_risk = abs(self.current_metrics.daily_pnl) if self.current_metrics else 0
        
        max_daily_risk = self.firm_rules.account_size * (self.risk_params['max_daily_risk'] / 100)
        
        return (current_daily_risk + trade_risk) <= max_daily_risk
    
    def _calculate_pip_value(self, symbol: str) -> float:
        """Calculate pip value for position sizing"""
        # Simplified pip value calculation
        if 'JPY' in symbol:
            return 0.01  # For JPY pairs
        else:
            return 0.0001  # For most other pairs
    
    def _update_risk_metrics(self, current_balance: float, current_drawdown: float):
        """Update current risk metrics"""
        # This would be updated with real trading data
        self.current_metrics = RiskMetrics(
            current_drawdown=current_drawdown,
            daily_pnl=0.0,  # Would be calculated from today's trades
            total_pnl=current_balance - self.firm_rules.account_size,
            profit_target_progress=(current_balance - self.firm_rules.account_size) / (self.firm_rules.account_size * self.firm_rules.profit_target / 100) * 100,
            days_traded=len(self.daily_pnl_history),
            largest_win_day=max(self.daily_pnl_history) if self.daily_pnl_history else 0,
            consistency_ratio=0.0,  # Would be calculated
            risk_score=50.0  # Would be calculated based on various factors
        )
    
    def get_challenge_status(self) -> Dict:
        """Get current challenge status and recommendations"""
        if not self.current_metrics:
            return {'status': 'No data available'}
        
        status = {
            'phase': self.firm_rules.phase.value,
            'firm': self.firm_rules.firm_type.value,
            'profit_target_progress': f"{self.current_metrics.profit_target_progress:.1f}%",
            'current_drawdown': f"{self.current_metrics.current_drawdown:.2f}%",
            'days_traded': self.current_metrics.days_traded,
            'min_days_required': self.firm_rules.min_trading_days,
            'consistency_status': 'COMPLIANT' if self._check_consistency_rule()[0] else 'AT_RISK',
            'risk_level': self._get_risk_level(),
            'recommendations': self._get_recommendations()
        }
        
        return status
    
    def _get_risk_level(self) -> str:
        """Determine current risk level"""
        if not self.current_metrics:
            return 'UNKNOWN'
        
        if self.current_metrics.current_drawdown > 4:
            return 'HIGH'
        elif self.current_metrics.current_drawdown > 2:
            return 'MEDIUM'
        elif self.current_metrics.profit_target_progress > 90:
            return 'LOW'  # Conservative near target
        else:
            return 'NORMAL'
    
    def _get_recommendations(self) -> List[str]:
        """Get trading recommendations based on current status"""
        recommendations = []
        
        if not self.current_metrics:
            return ['Insufficient data for recommendations']
        
        # Progress-based recommendations
        if self.current_metrics.profit_target_progress > 90:
            recommendations.append("🎯 Very close to target - take only highest probability setups")
        elif self.current_metrics.profit_target_progress > 70:
            recommendations.append("📈 Good progress - maintain conservative approach")
        elif self.current_metrics.profit_target_progress < 30:
            recommendations.append("🚀 Early stage - can take more aggressive approach")
        
        # Drawdown-based recommendations
        if self.current_metrics.current_drawdown > 3:
            recommendations.append("⚠️ Significant drawdown - reduce position sizes")
        elif self.current_metrics.current_drawdown > 1.5:
            recommendations.append("📉 Moderate drawdown - be more selective with trades")
        
        # Consistency recommendations
        if self.firm_rules.consistency_rule and not self._check_consistency_rule()[0]:
            recommendations.append("⚖️ Consistency rule at risk - limit today's profits")
        
        # Time-based recommendations
        days_remaining = 30 - self.current_metrics.days_traded  # Assuming 30-day challenge
        if days_remaining < 5 and self.current_metrics.profit_target_progress < 80:
            recommendations.append("⏰ Time pressure - may need to increase activity")
        
        return recommendations

# Predefined prop firm configurations
PROP_FIRM_CONFIGS = {
    PropFirmType.FTMO: {
        'evaluation': PropFirmRules(
            firm_type=PropFirmType.FTMO,
            phase=ChallengePhase.EVALUATION,
            account_size=100000,
            profit_target=10.0,
            max_daily_loss=5.0,
            max_total_loss=10.0,
            min_trading_days=4,
            max_lot_size=10.0,
            news_trading_allowed=False,
            weekend_holding_allowed=True,
            ea_allowed=True,
            max_positions=10,
            consistency_rule=True,
            consistency_percentage=50.0
        ),
        'verification': PropFirmRules(
            firm_type=PropFirmType.FTMO,
            phase=ChallengePhase.VERIFICATION,
            account_size=100000,
            profit_target=5.0,
            max_daily_loss=5.0,
            max_total_loss=10.0,
            min_trading_days=4,
            max_lot_size=10.0,
            news_trading_allowed=False,
            weekend_holding_allowed=True,
            ea_allowed=True,
            max_positions=10,
            consistency_rule=True,
            consistency_percentage=50.0
        )
    }
}

def create_prop_firm_optimizer(firm_type: PropFirmType, phase: ChallengePhase) -> PropFirmOptimizer:
    """Factory function to create prop firm optimizer"""
    if firm_type in PROP_FIRM_CONFIGS and phase.value in PROP_FIRM_CONFIGS[firm_type]:
        rules = PROP_FIRM_CONFIGS[firm_type][phase.value]
        return PropFirmOptimizer(rules)
    else:
        raise ValueError(f"Unsupported firm type {firm_type} or phase {phase}")