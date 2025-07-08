# 🚀 BLOB AI - Adaptive Signal Weighting System
# Strategic Enhancement #4: Beat the Banks with Context-Aware Signal Intelligence

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_BULLISH = "trending_bullish"      # Strong uptrend
    TRENDING_BEARISH = "trending_bearish"      # Strong downtrend
    RANGING_HIGH_VOL = "ranging_high_vol"      # Choppy, high volatility
    RANGING_LOW_VOL = "ranging_low_vol"        # Quiet, low volatility
    BREAKOUT_BULLISH = "breakout_bullish"      # Breaking higher
    BREAKOUT_BEARISH = "breakout_bearish"      # Breaking lower
    REVERSAL_ZONE = "reversal_zone"            # Potential reversal area
    CONSOLIDATION = "consolidation"            # Tight consolidation

class SignalConfidence(Enum):
    """Signal confidence levels"""
    VERY_HIGH = "very_high"    # 0.9+
    HIGH = "high"              # 0.7-0.9
    MEDIUM = "medium"          # 0.5-0.7
    LOW = "low"                # 0.3-0.5
    VERY_LOW = "very_low"      # <0.3

@dataclass
class RegimeMetrics:
    """Market regime analysis metrics"""
    atr_percentile: float          # ATR percentile (0-100)
    bb_width_percentile: float     # Bollinger Bands width percentile
    htf_bos_frequency: float       # Higher timeframe BOS frequency
    ema200_distance: float         # Distance from EMA200 (normalized)
    trend_strength: float          # Trend strength indicator
    volatility_regime: str         # 'high', 'medium', 'low'
    trend_regime: str              # 'trending', 'ranging'
    momentum_regime: str           # 'strong', 'weak', 'neutral'

@dataclass
class SignalWeight:
    """Signal weighting factors"""
    base_weight: float             # Base signal weight
    regime_multiplier: float       # Regime-based multiplier
    confluence_bonus: float        # Multi-timeframe confluence bonus
    volatility_penalty: float      # High volatility penalty
    session_bonus: float           # Session-based bonus
    correlation_factor: float      # DXY/yield correlation factor
    final_weight: float            # Final calculated weight
    confidence: SignalConfidence   # Overall confidence level

@dataclass
class EnhancedSignal:
    """Enhanced signal with adaptive weighting"""
    original_signal: Dict          # Original signal data
    signal_weight: SignalWeight    # Weighting analysis
    regime_context: RegimeMetrics  # Market regime context
    recommended_action: str        # 'take', 'reduce', 'skip', 'wait'
    risk_adjustment: float         # Suggested risk adjustment
    entry_timing: str              # 'immediate', 'wait_pullback', 'wait_confirmation'

class AdaptiveSignalWeighting:
    """🔍 Strategic Enhancement #4: Adaptive Signal Weighting System
    
    Features:
    - Market regime detection (ATR, BB width, HTF BOS frequency)
    - Dynamic signal weighting based on context
    - Multi-timeframe confluence analysis
    - Volatility-based risk adjustment
    - Session-aware signal enhancement
    - DXY/yield correlation integration
    - Smart signal filtering and ranking
    """
    
    def __init__(self, correlation_pairs: List[str] = None):
        # Configuration
        self.atr_period = 14
        self.bb_period = 20
        self.bb_std = 2.0
        self.ema200_period = 200
        self.regime_lookback = 100
        
        # Weighting parameters
        self.base_weights = {
            'order_block': 1.0,
            'fair_value_gap': 0.8,
            'liquidity_sweep': 1.2,
            'market_structure_break': 1.1,
            'change_of_character': 0.9
        }
        
        # Regime multipliers
        self.regime_multipliers = {
            MarketRegime.TRENDING_BULLISH: {'bullish': 1.3, 'bearish': 0.6},
            MarketRegime.TRENDING_BEARISH: {'bullish': 0.6, 'bearish': 1.3},
            MarketRegime.RANGING_HIGH_VOL: {'bullish': 0.7, 'bearish': 0.7},
            MarketRegime.RANGING_LOW_VOL: {'bullish': 1.1, 'bearish': 1.1},
            MarketRegime.BREAKOUT_BULLISH: {'bullish': 1.4, 'bearish': 0.5},
            MarketRegime.BREAKOUT_BEARISH: {'bullish': 0.5, 'bearish': 1.4},
            MarketRegime.REVERSAL_ZONE: {'bullish': 0.8, 'bearish': 0.8},
            MarketRegime.CONSOLIDATION: {'bullish': 1.0, 'bearish': 1.0}
        }
        
        # Session weights
        self.session_weights = {
            'london': 1.2,
            'new_york': 1.3,
            'asia': 0.8,
            'overlap': 1.4
        }
        
        # Correlation pairs for additional context
        self.correlation_pairs = correlation_pairs or ['DXY', 'US10Y']
        
        logger.info("🔍 Adaptive Signal Weighting System initialized")
    
    def analyze_and_weight_signals(self, signals: List[Dict], 
                                 market_data: Dict[str, pd.DataFrame],
                                 correlation_data: Dict[str, pd.DataFrame] = None) -> List[EnhancedSignal]:
        """Main function to analyze and weight signals"""
        try:
            if not signals:
                return []
            
            # Get primary timeframe data
            primary_df = market_data.get('H1', market_data.get('M15', None))
            if primary_df is None or len(primary_df) < self.regime_lookback:
                logger.warning("Insufficient data for regime analysis")
                return self._create_basic_enhanced_signals(signals)
            
            # Analyze market regime
            regime_metrics = self._analyze_market_regime(market_data)
            current_regime = self._determine_market_regime(regime_metrics)
            
            # Process each signal
            enhanced_signals = []
            for signal in signals:
                try:
                    enhanced_signal = self._enhance_signal(
                        signal, regime_metrics, current_regime, 
                        market_data, correlation_data
                    )
                    enhanced_signals.append(enhanced_signal)
                except Exception as e:
                    logger.error(f"Error enhancing signal: {e}")
                    continue
            
            # Sort by final weight (highest first)
            enhanced_signals.sort(key=lambda x: x.signal_weight.final_weight, reverse=True)
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"Error in adaptive signal weighting: {e}")
            return self._create_basic_enhanced_signals(signals)
    
    def _analyze_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> RegimeMetrics:
        """🎯 Analyze current market regime"""
        # Use H1 data as primary, fallback to available timeframes
        df = market_data.get('H1', market_data.get('M15', list(market_data.values())[0]))
        
        # Calculate ATR and percentile
        atr = self._calculate_atr(df)
        atr_percentile = self._calculate_percentile(atr, self.regime_lookback)
        
        # Calculate Bollinger Bands width
        bb_upper, bb_lower = self._calculate_bollinger_bands(df)
        bb_width = (bb_upper - bb_lower) / df['close']
        bb_width_percentile = self._calculate_percentile(bb_width, self.regime_lookback)
        
        # Calculate EMA200 distance
        ema200 = df['close'].ewm(span=self.ema200_period).mean()
        ema200_distance = (df['close'].iloc[-1] - ema200.iloc[-1]) / ema200.iloc[-1]
        
        # Calculate HTF BOS frequency (using H4 if available)
        htf_df = market_data.get('H4', df)
        htf_bos_frequency = self._calculate_bos_frequency(htf_df)
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(df)
        
        # Determine regime categories
        volatility_regime = self._categorize_volatility(atr_percentile, bb_width_percentile)
        trend_regime = self._categorize_trend(trend_strength, ema200_distance)
        momentum_regime = self._categorize_momentum(df)
        
        return RegimeMetrics(
            atr_percentile=atr_percentile,
            bb_width_percentile=bb_width_percentile,
            htf_bos_frequency=htf_bos_frequency,
            ema200_distance=ema200_distance,
            trend_strength=trend_strength,
            volatility_regime=volatility_regime,
            trend_regime=trend_regime,
            momentum_regime=momentum_regime
        )
    
    def _determine_market_regime(self, metrics: RegimeMetrics) -> MarketRegime:
        """🎯 Determine overall market regime"""
        # Trending vs Ranging
        is_trending = metrics.trend_regime == 'trending'
        is_high_vol = metrics.volatility_regime == 'high'
        is_bullish = metrics.ema200_distance > 0.002  # 0.2% above EMA200
        is_bearish = metrics.ema200_distance < -0.002  # 0.2% below EMA200
        
        # Breakout detection
        is_breakout = metrics.htf_bos_frequency > 0.3 and metrics.atr_percentile > 70
        
        # Reversal zone detection
        is_reversal_zone = (
            metrics.atr_percentile > 80 and 
            abs(metrics.ema200_distance) > 0.01 and
            metrics.momentum_regime == 'weak'
        )
        
        # Consolidation detection
        is_consolidation = (
            metrics.volatility_regime == 'low' and
            abs(metrics.ema200_distance) < 0.005 and
            metrics.trend_strength < 0.3
        )
        
        # Determine regime
        if is_reversal_zone:
            return MarketRegime.REVERSAL_ZONE
        elif is_consolidation:
            return MarketRegime.CONSOLIDATION
        elif is_breakout:
            if is_bullish:
                return MarketRegime.BREAKOUT_BULLISH
            elif is_bearish:
                return MarketRegime.BREAKOUT_BEARISH
            else:
                return MarketRegime.RANGING_HIGH_VOL
        elif is_trending:
            if is_bullish:
                return MarketRegime.TRENDING_BULLISH
            elif is_bearish:
                return MarketRegime.TRENDING_BEARISH
            else:
                return MarketRegime.CONSOLIDATION
        else:
            if is_high_vol:
                return MarketRegime.RANGING_HIGH_VOL
            else:
                return MarketRegime.RANGING_LOW_VOL
    
    def _enhance_signal(self, signal: Dict, regime_metrics: RegimeMetrics, 
                       current_regime: MarketRegime, market_data: Dict[str, pd.DataFrame],
                       correlation_data: Dict[str, pd.DataFrame] = None) -> EnhancedSignal:
        """🎯 Enhance individual signal with adaptive weighting"""
        
        # Get base weight
        signal_type = signal.get('type', 'unknown')
        base_weight = self.base_weights.get(signal_type, 0.8)
        
        # Get signal direction
        signal_direction = signal.get('direction', 'neutral')
        if signal_direction.lower() in ['buy', 'bullish', 'long']:
            direction = 'bullish'
        elif signal_direction.lower() in ['sell', 'bearish', 'short']:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Calculate regime multiplier
        regime_multiplier = self.regime_multipliers.get(current_regime, {'bullish': 1.0, 'bearish': 1.0})
        regime_factor = regime_multiplier.get(direction, 1.0)
        
        # Calculate confluence bonus
        confluence_bonus = self._calculate_confluence_bonus(signal, market_data)
        
        # Calculate volatility penalty
        volatility_penalty = self._calculate_volatility_penalty(regime_metrics)
        
        # Calculate session bonus
        session_bonus = self._calculate_session_bonus(signal)
        
        # Calculate correlation factor
        correlation_factor = self._calculate_correlation_factor(signal, correlation_data)
        
        # Calculate final weight
        final_weight = (
            base_weight * 
            regime_factor * 
            (1 + confluence_bonus) * 
            (1 - volatility_penalty) * 
            (1 + session_bonus) * 
            correlation_factor
        )
        
        # Determine confidence level
        confidence = self._determine_confidence_level(final_weight)
        
        # Create signal weight object
        signal_weight = SignalWeight(
            base_weight=base_weight,
            regime_multiplier=regime_factor,
            confluence_bonus=confluence_bonus,
            volatility_penalty=volatility_penalty,
            session_bonus=session_bonus,
            correlation_factor=correlation_factor,
            final_weight=final_weight,
            confidence=confidence
        )
        
        # Determine recommended action
        recommended_action = self._determine_recommended_action(final_weight, current_regime)
        
        # Calculate risk adjustment
        risk_adjustment = self._calculate_risk_adjustment(regime_metrics, final_weight)
        
        # Determine entry timing
        entry_timing = self._determine_entry_timing(signal, current_regime, final_weight)
        
        return EnhancedSignal(
            original_signal=signal,
            signal_weight=signal_weight,
            regime_context=regime_metrics,
            recommended_action=recommended_action,
            risk_adjustment=risk_adjustment,
            entry_timing=entry_timing
        )
    
    def _calculate_confluence_bonus(self, signal: Dict, market_data: Dict[str, pd.DataFrame]) -> float:
        """🎯 Calculate multi-timeframe confluence bonus"""
        confluence_score = 0.0
        
        # Check for multi-timeframe alignment
        timeframes = list(market_data.keys())
        if len(timeframes) > 1:
            # Simple confluence check - could be enhanced with actual MTF analysis
            confluence_score += 0.1 * (len(timeframes) - 1)
        
        # Check signal strength
        signal_strength = signal.get('strength', 0.5)
        if signal_strength > 0.8:
            confluence_score += 0.2
        elif signal_strength > 0.6:
            confluence_score += 0.1
        
        # Check for multiple signal types at same level
        if signal.get('confluence_count', 1) > 1:
            confluence_score += 0.15
        
        return min(confluence_score, 0.5)  # Cap at 50% bonus
    
    def _calculate_volatility_penalty(self, regime_metrics: RegimeMetrics) -> float:
        """🎯 Calculate volatility-based penalty"""
        penalty = 0.0
        
        # High volatility penalty
        if regime_metrics.volatility_regime == 'high':
            penalty += 0.2
        
        # Extreme ATR penalty
        if regime_metrics.atr_percentile > 90:
            penalty += 0.15
        
        # Wide Bollinger Bands penalty
        if regime_metrics.bb_width_percentile > 85:
            penalty += 0.1
        
        return min(penalty, 0.4)  # Cap at 40% penalty
    
    def _calculate_session_bonus(self, signal: Dict) -> float:
        """🎯 Calculate session-based bonus with proper timezone alignment"""
        # Use proper timezone alignment for session detection
        try:
            from session_timezone_aligner import SessionTimezoneAligner
            aligner = SessionTimezoneAligner()
            current_session = aligner.get_current_session(datetime.now())
            
            # Map session types to session names
            session_mapping = {
                'LONDON': 'london',
                'NEW_YORK': 'new_york', 
                'OVERLAP': 'overlap',
                'ASIA': 'asia'
            }
            
            session = session_mapping.get(current_session.value.upper(), 'asia')
        except Exception:
            # Fallback to simplified logic if timezone aligner fails
            current_hour = datetime.now().hour
            if 8 <= current_hour < 16:  # London session
                session = 'london'
            elif 13 <= current_hour < 21:  # NY session
                session = 'new_york'
            elif 8 <= current_hour < 13:  # Overlap
                session = 'overlap'
            else:  # Asia session
                session = 'asia'
        
        session_weight = self.session_weights.get(session, 1.0)
        return (session_weight - 1.0)  # Convert to bonus (0.0 to 0.4)
    
    def _calculate_correlation_factor(self, signal: Dict, correlation_data: Dict[str, pd.DataFrame] = None) -> float:
        """🎯 Calculate correlation-based factor"""
        if not correlation_data:
            return 1.0
        
        correlation_factor = 1.0
        
        # Example: USDJPY correlation with US10Y
        if 'USDJPY' in signal.get('symbol', '') and 'US10Y' in correlation_data:
            # Simplified correlation logic
            # In reality, you'd calculate actual correlation and adjust accordingly
            correlation_factor *= 1.05  # Small positive adjustment
        
        # DXY correlation for USD pairs
        if 'USD' in signal.get('symbol', '') and 'DXY' in correlation_data:
            correlation_factor *= 1.03
        
        return correlation_factor
    
    def _determine_confidence_level(self, final_weight: float) -> SignalConfidence:
        """Determine signal confidence level"""
        if final_weight >= 1.5:
            return SignalConfidence.VERY_HIGH
        elif final_weight >= 1.2:
            return SignalConfidence.HIGH
        elif final_weight >= 0.8:
            return SignalConfidence.MEDIUM
        elif final_weight >= 0.5:
            return SignalConfidence.LOW
        else:
            return SignalConfidence.VERY_LOW
    
    def _determine_recommended_action(self, final_weight: float, regime: MarketRegime) -> str:
        """Determine recommended action for signal"""
        if final_weight >= 1.3:
            return 'take'
        elif final_weight >= 1.0:
            if regime in [MarketRegime.REVERSAL_ZONE, MarketRegime.RANGING_HIGH_VOL]:
                return 'reduce'
            else:
                return 'take'
        elif final_weight >= 0.7:
            return 'reduce'
        elif final_weight >= 0.4:
            return 'wait'
        else:
            return 'skip'
    
    def _calculate_risk_adjustment(self, regime_metrics: RegimeMetrics, final_weight: float) -> float:
        """Calculate suggested risk adjustment"""
        base_risk = 1.0
        
        # Volatility adjustment
        if regime_metrics.volatility_regime == 'high':
            base_risk *= 0.7
        elif regime_metrics.volatility_regime == 'low':
            base_risk *= 1.2
        
        # Weight adjustment
        if final_weight > 1.3:
            base_risk *= 1.1
        elif final_weight < 0.7:
            base_risk *= 0.6
        
        return base_risk
    
    def _determine_entry_timing(self, signal: Dict, regime: MarketRegime, final_weight: float) -> str:
        """Determine optimal entry timing"""
        if final_weight >= 1.4:
            return 'immediate'
        elif regime in [MarketRegime.RANGING_HIGH_VOL, MarketRegime.REVERSAL_ZONE]:
            return 'wait_confirmation'
        elif final_weight >= 1.0:
            return 'wait_pullback'
        else:
            return 'wait_confirmation'
    
    # Helper calculation methods
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=self.atr_period).mean()
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window=self.bb_period).mean()
        std = df['close'].rolling(window=self.bb_period).std()
        
        upper = sma + (std * self.bb_std)
        lower = sma - (std * self.bb_std)
        
        return upper, lower
    
    def _calculate_percentile(self, series: pd.Series, lookback: int) -> float:
        """Calculate percentile of current value in lookback period"""
        if len(series) < lookback:
            return 50.0
        
        recent_values = series.tail(lookback)
        current_value = series.iloc[-1]
        
        if pd.isna(current_value):
            return 50.0
        
        percentile = (recent_values < current_value).sum() / len(recent_values) * 100
        return percentile
    
    def _calculate_bos_frequency(self, df: pd.DataFrame) -> float:
        """Calculate Break of Structure frequency"""
        # Simplified BOS frequency calculation
        # In practice, this would use actual BOS detection logic
        
        if len(df) < 20:
            return 0.0
        
        # Use price volatility as proxy for BOS frequency
        price_changes = df['close'].pct_change().abs()
        significant_moves = (price_changes > price_changes.quantile(0.8)).sum()
        
        return significant_moves / len(df)
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength"""
        if len(df) < 20:
            return 0.0
        
        # Simple trend strength using price position relative to moving averages
        sma20 = df['close'].rolling(window=20).mean()
        sma50 = df['close'].rolling(window=50).mean() if len(df) >= 50 else sma20
        
        current_price = df['close'].iloc[-1]
        
        if pd.isna(sma20.iloc[-1]) or pd.isna(sma50.iloc[-1]):
            return 0.0
        
        # Trend strength based on MA alignment and price position
        ma_alignment = 1.0 if sma20.iloc[-1] > sma50.iloc[-1] else -1.0
        price_position = (current_price - sma20.iloc[-1]) / sma20.iloc[-1]
        
        return abs(price_position) * ma_alignment
    
    def _categorize_volatility(self, atr_percentile: float, bb_width_percentile: float) -> str:
        """Categorize volatility regime"""
        avg_percentile = (atr_percentile + bb_width_percentile) / 2
        
        if avg_percentile > 75:
            return 'high'
        elif avg_percentile > 25:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_trend(self, trend_strength: float, ema200_distance: float) -> str:
        """Categorize trend regime"""
        if abs(trend_strength) > 0.4 and abs(ema200_distance) > 0.005:
            return 'trending'
        else:
            return 'ranging'
    
    def _categorize_momentum(self, df: pd.DataFrame) -> str:
        """Categorize momentum regime"""
        if len(df) < 10:
            return 'neutral'
        
        # Simple momentum using recent price changes
        recent_changes = df['close'].pct_change().tail(5)
        avg_change = recent_changes.mean()
        
        if abs(avg_change) > 0.002:  # 0.2% average change
            return 'strong'
        elif abs(avg_change) > 0.0005:  # 0.05% average change
            return 'weak'
        else:
            return 'neutral'
    
    def _create_basic_enhanced_signals(self, signals: List[Dict]) -> List[EnhancedSignal]:
        """Create basic enhanced signals when full analysis fails"""
        enhanced_signals = []
        
        for signal in signals:
            basic_weight = SignalWeight(
                base_weight=0.8,
                regime_multiplier=1.0,
                confluence_bonus=0.0,
                volatility_penalty=0.0,
                session_bonus=0.0,
                correlation_factor=1.0,
                final_weight=0.8,
                confidence=SignalConfidence.MEDIUM
            )
            
            basic_metrics = RegimeMetrics(
                atr_percentile=50.0,
                bb_width_percentile=50.0,
                htf_bos_frequency=0.1,
                ema200_distance=0.0,
                trend_strength=0.0,
                volatility_regime='medium',
                trend_regime='ranging',
                momentum_regime='neutral'
            )
            
            enhanced_signal = EnhancedSignal(
                original_signal=signal,
                signal_weight=basic_weight,
                regime_context=basic_metrics,
                recommended_action='take',
                risk_adjustment=1.0,
                entry_timing='immediate'
            )
            
            enhanced_signals.append(enhanced_signal)
        
        return enhanced_signals
    
    def get_regime_summary(self, regime_metrics: RegimeMetrics, current_regime: MarketRegime) -> Dict:
        """Get summary of current market regime"""
        return {
            'current_regime': current_regime.value,
            'volatility_regime': regime_metrics.volatility_regime,
            'trend_regime': regime_metrics.trend_regime,
            'momentum_regime': regime_metrics.momentum_regime,
            'atr_percentile': regime_metrics.atr_percentile,
            'bb_width_percentile': regime_metrics.bb_width_percentile,
            'trend_strength': regime_metrics.trend_strength,
            'ema200_distance_pct': regime_metrics.ema200_distance * 100
        }