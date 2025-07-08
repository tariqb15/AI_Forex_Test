#!/usr/bin/env python3
"""
Signal Conflict Resolver for BLOB AI Trading System

This module resolves conflicting BUY/SELL signals for the same currency pair
by implementing advanced signal consolidation and conflict resolution logic.

Author: BLOB AI Trading System
Version: 1.0.0
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Import trading signal classes
try:
    from forex_engine import TradingSignal, SignalStrength
except ImportError:
    # Fallback for testing
    class TradingSignal:
        pass
    class SignalStrength:
        WEAK = "WEAK"
        MODERATE = "MODERATE"
        STRONG = "STRONG"

logger = logging.getLogger(__name__)

@dataclass
class SignalConflictResult:
    """Result of signal conflict resolution"""
    resolved_signals: List[TradingSignal]
    conflicts_found: int
    conflicts_resolved: int
    consolidation_applied: bool
    reasoning: str

class SignalConflictResolver:
    """Advanced signal conflict resolution system"""
    
    def __init__(self, config=None):
        self.config = config
        self.min_signal_separation_minutes = 15  # Minimum time between signals for same pair
        self.confidence_threshold = 0.6  # Minimum confidence for signal consideration
        self.consolidation_weight_threshold = 0.3  # Threshold for signal consolidation
        
        # Signal quality weights
        self.signal_type_weights = {
            'ORDER_BLOCK_RETEST': 0.9,
            'FAIR_VALUE_GAP': 0.8,
            'STRUCTURE_BREAK': 0.85,
            'VOLATILITY_BREAKOUT': 0.7,
            'SESSION_MOMENTUM': 0.75,
            'SIMPLIFIED': 0.5
        }
        
        # Timeframe importance weights
        self.timeframe_weights = {
            'D1': 1.0,
            'H4': 0.9,
            'H1': 0.8,
            'M15': 0.7,
            'M5': 0.6,
            'M1': 0.5
        }
    
    def resolve_signal_conflicts(self, signals: List[TradingSignal]) -> SignalConflictResult:
        """Main method to resolve signal conflicts"""
        logger.info(f"🔍 Analyzing {len(signals)} signals for conflicts...")
        
        if not signals:
            return SignalConflictResult(
                resolved_signals=[],
                conflicts_found=0,
                conflicts_resolved=0,
                consolidation_applied=False,
                reasoning="No signals to analyze"
            )
        
        # Group signals by currency pair
        signals_by_pair = self._group_signals_by_pair(signals)
        
        resolved_signals = []
        total_conflicts = 0
        total_resolved = 0
        consolidation_applied = False
        
        for pair, pair_signals in signals_by_pair.items():
            logger.info(f"📊 Analyzing {len(pair_signals)} signals for {pair}")
            
            # Detect conflicts for this pair
            conflicts = self._detect_conflicts(pair_signals)
            total_conflicts += len(conflicts)
            
            if conflicts:
                logger.warning(f"⚠️ Found {len(conflicts)} conflicts for {pair}")
                
                # Resolve conflicts
                pair_resolved = self._resolve_pair_conflicts(pair, pair_signals, conflicts)
                resolved_signals.extend(pair_resolved.resolved_signals)
                total_resolved += pair_resolved.conflicts_resolved
                
                if pair_resolved.consolidation_applied:
                    consolidation_applied = True
            else:
                # No conflicts, but still apply quality filtering
                filtered_signals = self._apply_quality_filter(pair_signals)
                resolved_signals.extend(filtered_signals)
                logger.info(f"✅ No conflicts for {pair}, applied quality filter")
        
        # Final ranking and selection
        final_signals = self._rank_and_select_final_signals(resolved_signals)
        
        reasoning = self._generate_resolution_reasoning(total_conflicts, total_resolved, consolidation_applied)
        
        return SignalConflictResult(
            resolved_signals=final_signals,
            conflicts_found=total_conflicts,
            conflicts_resolved=total_resolved,
            consolidation_applied=consolidation_applied,
            reasoning=reasoning
        )
    
    def _group_signals_by_pair(self, signals: List[TradingSignal]) -> Dict[str, List[TradingSignal]]:
        """Group signals by currency pair"""
        signals_by_pair = defaultdict(list)
        
        for signal in signals:
            # Get currency pair from signal
            pair = getattr(signal, 'currency_pair', None) or getattr(signal, 'symbol', 'UNKNOWN')
            signals_by_pair[pair].append(signal)
        
        return dict(signals_by_pair)
    
    def _detect_conflicts(self, signals: List[TradingSignal]) -> List[Tuple[TradingSignal, TradingSignal]]:
        """Detect conflicting BUY/SELL signals for the same pair"""
        conflicts = []
        
        for i, signal1 in enumerate(signals):
            for j, signal2 in enumerate(signals[i+1:], i+1):
                if self._are_conflicting(signal1, signal2):
                    conflicts.append((signal1, signal2))
        
        return conflicts
    
    def _are_conflicting(self, signal1: TradingSignal, signal2: TradingSignal) -> bool:
        """Check if two signals are conflicting"""
        # Different directions = conflict
        if signal1.direction != signal2.direction:
            # Check if they're close in time (within separation threshold)
            time_diff = abs((signal1.timestamp - signal2.timestamp).total_seconds() / 60)
            if time_diff <= self.min_signal_separation_minutes:
                return True
        
        return False
    
    def _resolve_pair_conflicts(self, pair: str, signals: List[TradingSignal], conflicts: List[Tuple]) -> SignalConflictResult:
        """Resolve conflicts for a specific currency pair"""
        logger.info(f"🔧 Resolving {len(conflicts)} conflicts for {pair}")
        
        # Try consolidation first
        consolidation_result = self._attempt_signal_consolidation(signals)
        
        if consolidation_result.consolidation_applied:
            logger.info(f"✅ Successfully consolidated signals for {pair}")
            return consolidation_result
        
        # If consolidation fails, use conflict resolution strategies
        resolved_signals = self._apply_conflict_resolution_strategies(signals, conflicts)
        
        return SignalConflictResult(
            resolved_signals=resolved_signals,
            conflicts_found=len(conflicts),
            conflicts_resolved=len(conflicts),
            consolidation_applied=False,
            reasoning=f"Applied conflict resolution strategies for {pair}"
        )
    
    def _attempt_signal_consolidation(self, signals: List[TradingSignal]) -> SignalConflictResult:
        """Attempt to consolidate multiple signals into a stronger single signal"""
        if len(signals) < 2:
            return SignalConflictResult(
                resolved_signals=signals,
                conflicts_found=0,
                conflicts_resolved=0,
                consolidation_applied=False,
                reasoning="Insufficient signals for consolidation"
            )
        
        # Group by direction
        buy_signals = [s for s in signals if s.direction == 'BUY']
        sell_signals = [s for s in signals if s.direction == 'SELL']
        
        consolidated_signals = []
        
        # Consolidate BUY signals if multiple exist
        if len(buy_signals) > 1:
            consolidated_buy = self._consolidate_same_direction_signals(buy_signals, 'BUY')
            if consolidated_buy:
                consolidated_signals.append(consolidated_buy)
        elif len(buy_signals) == 1:
            consolidated_signals.extend(buy_signals)
        
        # Consolidate SELL signals if multiple exist
        if len(sell_signals) > 1:
            consolidated_sell = self._consolidate_same_direction_signals(sell_signals, 'SELL')
            if consolidated_sell:
                consolidated_signals.append(consolidated_sell)
        elif len(sell_signals) == 1:
            consolidated_signals.extend(sell_signals)
        
        # If we have both BUY and SELL after consolidation, choose the stronger one
        if len(consolidated_signals) > 1:
            final_signal = self._choose_strongest_signal(consolidated_signals)
            consolidated_signals = [final_signal] if final_signal else []
        
        return SignalConflictResult(
            resolved_signals=consolidated_signals,
            conflicts_found=len(signals) - len(consolidated_signals),
            conflicts_resolved=len(signals) - len(consolidated_signals),
            consolidation_applied=len(consolidated_signals) < len(signals),
            reasoning="Signal consolidation applied"
        )
    
    def _consolidate_same_direction_signals(self, signals: List[TradingSignal], direction: str) -> Optional[TradingSignal]:
        """Consolidate multiple signals of the same direction into one stronger signal"""
        if not signals:
            return None
        
        if len(signals) == 1:
            return signals[0]
        
        # Calculate weighted averages and combined strength
        total_weight = 0
        weighted_confidence = 0
        weighted_risk_reward = 0
        weighted_entry = 0
        weighted_sl = 0
        weighted_tp = 0
        
        signal_types = []
        reasoning_parts = []
        
        for signal in signals:
            # Calculate signal weight based on type and confidence
            type_weight = self.signal_type_weights.get(signal.signal_type, 0.5)
            confidence_weight = getattr(signal, 'confidence', 0.5)
            weight = type_weight * confidence_weight
            
            total_weight += weight
            weighted_confidence += confidence_weight * weight
            weighted_risk_reward += getattr(signal, 'risk_reward', 1.0) * weight
            weighted_entry += signal.entry_price * weight
            weighted_sl += signal.stop_loss * weight
            weighted_tp += signal.take_profit * weight
            
            signal_types.append(signal.signal_type)
            reasoning_parts.append(getattr(signal, 'reasoning', 'Signal'))
        
        if total_weight == 0:
            return signals[0]  # Fallback to first signal
        
        # Create consolidated signal
        base_signal = signals[0]  # Use first signal as template
        
        # Create new consolidated signal
        consolidated_signal = TradingSignal(
            timestamp=datetime.now(),
            signal_type=f"CONSOLIDATED_{'+'.join(set(signal_types))}",
            direction=direction,
            strength=SignalStrength.STRONG,  # Consolidated signals are stronger
            entry_price=weighted_entry / total_weight,
            stop_loss=weighted_sl / total_weight,
            take_profit=weighted_tp / total_weight,
            risk_reward=weighted_risk_reward / total_weight,
            confidence=min(weighted_confidence / total_weight * 1.2, 1.0),  # Boost confidence
            reasoning=f"Consolidated from {len(signals)} signals: {', '.join(reasoning_parts)}",
            session=getattr(base_signal, 'session', 'UNKNOWN'),
            market_structure=getattr(base_signal, 'market_structure', 'UNKNOWN'),
            symbol=getattr(base_signal, 'symbol', getattr(base_signal, 'currency_pair', 'UNKNOWN'))
        )
        
        # Copy currency_pair if it exists
        if hasattr(base_signal, 'currency_pair'):
            consolidated_signal.currency_pair = base_signal.currency_pair
        
        logger.info(f"✅ Consolidated {len(signals)} {direction} signals into one with confidence {consolidated_signal.confidence:.2f}")
        
        return consolidated_signal
    
    def _apply_conflict_resolution_strategies(self, signals: List[TradingSignal], conflicts: List[Tuple]) -> List[TradingSignal]:
        """Apply various strategies to resolve signal conflicts"""
        # Strategy 1: Choose highest confidence signal
        if len(signals) == 2:
            signal1, signal2 = signals[0], signals[1]
            conf1 = getattr(signal1, 'confidence', 0.5)
            conf2 = getattr(signal2, 'confidence', 0.5)
            
            if abs(conf1 - conf2) > 0.1:  # Significant confidence difference
                winner = signal1 if conf1 > conf2 else signal2
                logger.info(f"🎯 Chose {winner.direction} signal with confidence {getattr(winner, 'confidence', 0.5):.2f}")
                return [winner]
        
        # Strategy 2: Choose based on signal type priority
        prioritized = self._prioritize_by_signal_type(signals)
        if prioritized:
            logger.info(f"🎯 Chose {prioritized.direction} signal based on type priority: {prioritized.signal_type}")
            return [prioritized]
        
        # Strategy 3: Choose based on risk/reward ratio
        best_rr = max(signals, key=lambda s: getattr(s, 'risk_reward', 1.0))
        logger.info(f"🎯 Chose {best_rr.direction} signal with best R:R {getattr(best_rr, 'risk_reward', 1.0):.2f}")
        return [best_rr]
    
    def _prioritize_by_signal_type(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Choose signal based on signal type priority"""
        best_signal = None
        best_weight = 0
        
        for signal in signals:
            weight = self.signal_type_weights.get(signal.signal_type, 0.5)
            if weight > best_weight:
                best_weight = weight
                best_signal = signal
        
        return best_signal
    
    def _choose_strongest_signal(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Choose the strongest signal from a list"""
        if not signals:
            return None
        
        # Calculate composite score for each signal
        best_signal = None
        best_score = 0
        
        for signal in signals:
            confidence = getattr(signal, 'confidence', 0.5)
            risk_reward = getattr(signal, 'risk_reward', 1.0)
            type_weight = self.signal_type_weights.get(signal.signal_type, 0.5)
            
            # Composite score: confidence * risk_reward * type_weight
            score = confidence * min(risk_reward, 3.0) * type_weight  # Cap R:R at 3.0
            
            if score > best_score:
                best_score = score
                best_signal = signal
        
        return best_signal
    
    def _apply_quality_filter(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Apply quality filter to signals"""
        filtered = []
        
        for signal in signals:
            confidence = getattr(signal, 'confidence', 0.5)
            risk_reward = getattr(signal, 'risk_reward', 1.0)
            
            # Quality criteria
            if (confidence >= self.confidence_threshold and 
                risk_reward >= 1.0 and 
                signal.entry_price > 0 and 
                signal.stop_loss > 0 and 
                signal.take_profit > 0):
                filtered.append(signal)
            else:
                logger.info(f"🚫 Filtered low-quality signal: conf={confidence:.2f}, R:R={risk_reward:.2f}")
        
        return filtered
    
    def _rank_and_select_final_signals(self, signals: List[TradingSignal], max_signals: int = 5) -> List[TradingSignal]:
        """Final ranking and selection of signals"""
        if not signals:
            return []
        
        # Sort by composite score
        ranked_signals = sorted(signals, key=self._calculate_signal_score, reverse=True)
        
        # Select top signals
        selected = ranked_signals[:max_signals]
        
        logger.info(f"📊 Selected {len(selected)} final signals from {len(signals)} candidates")
        
        return selected
    
    def _calculate_signal_score(self, signal: TradingSignal) -> float:
        """Calculate composite score for signal ranking"""
        confidence = getattr(signal, 'confidence', 0.5)
        risk_reward = getattr(signal, 'risk_reward', 1.0)
        type_weight = self.signal_type_weights.get(signal.signal_type, 0.5)
        
        # Bonus for consolidated signals
        consolidation_bonus = 1.2 if 'CONSOLIDATED' in signal.signal_type else 1.0
        
        return confidence * min(risk_reward, 3.0) * type_weight * consolidation_bonus
    
    def _generate_resolution_reasoning(self, conflicts_found: int, conflicts_resolved: int, consolidation_applied: bool) -> str:
        """Generate human-readable reasoning for the resolution process"""
        parts = []
        
        if conflicts_found > 0:
            parts.append(f"Found {conflicts_found} signal conflicts")
            parts.append(f"Resolved {conflicts_resolved} conflicts")
        else:
            parts.append("No signal conflicts detected")
        
        if consolidation_applied:
            parts.append("Applied signal consolidation")
        
        parts.append("Applied quality filtering and ranking")
        
        return ". ".join(parts) + "."