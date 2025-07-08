#!/usr/bin/env python3
"""
Edge Case Handler for BLOB AI Trading System
Handles complex trading scenarios and edge cases to prevent false signals

Author: BLOB AI Trading System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

class EdgeCaseType(Enum):
    NO_LIQUIDITY_SWEEP_BEFORE_BOS = "no_liquidity_sweep_before_bos"
    ASIA_SESSION_FAKEOUT = "asia_session_fakeout"
    STACKED_ORDER_BLOCKS = "stacked_order_blocks"
    FVG_RE_ENTRY = "fvg_re_entry"
    NEWS_SPIKE_VOLATILITY = "news_spike_volatility"
    WEEKEND_GAP_BOS = "weekend_gap_bos"
    CONTINUATION_WITHOUT_RETEST = "continuation_without_retest"
    HTF_CONSOLIDATION_TRAP = "htf_consolidation_trap"
    WEAK_CHOCH_REVERSAL = "weak_choch_reversal"
    NO_MITIGATION_WIPEOUT = "no_mitigation_wipeout"

@dataclass
class EdgeCaseResult:
    case_type: EdgeCaseType
    is_valid: bool
    confidence_adjustment: float  # -1.0 to 1.0
    reasoning: str
    action: str
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []  # "BLOCK", "WAIT", "PROCEED", "MODIFY"

class EdgeCaseHandler:
    """
    Comprehensive edge case handler for trading scenarios
    Implements sophisticated logic to handle complex market conditions
    """
    
    def __init__(self):
        self.session_times = {
            'ASIA': [(0, 9)],  # UTC hours
            'LONDON': [(8, 17)],
            'NEW_YORK': [(13, 22)],
            'OVERLAP': [(13, 17)]
        }
        self.news_impact_threshold = 3.0  # ATR multiplier for high impact
        self.weekend_gap_threshold = 0.5  # % gap threshold
        self.htf_range_threshold = 0.8  # HTF range position threshold
        
    def validate_signal(self, signal_data: Dict, market_data: Dict, context: Dict) -> EdgeCaseResult:
        """
        Main validation function that checks all edge cases
        
        Args:
            signal_data: Trading signal information
            market_data: Current market data and analysis
            context: Additional context (session, news, etc.)
        
        Returns:
            EdgeCaseResult with validation outcome
        """
        # Ensure market_data is a dictionary
        if not isinstance(market_data, dict):
            logger.warning(f"market_data is not a dictionary: {type(market_data)} - {market_data}")
            market_data = {}
            
        # Ensure signal_data is a dictionary
        if not isinstance(signal_data, dict):
            logger.warning(f"signal_data is not a dictionary: {type(signal_data)} - {signal_data}")
            signal_data = {}
            
        # Ensure context is a dictionary
        if not isinstance(context, dict):
            logger.warning(f"context is not a dictionary: {type(context)} - {context}")
            context = {}
        
        # Run all edge case checks
        edge_cases = [
            self._check_liquidity_sweep_before_bos,
            self._check_asia_session_fakeout,
            self._check_stacked_order_blocks,
            self._check_fvg_re_entry,
            self._check_news_spike_volatility,
            self._check_weekend_gap_bos,
            self._check_continuation_without_retest,
            self._check_htf_consolidation_trap,
            self._check_weak_choch_reversal,
            self._check_no_mitigation_wipeout
        ]
        
        results = []
        for check_func in edge_cases:
            try:
                result = check_func(signal_data, market_data, context)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Edge case check failed: {check_func.__name__}: {e}")
        
        # Determine final outcome
        return self._consolidate_results(results)
    
    def _check_liquidity_sweep_before_bos(self, signal_data: Dict, market_data: Dict, context: Dict) -> Optional[EdgeCaseResult]:
        """
        Scenario 1: BOS happens but no liquidity was swept before
        Problem: Weak confirmation
        Solution: No entry — wait for sweep before BOS
        """
        structure_breaks = market_data.get('structure_breaks', [])
        liquidity_sweeps = market_data.get('liquidity_sweeps', [])
        
        if not structure_breaks:
            return None
            
        # Get the most recent BOS
        recent_bos = structure_breaks[-1] if structure_breaks else None
        if not recent_bos:
            return None
            
        bos_time = recent_bos.get('timestamp')
        if not bos_time:
            return None
            
        # Check for liquidity sweeps before this BOS (within last 10 candles)
        sweep_before_bos = False
        for sweep in liquidity_sweeps:
            sweep_time = sweep.get('timestamp')
            if sweep_time and sweep_time < bos_time:
                # Check if sweep was recent enough (within reasonable timeframe)
                time_diff = (bos_time - sweep_time).total_seconds() / 60  # minutes
                if time_diff <= 240:  # Within 4 hours
                    sweep_before_bos = True
                    break
        
        if not sweep_before_bos:
            return EdgeCaseResult(
                case_type=EdgeCaseType.NO_LIQUIDITY_SWEEP_BEFORE_BOS,
                is_valid=False,
                confidence_adjustment=-0.8,
                reasoning="BOS occurred without prior liquidity sweep - weak confirmation",
                action="WAIT"
            )
        
        return None
    
    def _check_asia_session_fakeout(self, signal_data: Dict, market_data: Dict, context: Dict) -> Optional[EdgeCaseResult]:
        """
        Scenario 2: Price taps OB during Asia session, moves away
        Problem: Low liquidity trap
        Solution: Wait for London confirmation; ignore Asia fakeout
        """
        current_session = context.get('current_session', '')
        signal_timestamp = signal_data.get('timestamp', datetime.now())
        
        # Check if signal occurred during Asia session
        if current_session == 'ASIA' or self._is_asia_session(signal_timestamp):
            # Check if this is an order block tap
            if signal_data.get('signal_type') in ['SMC_OrderBlock', 'Order Block Support', 'Order Block Resistance']:
                # Check for immediate rejection (price moving away from OB)
                current_price = market_data.get('current_price', {})
                entry_price = signal_data.get('entry_price', 0)
                
                if current_price and entry_price:
                    price_movement = abs(current_price.get('bid', entry_price) - entry_price)
                    atr = market_data.get('atr', 0.001)
                    
                    # If price moved away significantly during Asia
                    if price_movement > (atr * 0.5):
                        return EdgeCaseResult(
                            case_type=EdgeCaseType.ASIA_SESSION_FAKEOUT,
                            is_valid=False,
                            confidence_adjustment=-0.7,
                            reasoning="Asia session OB tap with immediate rejection - likely fakeout",
                            action="WAIT"
                        )
        
        return None
    
    def _check_stacked_order_blocks(self, signal_data: Dict, market_data: Dict, context: Dict) -> Optional[EdgeCaseResult]:
        """
        Scenario 3: Multiple OBs stacked – which one to trade?
        Problem: Zone confusion
        Solution: Prioritize OB with FVG + BOS; mitigate top-down
        """
        order_blocks = market_data.get('order_blocks', [])
        fair_value_gaps = market_data.get('fair_value_gaps', [])
        structure_breaks = market_data.get('structure_breaks', [])
        
        # Ensure all inputs are proper data structures
        if not isinstance(order_blocks, list):
            logger.warning(f"Order blocks is not a list: {type(order_blocks)} - {order_blocks}")
            return None
            
        # Handle fair_value_gaps dictionary structure
        if isinstance(fair_value_gaps, dict):
            fair_value_gaps = fair_value_gaps.get('all_fvgs', [])
        if not isinstance(fair_value_gaps, list):
            logger.warning(f"Fair value gaps is not a list: {type(fair_value_gaps)} - {fair_value_gaps}")
            fair_value_gaps = []
            
        if not isinstance(structure_breaks, list):
            logger.warning(f"Structure breaks is not a list: {type(structure_breaks)} - {structure_breaks}")
            structure_breaks = []
        
        if len(order_blocks) < 2:
            return None
            
        current_price = market_data.get('current_price', {}).get('bid', 0)
        
        # Find OBs near current price (within 1% range)
        nearby_obs = []
        for ob in order_blocks:
            # Check if ob is a dictionary (not a string)
            if not isinstance(ob, dict):
                logger.warning(f"Order block is not a dictionary: {type(ob)} - {ob}")
                continue
                
            ob_center = (ob.get('high', 0) + ob.get('low', 0)) / 2
            if current_price > 0 and abs(current_price - ob_center) / current_price < 0.01:  # Within 1%
                nearby_obs.append(ob)
            elif current_price == 0:  # Fallback if no current price
                nearby_obs.append(ob)
        
        if len(nearby_obs) > 1:
            # Prioritize OB with best confluence
            best_ob = self._select_best_order_block(nearby_obs, fair_value_gaps, structure_breaks)
            
            # Check if current signal is using the best OB
            signal_ob_level = signal_data.get('entry_price', 0)
            best_ob_range = (best_ob.get('low', 0), best_ob.get('high', 0))
            
            if not (best_ob_range[0] <= signal_ob_level <= best_ob_range[1]):
                return EdgeCaseResult(
                    case_type=EdgeCaseType.STACKED_ORDER_BLOCKS,
                    is_valid=False,
                    confidence_adjustment=-0.6,
                    reasoning="Multiple stacked OBs - signal not using highest confluence zone",
                    action="MODIFY"
                )
        
        return None
    
    def _check_fvg_re_entry(self, signal_data: Dict, market_data: Dict, context: Dict) -> Optional[EdgeCaseResult]:
        """
        Scenario 4: FVG is mitigated once, price returns again
        Problem: Old vs new logic
        Solution: Skip re-entry unless new BOS/sweep occurs
        """
        if signal_data.get('signal_type') != 'SMC_FairValueGap':
            return None
            
        fair_value_gaps = market_data.get('fair_value_gaps', [])
        structure_breaks = market_data.get('structure_breaks', [])
        liquidity_sweeps = market_data.get('liquidity_sweeps', [])
        
        # Find the FVG being traded
        signal_price = signal_data.get('entry_price', 0)
        target_fvg = None
        
        for fvg in fair_value_gaps:
            if fvg.get('bottom', 0) <= signal_price <= fvg.get('top', 0):
                target_fvg = fvg
                break
        
        if target_fvg and target_fvg.get('bounce_history', 0) > 0:
            # FVG has been touched before - check for new confirmation
            fvg_timestamp = target_fvg.get('timestamp')
            
            # Look for new BOS or sweep after last FVG touch
            new_confirmation = False
            for event in structure_breaks + liquidity_sweeps:
                event_time = event.get('timestamp')
                if event_time and fvg_timestamp and event_time > fvg_timestamp:
                    new_confirmation = True
                    break
            
            if not new_confirmation:
                return EdgeCaseResult(
                    case_type=EdgeCaseType.FVG_RE_ENTRY,
                    is_valid=False,
                    confidence_adjustment=-0.5,
                    reasoning="FVG re-entry without new BOS/sweep confirmation",
                    action="BLOCK"
                )
        
        return None
    
    def _check_news_spike_volatility(self, signal_data: Dict, market_data: Dict, context: Dict) -> Optional[EdgeCaseResult]:
        """
        Scenario 5: NFP spike causes BOS + sweep instantly
        Problem: Extreme volatility
        Solution: Avoid entry; filter out high-news candles
        """
        # Check for extreme volatility conditions
        volatility_data = market_data.get('volatility_analysis', {})
        current_volatility = volatility_data.get('current_volatility_percentile', 0.5)
        atr_ratio = volatility_data.get('atr_ratio', 1.0)
        
        # Check for news events
        news_events = context.get('news_events', [])
        high_impact_news = any(event.get('impact', '') == 'HIGH' for event in news_events)
        
        # Extreme volatility conditions
        if (current_volatility > 0.95 or atr_ratio > self.news_impact_threshold) and high_impact_news:
            return EdgeCaseResult(
                case_type=EdgeCaseType.NEWS_SPIKE_VOLATILITY,
                is_valid=False,
                confidence_adjustment=-0.9,
                reasoning="Extreme volatility during high-impact news - avoid entry",
                action="BLOCK"
            )
        
        # Check for abnormally large candles
        recent_candles = market_data.get('recent_candles', [])
        if recent_candles:
            latest_candle = recent_candles[-1]
            candle_range = latest_candle.get('high', 0) - latest_candle.get('low', 0)
            avg_range = np.mean([c.get('high', 0) - c.get('low', 0) for c in recent_candles[-10:]])
            
            if candle_range > (avg_range * 3.0):  # 3x average range
                return EdgeCaseResult(
                    case_type=EdgeCaseType.NEWS_SPIKE_VOLATILITY,
                    is_valid=False,
                    confidence_adjustment=-0.7,
                    reasoning="Abnormally large candle detected - potential news spike",
                    action="WAIT"
                )
        
        return None
    
    def _check_weekend_gap_bos(self, signal_data: Dict, market_data: Dict, context: Dict) -> Optional[EdgeCaseResult]:
        """
        Scenario 6: Price breaks structure during weekend gaps
        Problem: Illiquid move
        Solution: Ignore BOS caused by weekend gaps or no volume
        """
        signal_timestamp = signal_data.get('timestamp', datetime.now())
        
        # Check if signal occurred during weekend or market open gap
        if self._is_weekend_gap_period(signal_timestamp):
            return EdgeCaseResult(
                case_type=EdgeCaseType.WEEKEND_GAP_BOS,
                is_valid=False,
                confidence_adjustment=-0.8,
                reasoning="BOS occurred during weekend gap - illiquid conditions",
                action="BLOCK"
            )
        
        # Check for gap conditions in price data
        recent_candles = market_data.get('recent_candles', [])
        if len(recent_candles) >= 2:
            current_candle = recent_candles[-1]
            previous_candle = recent_candles[-2]
            
            gap_size = abs(current_candle.get('open', 0) - previous_candle.get('close', 0))
            avg_range = np.mean([c.get('high', 0) - c.get('low', 0) for c in recent_candles[-10:]])
            
            if gap_size > (avg_range * self.weekend_gap_threshold):
                return EdgeCaseResult(
                    case_type=EdgeCaseType.WEEKEND_GAP_BOS,
                    is_valid=False,
                    confidence_adjustment=-0.6,
                    reasoning="Large price gap detected - potentially illiquid move",
                    action="WAIT"
                )
        
        return None
    
    def _check_continuation_without_retest(self, signal_data: Dict, market_data: Dict, context: Dict) -> Optional[EdgeCaseResult]:
        """
        Scenario 7: Trend continues after BOS but no OB retest
        Problem: Missed trade
        Solution: Bot must allow continuation entry if trend structure remains valid
        """
        structure_breaks = market_data.get('structure_breaks', [])
        trend_context = market_data.get('trend_context', {})
        signal_type = signal_data.get('signal_type', '')
        
        # Only apply to breakout/continuation signals
        if 'breakout' not in signal_type.lower() and 'continuation' not in signal_type.lower():
            return None
            
        if not structure_breaks:
            return None
            
        recent_bos = structure_breaks[-1]
        trend_phase = trend_context.get('phase', 'unknown')
        momentum_strength = trend_context.get('momentum_strength', 0.5)  # Default moderate
        
        # More lenient conditions for continuation trades
        strong_trend = (trend_phase in ['early', 'mid', 'strong'] and 
                       momentum_strength > 0.5 and 
                       recent_bos.get('strength', 0.5) > 0.5)
        
        # Also check for multiple consecutive BOS (strong momentum)
        consecutive_bos = len([sb for sb in structure_breaks[-3:] if sb.get('strength', 0) > 0.4])
        
        if strong_trend or consecutive_bos >= 2:
            # Allow continuation entry with adjusted confidence
            return EdgeCaseResult(
                case_type=EdgeCaseType.CONTINUATION_WITHOUT_RETEST,
                is_valid=True,
                confidence_adjustment=0.3,
                reasoning="Strong trend continuation - allow entry without retest",
                action="PROCEED"
            )
        
        return None
    
    def _check_htf_consolidation_trap(self, signal_data: Dict, market_data: Dict, context: Dict) -> Optional[EdgeCaseResult]:
        """
        Scenario 8: Sweep + BOS happen but OB is in middle of HTF range
        Problem: Inside consolidation
        Solution: Filter trades within HTF chop zone
        """
        htf_analysis = context.get('htf_analysis', {})
        current_price = market_data.get('current_price', {}).get('bid', 0)
        
        # Check HTF range conditions
        htf_high = htf_analysis.get('range_high', 0)
        htf_low = htf_analysis.get('range_low', 0)
        htf_structure = htf_analysis.get('market_structure', '')
        
        if htf_high and htf_low and htf_structure == 'RANGING':
            htf_range = htf_high - htf_low
            price_position = (current_price - htf_low) / htf_range
            
            # If price is in middle 60% of HTF range
            if 0.2 < price_position < 0.8:
                return EdgeCaseResult(
                    case_type=EdgeCaseType.HTF_CONSOLIDATION_TRAP,
                    is_valid=False,
                    confidence_adjustment=-0.6,
                    reasoning="Signal within HTF consolidation zone - low probability",
                    action="BLOCK"
                )
        
        return None
    
    def _check_weak_choch_reversal(self, signal_data: Dict, market_data: Dict, context: Dict) -> Optional[EdgeCaseResult]:
        """
        Scenario 9: Entry after ChoCH but no follow-through
        Problem: Weak reversal
        Solution: Apply confirmation candles or internal structure BOS
        """
        choch_signals = market_data.get('choch_signals', [])
        structure_breaks = market_data.get('structure_breaks', [])
        
        if not choch_signals:
            return None
            
        recent_choch = choch_signals[-1]
        choch_strength = recent_choch.get('strength', 0)
        
        # Check for weak ChoCH
        if choch_strength < 0.5:
            # Look for additional confirmation
            confirmation_found = False
            
            # Check for follow-up structure breaks
            choch_time = recent_choch.get('timestamp')
            for sb in structure_breaks:
                sb_time = sb.get('timestamp')
                if sb_time and choch_time and sb_time > choch_time:
                    confirmation_found = True
                    break
            
            if not confirmation_found:
                return EdgeCaseResult(
                    case_type=EdgeCaseType.WEAK_CHOCH_REVERSAL,
                    is_valid=False,
                    confidence_adjustment=-0.5,
                    reasoning="Weak ChoCH without follow-through confirmation",
                    action="WAIT"
                )
        
        return None
    
    def _check_no_mitigation_wipeout(self, signal_data: Dict, market_data: Dict, context: Dict) -> Optional[EdgeCaseResult]:
        """
        Scenario 10: Massive candle wipes through OB + FVG
        Problem: No mitigation
        Solution: Cancel entry – no mitigation = no intent
        """
        recent_candles = market_data.get('recent_candles', [])
        order_blocks = market_data.get('order_blocks', [])
        fair_value_gaps = market_data.get('fair_value_gaps', [])
        
        if not recent_candles:
            return None
            
        latest_candle = recent_candles[-1]
        candle_range = latest_candle.get('high', 0) - latest_candle.get('low', 0)
        
        # Check if massive candle wiped through zones without mitigation
        zones_wiped = 0
        
        # Check OBs
        for ob in order_blocks[-3:]:
            ob_high = ob.get('high', 0)
            ob_low = ob.get('low', 0)
            if (latest_candle.get('low', 0) < ob_low and 
                latest_candle.get('high', 0) > ob_high):
                zones_wiped += 1
        
        # Check FVGs
        for fvg in fair_value_gaps[-3:]:
            fvg_top = fvg.get('top', 0)
            fvg_bottom = fvg.get('bottom', 0)
            if (latest_candle.get('low', 0) < fvg_bottom and 
                latest_candle.get('high', 0) > fvg_top):
                zones_wiped += 1
        
        # Calculate average candle range
        avg_range = np.mean([c.get('high', 0) - c.get('low', 0) for c in recent_candles[-10:]])
        
        # If massive candle (3x average) wiped through multiple zones
        if candle_range > (avg_range * 3.0) and zones_wiped >= 2:
            return EdgeCaseResult(
                case_type=EdgeCaseType.NO_MITIGATION_WIPEOUT,
                is_valid=False,
                confidence_adjustment=-0.9,
                reasoning="Massive candle wiped through zones without mitigation",
                action="BLOCK"
            )
        
        return None
    
    def _consolidate_results(self, results: List[EdgeCaseResult]) -> EdgeCaseResult:
        """
        Consolidate multiple edge case results into final decision
        """
        if not results:
            return EdgeCaseResult(
                case_type=EdgeCaseType.NO_LIQUIDITY_SWEEP_BEFORE_BOS,  # Default
                is_valid=True,
                confidence_adjustment=0.0,
                reasoning="No edge cases detected",
                action="PROCEED"
            )
        
        # Find most restrictive action
        actions = [r.action for r in results]
        if "BLOCK" in actions:
            final_action = "BLOCK"
        elif "WAIT" in actions:
            final_action = "WAIT"
        elif "MODIFY" in actions:
            final_action = "MODIFY"
        else:
            final_action = "PROCEED"
        
        # Calculate combined confidence adjustment
        total_adjustment = sum(r.confidence_adjustment for r in results)
        avg_adjustment = total_adjustment / len(results)
        
        # Determine overall validity
        blocking_results = [r for r in results if not r.is_valid]
        is_valid = len(blocking_results) == 0
        
        # Combine reasoning
        reasoning_parts = [r.reasoning for r in results]
        combined_reasoning = "; ".join(reasoning_parts)
        
        return EdgeCaseResult(
            case_type=results[0].case_type,  # Use first detected case
            is_valid=is_valid,
            confidence_adjustment=avg_adjustment,
            reasoning=combined_reasoning,
            action=final_action
        )
    
    def _select_best_order_block(self, order_blocks: List[Dict], fvgs: List[Dict], structure_breaks: List[Dict]) -> Dict:
        """
        Select the best order block from multiple stacked OBs based on confluence
        """
        best_ob = order_blocks[0]
        best_score = 0
        
        for ob in order_blocks:
            score = 0
            
            # Base score from OB strength
            score += ob.get('strength', 0) * 100
            
            # Bonus for FVG confluence
            ob_center = (ob.get('high', 0) + ob.get('low', 0)) / 2
            for fvg in fvgs:
                if fvg.get('bottom', 0) <= ob_center <= fvg.get('top', 0):
                    score += 50
                    break
            
            # Bonus for recent structure break
            for sb in structure_breaks[-2:]:
                sb_price = sb.get('price', 0)
                if abs(sb_price - ob_center) / ob_center < 0.005:  # Within 0.5%
                    score += 30
            
            # Prefer unmitigated OBs
            if not ob.get('mitigated', False):
                score += 20
            
            if score > best_score:
                best_score = score
                best_ob = ob
        
        return best_ob
    
    def _is_asia_session(self, timestamp: datetime) -> bool:
        """Check if timestamp is during Asia session with proper timezone alignment"""
        try:
            from session_timezone_aligner import SessionTimezoneAligner
            aligner = SessionTimezoneAligner()
            current_session = aligner.get_current_session(timestamp)
            return current_session.value.upper() == 'ASIA'
        except Exception:
            # Fallback to UTC hour logic
            utc_hour = timestamp.hour
            return 0 <= utc_hour <= 9
    
    def _is_weekend_gap_period(self, timestamp: datetime) -> bool:
        """Check if timestamp is during weekend gap period"""
        weekday = timestamp.weekday()
        hour = timestamp.hour
        
        # Friday after 22:00 UTC or Sunday before 22:00 UTC
        if (weekday == 4 and hour >= 22) or (weekday == 6 and hour < 22):
            return True
        
        # Saturday (full day)
        if weekday == 5:
            return True
            
        return False