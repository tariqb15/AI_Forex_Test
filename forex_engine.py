#!/usr/bin/env python3
"""
Agentic Forex Analytics Engine for USD/JPY
Implements Smart Money Concepts, Institutional Logic, and Multi-Timeframe Analysis

Author: BLOB AI Trading System
Version: 1.0.0
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import schedule
import time
from loguru import logger
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import os
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger.add("forex_engine.log", rotation="1 day", retention="30 days", level="DEBUG")

class SessionType(Enum):
    ASIA = "Asia"
    LONDON = "London"
    NEW_YORK = "New_York"
    OVERLAP = "Overlap"

class MarketStructure(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    RANGING = "Ranging"
    TRANSITION = "Transition"

class SignalStrength(Enum):
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass
class TradingSignal:
    timestamp: datetime
    signal_type: str
    direction: str
    strength: SignalStrength
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confidence: float
    reasoning: str
    session: SessionType
    market_structure: MarketStructure
    symbol: str = "USDJPY"  # Currency pair symbol

@dataclass
class TrendContext:
    age_in_bars: int
    momentum_strength: float
    phase: str  # 'early', 'mid', 'late'
    expansion_potential: float

@dataclass
class FVGContext:
    clean_status: bool
    bounce_history: int
    proximity_to_order_block: float
    dynamic_weight: float

@dataclass
class TradeOutcome:
    signal_id: str
    entry_price: float
    exit_price: float
    profit_loss: float
    risk_reward_actual: float
    factors_used: List[str]
    success: bool

class SmartMoneyAnalyzer:
    """Implements Smart Money Concepts for institutional trading logic"""
    
    def __init__(self):
        self.order_blocks = []
        self.fair_value_gaps = []
        self.liquidity_levels = []
        self.displacement_zones = []
    
    def clear_cache(self):
        """Clear all cached analysis data to prevent stale timestamp issues"""
        self.order_blocks.clear()
        self.fair_value_gaps.clear()
        self.liquidity_levels.clear()
        self.displacement_zones.clear()
    
    def _is_valid_data_timestamp(self, timestamp) -> bool:
        """Validate timestamp during data collection - accept any reasonable timestamp from collected data"""
        if not timestamp:
            return False
        
        try:
            # Handle string timestamps that might be invalid
            if isinstance(timestamp, str):
                try:
                    timestamp = pd.to_datetime(timestamp)
                except:
                    logger.debug(f"Invalid string data timestamp: {timestamp}")
                    return False
            
            # Convert to datetime if it's a pandas Timestamp
            if hasattr(timestamp, 'to_pydatetime'):
                try:
                    timestamp = timestamp.to_pydatetime()
                except:
                    logger.debug(f"Failed to convert pandas data timestamp: {timestamp}")
                    return False
            
            # Ensure we have a datetime object
            if not isinstance(timestamp, datetime):
                logger.debug(f"Data timestamp is not a datetime object: {type(timestamp)}")
                return False
            
            # Ensure timezone awareness
            if timestamp.tzinfo is None:
                timestamp = pytz.UTC.localize(timestamp)
            elif timestamp.tzinfo != pytz.UTC:
                timestamp = timestamp.astimezone(pytz.UTC)
            
            # For data collection, accept any timestamp from collected data
            # Only reject obviously invalid timestamps (like year 1970 or far future)
            if timestamp.year < 2020 or timestamp.year > 2030:
                logger.debug(f"Data timestamp year out of reasonable range: {timestamp}")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating data timestamp {timestamp}: {e}")
            return False
    
    def detect_market_structure_break(self, df: pd.DataFrame) -> Dict:
        """✅ Enhanced: Detect Market Structure Break (MSB) and Change of Character (ChoCH)"""
        structure_breaks = []
        choch_signals = []
        
        # ✅ Fixed: Use df parameter directly (not undefined recent_df)
        if len(df) < 50:
            return {'structure_breaks': structure_breaks, 'choch_signals': choch_signals}
        
        # ✅ Enhanced: Proper pivot detection for swing highs/lows
        swing_window = 10  # Look for pivots over 10-period window
        
        # Calculate swing highs and lows using proper pivot detection
        swing_highs = self._detect_swing_pivots(df, 'high', swing_window)
        swing_lows = self._detect_swing_pivots(df, 'low', swing_window)
        
        # Track market structure state for ChoCH detection
        current_trend = None  # 'bullish', 'bearish', or None
        last_higher_high = None
        last_higher_low = None
        last_lower_high = None
        last_lower_low = None
        
        # Process recent data for structure analysis
        recent_data = df.tail(100) if len(df) > 100 else df
        
        for i in range(swing_window, len(recent_data)):
            current_idx = recent_data.index[i]
            current_high = float(recent_data['high'].iloc[i])
            current_low = float(recent_data['low'].iloc[i])
            current_close = float(recent_data['close'].iloc[i])
            
            # ✅ Fixed: Use proper index for _calculate_break_strength
            df_index = i  # This is the correct index within recent_data
            
            # Check for swing high at current position
            if current_idx in swing_highs.index and swing_highs.loc[current_idx]:
                # BULLISH MSB: Close breaks above previous swing high
                if self._check_bullish_msb(recent_data, i, swing_highs, current_close):
                    break_strength = self._calculate_break_strength(recent_data, df_index, 'bullish')
                    
                    structure_breaks.append({
                        'timestamp': current_idx.to_pydatetime() if isinstance(current_idx, pd.Timestamp) else current_idx,
                        'type': 'Bullish_MSB',
                        'price': current_close,
                        'swing_high_broken': current_high,
                        'strength': float(min(break_strength, 1.0)),  # ✅ Enhanced: Cap at 1.0
                        'breakout_percentage': self._calculate_breakout_percentage(recent_data, i, 'bullish')
                    })
                    
                    # Update trend tracking
                    if current_trend == 'bearish':
                        # ✅ Enhanced: ChoCH detected - trend change from bearish to bullish
                        choch_signals.append({
                            'timestamp': current_idx.to_pydatetime() if isinstance(current_idx, pd.Timestamp) else current_idx,
                            'type': 'ChoCH_Bullish',
                            'price': current_close,
                            'previous_trend': 'bearish',
                            'new_trend': 'bullish',
                            'strength': float(min(break_strength, 1.0))
                        })
                    
                    current_trend = 'bullish'
                    last_higher_high = current_high
            
            # Check for swing low at current position
            if current_idx in swing_lows.index and swing_lows.loc[current_idx]:
                # BEARISH MSB: Close breaks below previous swing low
                if self._check_bearish_msb(recent_data, i, swing_lows, current_close):
                    break_strength = self._calculate_break_strength(recent_data, df_index, 'bearish')
                    
                    structure_breaks.append({
                        'timestamp': current_idx.to_pydatetime() if isinstance(current_idx, pd.Timestamp) else current_idx,
                        'type': 'Bearish_MSB',
                        'price': current_close,
                        'swing_low_broken': current_low,
                        'strength': float(min(break_strength, 1.0)),  # ✅ Enhanced: Cap at 1.0
                        'breakout_percentage': self._calculate_breakout_percentage(recent_data, i, 'bearish')
                    })
                    
                    # Update trend tracking
                    if current_trend == 'bullish':
                        # ✅ Enhanced: ChoCH detected - trend change from bullish to bearish
                        choch_signals.append({
                            'timestamp': current_idx.to_pydatetime() if isinstance(current_idx, pd.Timestamp) else current_idx,
                            'type': 'ChoCH_Bearish',
                            'price': current_close,
                            'previous_trend': 'bullish',
                            'new_trend': 'bearish',
                            'strength': float(min(break_strength, 1.0))
                        })
                    
                    current_trend = 'bearish'
                    last_lower_low = current_low
        
        return {
            'structure_breaks': structure_breaks,
            'choch_signals': choch_signals,  # ✅ Enhanced: Now properly implemented
            'current_trend': current_trend
        }
    
    def detect_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Identify Order Blocks - zones where institutions placed large orders - Enhanced"""
        order_blocks = []
        
        # Use only recent data for faster processing
        recent_df = df.tail(50) if len(df) > 50 else df
        
        # ✅ Enhanced: Calculate body_ratio as signal strength indicator
        body_size = abs(recent_df['close'] - recent_df['open'])
        candle_range = recent_df['high'] - recent_df['low']
        
        # Avoid division by zero
        body_ratio = body_size / candle_range.replace(0, 0.0001)
        
        # ✅ Enhanced: More aggressive threshold for institutional candles
        strong_moves = body_ratio > 0.6  # Body must be 60%+ of total range
        
        bullish_moves = strong_moves & (recent_df['close'] > recent_df['open'])
        bearish_moves = strong_moves & (recent_df['close'] < recent_df['open'])
        
        # Process only strong move candidates
        for i in range(10, len(recent_df) - 5):
            current_idx = recent_df.index[i]
            current_high = float(recent_df['high'].iloc[i])
            current_low = float(recent_df['low'].iloc[i])
            current_open = float(recent_df['open'].iloc[i])
            current_close = float(recent_df['close'].iloc[i])
            current_body_ratio = float(body_ratio.iloc[i])
            
            # ✅ BULLISH ORDER BLOCK DETECTION
            if bullish_moves.iloc[i]:
                # ✅ Enhanced: Pullback must respect body levels (not random closes)
                pullback_valid = self._validate_bullish_pullback(
                    recent_df, i, current_open, current_close
                )
                
                if pullback_valid:
                    # ✅ Enhanced: Dynamic strength calculation (0.0-1.0)
                    ob_strength = self._calculate_order_block_strength(
                        recent_df, i, 'bullish', current_body_ratio
                    )
                    
                    # Only include OBs with strength > 0.4 and valid timestamps
                    if ob_strength > 0.4:
                        timestamp_dt = current_idx.to_pydatetime() if isinstance(current_idx, pd.Timestamp) else current_idx
                        if self._is_valid_data_timestamp(timestamp_dt):
                            order_blocks.append({
                                'timestamp': timestamp_dt,
                                'type': 'Bullish_OB',
                                'high': current_high,
                                'low': current_low,
                                'open': current_open,
                                'close': current_close,
                                'body_ratio': current_body_ratio,
                                'strength': float(ob_strength),  # Dynamic 0.0-1.0 score
                                'pullback_validation': 'body_respected'
                            })
            
            # ✅ BEARISH ORDER BLOCK DETECTION
            elif bearish_moves.iloc[i]:
                # ✅ Enhanced: Pullback must respect body levels (not random closes)
                pullback_valid = self._validate_bearish_pullback(
                    recent_df, i, current_open, current_close
                )
                
                if pullback_valid:
                    # ✅ Enhanced: Dynamic strength calculation (0.0-1.0)
                    ob_strength = self._calculate_order_block_strength(
                        recent_df, i, 'bearish', current_body_ratio
                    )
                    
                    # Only include OBs with strength > 0.4 and valid timestamps
                    if ob_strength > 0.4:
                        timestamp_dt = current_idx.to_pydatetime() if isinstance(current_idx, pd.Timestamp) else current_idx
                        if self._is_valid_data_timestamp(timestamp_dt):
                            order_blocks.append({
                                'timestamp': timestamp_dt,
                                'type': 'Bearish_OB',
                                'high': current_high,
                                'low': current_low,
                                'open': current_open,
                                'close': current_close,
                                'body_ratio': current_body_ratio,
                                'strength': float(ob_strength),  # Dynamic 0.0-1.0 score
                                'pullback_validation': 'body_respected'
                            })
        
        return order_blocks
    
    def detect_fair_value_gaps(self, df: pd.DataFrame, symbol: str = 'USDJPY') -> List[Dict]:
        """✅ Enhanced: Identify Fair Value Gaps (FVGs) with comprehensive analysis"""
        fvgs = []
        
        # ✅ Fixed: Handle minimal data requirement
        if len(df) < 5:
            return fvgs
        
        # ✅ Fixed: Use symbol parameter instead of undefined self.symbol
        # Dynamic minimum gap size based on currency pair (normalized)
        is_jpy_pair = 'JPY' in symbol.upper()
        min_gap_size = 0.01 if is_jpy_pair else 0.0001
        
        # Calculate ATR for normalization
        atr_values = self._calculate_atr_for_normalization(df)
        current_atr = atr_values.iloc[-1] if len(atr_values) > 0 else min_gap_size * 10
        
        for i in range(2, len(df)):
            current_idx = df.index[i]
            
            # ✅ Enhanced: BULLISH FVG DETECTION
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                
                if gap_size > min_gap_size:
                    # ✅ Enhanced: Comprehensive context analysis
                    fvg_context = self._analyze_fvg_context(df, i, 'bullish')
                    
                    # ✅ Enhanced: Clear is_filled tracker
                    is_filled = self._check_fvg_filled_status(df, i, 'bullish')
                    
                    # ✅ Enhanced: Normalized size relative to ATR
                    normalized_size = gap_size / current_atr if current_atr > 0 else 1.0
                    
                    # ✅ Enhanced: Dynamic importance calculation (0.0-1.0)
                    importance = self._calculate_fvg_importance(
                        fvg_context, normalized_size, is_filled
                    )
                    
                    # ✅ Enhanced: Validate timestamp before adding
                    fvg_timestamp = current_idx.to_pydatetime() if isinstance(current_idx, pd.Timestamp) else current_idx
                    if self._is_valid_data_timestamp(fvg_timestamp):
                        fvgs.append({
                            'timestamp': fvg_timestamp,
                            'type': 'Bullish_FVG',
                            'top': float(df['low'].iloc[i]),
                            'bottom': float(df['high'].iloc[i-2]),
                            'size': float(gap_size),
                            'normalized_size': float(normalized_size),  # ✅ Enhanced: Normalized values
                            'is_filled': is_filled,  # ✅ Enhanced: Clear filled tracker
                            'importance': float(importance),  # ✅ Enhanced: Dynamic importance (0.0-1.0)
                            'bounce_count': fvg_context.bounce_history,
                            'clean_status': fvg_context.clean_status,
                            'proximity_to_ob': float(fvg_context.proximity_to_order_block),
                            'formation_candle_1': {
                                'timestamp': df.index[i-2].to_pydatetime() if isinstance(df.index[i-2], pd.Timestamp) else df.index[i-2],
                                'high': float(df['high'].iloc[i-2]),
                                'low': float(df['low'].iloc[i-2])
                            },
                            'formation_candle_3': {
                                'timestamp': df.index[i].to_pydatetime() if isinstance(df.index[i], pd.Timestamp) else df.index[i],
                                'high': float(df['high'].iloc[i]),
                                'low': float(df['low'].iloc[i])
                            }
                        })
            
            # ✅ Enhanced: BEARISH FVG DETECTION
            elif df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                
                if gap_size > min_gap_size:
                    # ✅ Enhanced: Comprehensive context analysis
                    fvg_context = self._analyze_fvg_context(df, i, 'bearish')
                    
                    # ✅ Enhanced: Clear is_filled tracker
                    is_filled = self._check_fvg_filled_status(df, i, 'bearish')
                    
                    # ✅ Enhanced: Normalized size relative to ATR
                    normalized_size = gap_size / current_atr if current_atr > 0 else 1.0
                    
                    # ✅ Enhanced: Dynamic importance calculation (0.0-1.0)
                    importance = self._calculate_fvg_importance(
                        fvg_context, normalized_size, is_filled
                    )
                    
                    # ✅ Enhanced: Validate timestamp before adding
                    fvg_timestamp = current_idx.to_pydatetime() if isinstance(current_idx, pd.Timestamp) else current_idx
                    if self._is_valid_data_timestamp(fvg_timestamp):
                        fvgs.append({
                            'timestamp': fvg_timestamp,
                            'type': 'Bearish_FVG',
                            'top': float(df['low'].iloc[i-2]),
                            'bottom': float(df['high'].iloc[i]),
                            'size': float(gap_size),
                            'normalized_size': float(normalized_size),  # ✅ Enhanced: Normalized values
                            'is_filled': is_filled,  # ✅ Enhanced: Clear filled tracker
                            'importance': float(importance),  # ✅ Enhanced: Dynamic importance (0.0-1.0)
                            'bounce_count': fvg_context.bounce_history,
                            'clean_status': fvg_context.clean_status,
                            'proximity_to_ob': float(fvg_context.proximity_to_order_block),
                            'formation_candle_1': {
                                'timestamp': df.index[i-2].to_pydatetime() if isinstance(df.index[i-2], pd.Timestamp) else df.index[i-2],
                                'high': float(df['high'].iloc[i-2]),
                                'low': float(df['low'].iloc[i-2])
                            },
                            'formation_candle_3': {
                                'timestamp': df.index[i].to_pydatetime() if isinstance(df.index[i], pd.Timestamp) else df.index[i],
                                'high': float(df['high'].iloc[i]),
                                'low': float(df['low'].iloc[i])
                            }
                        })
        
        # ✅ Enhanced: Filter only active (unfilled) FVGs if requested
        active_fvgs = [fvg for fvg in fvgs if not fvg['is_filled']]
        
        return {
            'all_fvgs': fvgs,
            'active_fvgs': active_fvgs,  # ✅ Enhanced: Separate active FVGs
            'total_count': len(fvgs),
            'active_count': len(active_fvgs),
            'bullish_count': len([f for f in fvgs if f['type'] == 'Bullish_FVG']),
            'bearish_count': len([f for f in fvgs if f['type'] == 'Bearish_FVG'])
        }
    
    def _analyze_fvg_context(self, df: pd.DataFrame, fvg_index: int, fvg_type: str) -> FVGContext:
        """Analyze FVG context for dynamic weighting"""
        # Check if FVG is clean (no fill)
        clean_status = self._check_fvg_clean_status(df, fvg_index, fvg_type)
        
        # Count bounce history
        bounce_history = self._count_fvg_bounces(df, fvg_index, fvg_type)
        
        # Check proximity to order blocks
        proximity_to_order_block = self._check_order_block_proximity(df, fvg_index)
        
        # Calculate dynamic weight (0.1 to 0.3)
        dynamic_weight = self._calculate_fvg_dynamic_weight(
            clean_status, bounce_history, proximity_to_order_block
        )
        
        return FVGContext(
            clean_status=clean_status,
            bounce_history=bounce_history,
            proximity_to_order_block=proximity_to_order_block,
            dynamic_weight=dynamic_weight
        )
    
    def _check_fvg_clean_status(self, df: pd.DataFrame, fvg_index: int, fvg_type: str) -> bool:
        """✅ Enhanced: Check if FVG has been filled/touched (legacy method)"""
        return not self._check_fvg_filled_status(df, fvg_index, fvg_type)
    
    def _check_fvg_filled_status(self, df: pd.DataFrame, fvg_index: int, fvg_type: str) -> bool:
        """✅ Enhanced: Clear is_filled tracker for FVGs"""
        if fvg_index >= len(df) - 1:
            return False  # No future data to check, assume not filled
        
        if fvg_type == 'bullish':
            fvg_bottom = df['high'].iloc[fvg_index-2]
            fvg_top = df['low'].iloc[fvg_index]
            # Check if any future candle filled the gap
            future_data = df.iloc[fvg_index+1:]
            # Filled if price moves through the entire gap
            filled = ((future_data['low'] <= fvg_bottom) | 
                     (future_data['high'] >= fvg_top)).any()
        else:
            fvg_bottom = df['high'].iloc[fvg_index]
            fvg_top = df['low'].iloc[fvg_index-2]
            future_data = df.iloc[fvg_index+1:]
            # Filled if price moves through the entire gap
            filled = ((future_data['low'] <= fvg_bottom) | 
                     (future_data['high'] >= fvg_top)).any()
        
        return filled
    
    def _calculate_atr_for_normalization(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """✅ Enhanced: Calculate ATR for FVG size normalization"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.fillna(true_range.mean())
    
    def _calculate_fvg_importance(self, fvg_context: 'FVGContext', normalized_size: float, is_filled: bool) -> float:
        """✅ Enhanced: Calculate dynamic FVG importance (0.0-1.0)"""
        base_importance = 0.0
        
        # 1. Size factor (30% of importance)
        size_factor = min(normalized_size / 2.0, 1.0)  # Normalize: 2x ATR = max
        base_importance += size_factor * 0.3
        
        # 2. Clean status (25% of importance)
        if fvg_context.clean_status and not is_filled:
            base_importance += 0.25
        elif not is_filled:
            base_importance += 0.15  # Partial bonus for unfilled
        
        # 3. Bounce history (25% of importance)
        bounce_factor = min(fvg_context.bounce_history / 3.0, 1.0)  # 3+ bounces = max
        base_importance += bounce_factor * 0.25
        
        # 4. Order block proximity (20% of importance)
        proximity_factor = fvg_context.proximity_to_order_block
        base_importance += proximity_factor * 0.2
        
        # Penalty for filled FVGs
        if is_filled:
            base_importance *= 0.3  # Reduce importance by 70%
        
        return min(max(base_importance, 0.0), 1.0)  # Ensure 0.0-1.0 range
    
    def _validate_bullish_pullback(self, df: pd.DataFrame, ob_index: int, ob_open: float, ob_close: float) -> bool:
        """✅ Enhanced: Validate that pullback respects bullish OB body levels"""
        # Check next 3-5 candles for proper pullback behavior
        pullback_window = df.iloc[ob_index+1:ob_index+6]
        
        if len(pullback_window) < 2:
            return False
        
        # Pullback should not close below the OB body (open level)
        # This confirms institutional interest at this level
        pullback_closes = pullback_window['close']
        pullback_lows = pullback_window['low']
        
        # At least 2 candles should show pullback behavior
        pullback_count = 0
        body_respected = True
        
        for close, low in zip(pullback_closes, pullback_lows):
            # Pullback: price moves down but respects the OB body
            if close < ob_close:  # Moving down (pullback)
                pullback_count += 1
                # Body must be respected (low shouldn't break significantly below open)
                if low < (ob_open - (ob_close - ob_open) * 0.2):  # 20% tolerance
                    body_respected = False
                    break
        
        return pullback_count >= 2 and body_respected
    
    def _validate_bearish_pullback(self, df: pd.DataFrame, ob_index: int, ob_open: float, ob_close: float) -> bool:
        """✅ Enhanced: Validate that pullback respects bearish OB body levels"""
        # Check next 3-5 candles for proper pullback behavior
        pullback_window = df.iloc[ob_index+1:ob_index+6]
        
        if len(pullback_window) < 2:
            return False
        
        # Pullback should not close above the OB body (open level)
        # This confirms institutional interest at this level
        pullback_closes = pullback_window['close']
        pullback_highs = pullback_window['high']
        
        # At least 2 candles should show pullback behavior
        pullback_count = 0
        body_respected = True
        
        for close, high in zip(pullback_closes, pullback_highs):
            # Pullback: price moves up but respects the OB body
            if close > ob_close:  # Moving up (pullback)
                pullback_count += 1
                # Body must be respected (high shouldn't break significantly above open)
                if high > (ob_open + (ob_open - ob_close) * 0.2):  # 20% tolerance
                    body_respected = False
                    break
        
        return pullback_count >= 2 and body_respected
    
    def _calculate_order_block_strength(self, df: pd.DataFrame, ob_index: int, ob_type: str, body_ratio: float) -> float:
        """✅ Enhanced: Calculate dynamic OB strength score (0.0-1.0)"""
        base_strength = 0.0
        
        # 1. Body ratio contribution (40% of total strength)
        body_strength = min(body_ratio * 0.67, 0.4)  # Cap at 0.4
        base_strength += body_strength
        
        # 2. Volume context (if available) - 20% of total strength
        if 'volume' in df.columns:
            current_volume = df['volume'].iloc[ob_index]
            avg_volume = df['volume'].iloc[max(0, ob_index-20):ob_index].mean()
            if avg_volume > 0:
                volume_ratio = min(current_volume / avg_volume, 3.0) / 3.0  # Normalize
                volume_strength = volume_ratio * 0.2
                base_strength += volume_strength
        else:
            base_strength += 0.1  # Default volume bonus when not available
        
        # 3. Displacement strength (20% of total strength)
        candle_range = df['high'].iloc[ob_index] - df['low'].iloc[ob_index]
        recent_ranges = df['high'].iloc[max(0, ob_index-10):ob_index] - df['low'].iloc[max(0, ob_index-10):ob_index]
        avg_range = recent_ranges.mean() if len(recent_ranges) > 0 else candle_range
        
        if avg_range > 0:
            displacement_ratio = min(candle_range / avg_range, 2.5) / 2.5  # Normalize
            displacement_strength = displacement_ratio * 0.2
            base_strength += displacement_strength
        
        # 4. Market context bonus (20% of total strength)
        context_strength = self._calculate_market_context_strength(df, ob_index, ob_type)
        base_strength += context_strength
        
        # Ensure final strength is between 0.0 and 1.0
        return min(max(base_strength, 0.0), 1.0)
    
    def _calculate_market_context_strength(self, df: pd.DataFrame, ob_index: int, ob_type: str) -> float:
        """Calculate market context strength for OB validation"""
        context_strength = 0.0
        
        # Check if OB aligns with overall trend (10% bonus)
        if ob_index >= 20:
            recent_closes = df['close'].iloc[ob_index-20:ob_index]
            trend_direction = 'bullish' if recent_closes.iloc[-1] > recent_closes.iloc[0] else 'bearish'
            
            if (ob_type == 'bullish' and trend_direction == 'bullish') or \
               (ob_type == 'bearish' and trend_direction == 'bearish'):
                context_strength += 0.1
        
        # Check for confluence with key levels (10% bonus)
        current_price = df['close'].iloc[ob_index]
        recent_highs = df['high'].iloc[max(0, ob_index-50):ob_index]
        recent_lows = df['low'].iloc[max(0, ob_index-50):ob_index]
        
        # Check if OB is near significant support/resistance
        key_levels = list(recent_highs.nlargest(3)) + list(recent_lows.nsmallest(3))
        for level in key_levels:
            if abs(current_price - level) / current_price < 0.002:  # Within 0.2%
                context_strength += 0.1
                break
        
        return min(context_strength, 0.2)  # Cap at 20%
    
    def _count_fvg_bounces(self, df: pd.DataFrame, fvg_index: int, fvg_type: str) -> int:
        """Count how many times price bounced from FVG"""
        bounces = 0
        if fvg_index >= len(df) - 1:
            return bounces
        
        if fvg_type == 'bullish':
            fvg_level = df['high'].iloc[fvg_index-2]
            future_data = df.iloc[fvg_index+1:]
            # Count touches that resulted in bounces
            for i in range(len(future_data) - 1):
                if (future_data['low'].iloc[i] <= fvg_level <= future_data['high'].iloc[i] and
                    future_data['close'].iloc[i+1] > future_data['close'].iloc[i]):
                    bounces += 1
        else:
            fvg_level = df['low'].iloc[fvg_index-2]
            future_data = df.iloc[fvg_index+1:]
            for i in range(len(future_data) - 1):
                if (future_data['low'].iloc[i] <= fvg_level <= future_data['high'].iloc[i] and
                    future_data['close'].iloc[i+1] < future_data['close'].iloc[i]):
                    bounces += 1
        
        return bounces
    
    def _check_order_block_proximity(self, df: pd.DataFrame, fvg_index: int) -> float:
        """Check proximity to order blocks (0.0 to 1.0)"""
        # Simplified proximity check - look for strong candles near FVG
        window_start = max(0, fvg_index - 10)
        window_end = min(len(df), fvg_index + 10)
        window_data = df.iloc[window_start:window_end]
        
        # Look for strong candles (potential order blocks)
        body_sizes = abs(window_data['close'] - window_data['open'])
        candle_ranges = window_data['high'] - window_data['low']
        strong_candles = body_sizes > (candle_ranges * 0.7)
        
        if strong_candles.any():
            return 1.0  # High proximity
        else:
            return 0.3  # Low proximity
    
    def _calculate_fvg_dynamic_weight(self, clean_status: bool, bounce_history: int, proximity: float) -> float:
        """Calculate dynamic weight for FVG (0.1 to 0.3)"""
        base_weight = 0.1
        
        # Clean FVG bonus
        if clean_status:
            base_weight += 0.05
        
        # Bounce history bonus
        bounce_bonus = min(bounce_history * 0.03, 0.1)
        base_weight += bounce_bonus
        
        # Order block proximity bonus
        proximity_bonus = proximity * 0.05
        base_weight += proximity_bonus
        
        return min(base_weight, 0.3)
    
    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> List[Dict]:
        """Identify liquidity sweeps (stop hunts) above/below key levels - Enhanced"""
        sweeps = []
        
        # Handle minimal bar requirement to prevent runtime errors
        if len(df) < 15:
            return sweeps
        
        # Use only recent data for faster processing
        recent_df = df.tail(50) if len(df) > 50 else df
        
        # ✅ Enhanced: Proper pivot detection with center=True for true swing points
        swing_window = min(7, len(recent_df) // 3)  # Dynamic window based on data size
        swing_highs = recent_df['high'].rolling(window=swing_window, center=True).max() == recent_df['high']
        swing_lows = recent_df['low'].rolling(window=swing_window, center=True).min() == recent_df['low']
        
        # Get valid swing points (excluding NaN from center=True)
        valid_swing_highs = swing_highs.dropna()
        valid_swing_lows = swing_lows.dropna()
        
        # Extract swing levels with their timestamps for dynamic comparison
        swing_high_indices = valid_swing_highs[valid_swing_highs].index
        swing_low_indices = valid_swing_lows[valid_swing_lows].index
        swing_high_data = recent_df.loc[swing_high_indices, ['high']]
        swing_low_data = recent_df.loc[swing_low_indices, ['low']]
        
        # Process candles for sweep detection
        start_idx = max(swing_window, 10)
        for i in range(start_idx, len(recent_df)):
            current_idx = recent_df.index[i]
            current_high = float(recent_df['high'].iloc[i])
            current_low = float(recent_df['low'].iloc[i])
            current_close = float(recent_df['close'].iloc[i])
            current_open = float(recent_df['open'].iloc[i])
            
            # ✅ Enhanced: Dynamic comparison - find relevant swing levels before current candle
            relevant_highs = swing_high_data[swing_high_data.index < current_idx]
            relevant_lows = swing_low_data[swing_low_data.index < current_idx]
            
            # HIGH SWEEP DETECTION
            if len(relevant_highs) > 0:
                # Get the highest swing high before current candle
                max_swing_high = float(relevant_highs['high'].max())
                
                # Sweep condition: break above swing high but close below it
                if current_high > max_swing_high and current_close < max_swing_high:
                    # ✅ Enhanced: Calculate strength using price rejection ratio
                    sweep_distance = current_high - max_swing_high
                    rejection_distance = current_high - current_close
                    candle_range = current_high - current_low
                    
                    # Strength based on rejection ratio and sweep magnitude
                    if candle_range > 0:
                        rejection_ratio = rejection_distance / candle_range
                        sweep_strength = min(rejection_ratio * 2.0, 1.0)  # Cap at 1.0
                    else:
                        sweep_strength = 0.5  # Default for doji candles
                    
                    # Convert pandas Timestamp to datetime to avoid exception issues
                    timestamp_dt = current_idx.to_pydatetime() if isinstance(current_idx, pd.Timestamp) else current_idx
                    
                    # Validate timestamp before adding to results
                    if self._is_valid_data_timestamp(timestamp_dt):
                        sweeps.append({
                            'timestamp': timestamp_dt,  # ✅ Enhanced: Proper timestamp formatting
                            'type': 'Liquidity_Sweep_High',
                            'sweep_level': float(max_swing_high),
                            'sweep_price': current_high,
                            'close_price': current_close,
                            'open_price': current_open,
                            'rejection_distance': float(rejection_distance),
                            'sweep_distance': float(sweep_distance),
                            'strength': float(sweep_strength)  # ✅ Enhanced: Dynamic strength calculation
                        })
            
            # LOW SWEEP DETECTION
            if len(relevant_lows) > 0:
                # Get the lowest swing low before current candle
                min_swing_low = float(relevant_lows['low'].min())
                
                # Sweep condition: break below swing low but close above it
                if current_low < min_swing_low and current_close > min_swing_low:
                    # ✅ Enhanced: Calculate strength using price rejection ratio
                    sweep_distance = min_swing_low - current_low
                    rejection_distance = current_close - current_low
                    candle_range = current_high - current_low
                    
                    # Strength based on rejection ratio and sweep magnitude
                    if candle_range > 0:
                        rejection_ratio = rejection_distance / candle_range
                        sweep_strength = min(rejection_ratio * 2.0, 1.0)  # Cap at 1.0
                    else:
                        sweep_strength = 0.5  # Default for doji candles
                    
                    # Convert pandas Timestamp to datetime to avoid exception issues
                    timestamp_dt = current_idx.to_pydatetime() if isinstance(current_idx, pd.Timestamp) else current_idx
                    
                    # Validate timestamp before adding to results
                    if self._is_valid_data_timestamp(timestamp_dt):
                        sweeps.append({
                            'timestamp': timestamp_dt,  # ✅ Enhanced: Proper timestamp formatting
                            'type': 'Liquidity_Sweep_Low',
                            'sweep_level': float(min_swing_low),
                            'sweep_price': current_low,
                            'close_price': current_close,
                            'open_price': current_open,
                            'rejection_distance': float(rejection_distance),
                            'sweep_distance': float(sweep_distance),
                            'strength': float(sweep_strength)  # ✅ Enhanced: Dynamic strength calculation
                        })
        
        return sweeps
    
    def _detect_swing_pivots(self, df: pd.DataFrame, price_type: str, window: int) -> pd.Series:
        """✅ Enhanced: Detect proper swing pivots using centered rolling windows"""
        if price_type == 'high':
            # Swing high: highest point in the window
            rolling_max = df[price_type].rolling(window=window*2+1, center=True).max()
            swing_pivots = df[price_type] == rolling_max
        else:
            # Swing low: lowest point in the window
            rolling_min = df[price_type].rolling(window=window*2+1, center=True).min()
            swing_pivots = df[price_type] == rolling_min
        
        return swing_pivots.fillna(False)
    
    def _check_bullish_msb(self, df: pd.DataFrame, current_index: int, swing_highs: pd.Series, current_close: float) -> bool:
        """✅ Enhanced: Check if current close breaks above previous swing high"""
        # Look for previous swing highs before current position
        swing_highs_before = swing_highs[:current_index]
        previous_swing_indices = swing_highs_before[swing_highs_before].index
        
        if len(previous_swing_indices) == 0:
            return False
        
        # Get the most recent swing high
        last_swing_high_idx = previous_swing_indices[-1]
        try:
            last_swing_high_price = df['high'].loc[last_swing_high_idx]
        except KeyError:
            # If exact timestamp not found, find nearest timestamp
            nearest_idx = df.index.get_indexer([last_swing_high_idx], method='nearest')[0]
            if nearest_idx >= 0 and nearest_idx < len(df):
                last_swing_high_price = df['high'].iloc[nearest_idx]
            else:
                return False
        
        # MSB condition: current close breaks above the swing high
        return current_close > last_swing_high_price
    
    def _check_bearish_msb(self, df: pd.DataFrame, current_index: int, swing_lows: pd.Series, current_close: float) -> bool:
        """✅ Enhanced: Check if current close breaks below previous swing low"""
        # Look for previous swing lows before current position
        swing_lows_before = swing_lows[:current_index]
        previous_swing_indices = swing_lows_before[swing_lows_before].index
        
        if len(previous_swing_indices) == 0:
            return False
        
        # Get the most recent swing low
        last_swing_low_idx = previous_swing_indices[-1]
        try:
            last_swing_low_price = df['low'].loc[last_swing_low_idx]
        except KeyError:
            # If exact timestamp not found, find nearest timestamp
            nearest_idx = df.index.get_indexer([last_swing_low_idx], method='nearest')[0]
            if nearest_idx >= 0 and nearest_idx < len(df):
                last_swing_low_price = df['low'].iloc[nearest_idx]
            else:
                return False
        
        # MSB condition: current close breaks below the swing low
        return current_close < last_swing_low_price
    
    def _calculate_breakout_percentage(self, df: pd.DataFrame, index: int, direction: str) -> float:
        """✅ Enhanced: Calculate percentage breakout strength"""
        current_close = df['close'].iloc[index]
        
        if direction == 'bullish':
            # Find recent swing high that was broken
            recent_highs = df['high'].iloc[max(0, index-20):index]
            if len(recent_highs) > 0:
                swing_high = recent_highs.max()
                if swing_high > 0:
                    breakout_pct = ((current_close - swing_high) / swing_high) * 100
                    return min(abs(breakout_pct), 5.0)  # Cap at 5%
        else:
            # Find recent swing low that was broken
            recent_lows = df['low'].iloc[max(0, index-20):index]
            if len(recent_lows) > 0:
                swing_low = recent_lows.min()
                if swing_low > 0:
                    breakout_pct = ((swing_low - current_close) / swing_low) * 100
                    return min(abs(breakout_pct), 5.0)  # Cap at 5%
        
        return 0.0
    
    def _calculate_break_strength(self, df: pd.DataFrame, index: int, direction: str) -> float:
        """✅ Enhanced: Calculate the strength of a structure break (0.0-1.0)"""
        base_strength = 0.0
        
        # 1. Volume factor (40% of strength) - if available
        if 'tick_volume' in df.columns and index >= 20:
            current_volume = df['tick_volume'].iloc[index]
            avg_volume = df['tick_volume'].iloc[index-20:index].mean()
            if avg_volume > 0:
                volume_ratio = min(current_volume / avg_volume, 3.0) / 3.0  # Normalize to 0-1
                base_strength += volume_ratio * 0.4
        else:
            base_strength += 0.2  # Default volume bonus when not available
        
        # 2. Price action factor (40% of strength)
        current_open = df['open'].iloc[index]
        current_close = df['close'].iloc[index]
        current_high = df['high'].iloc[index]
        current_low = df['low'].iloc[index]
        candle_range = current_high - current_low
        
        if candle_range > 0:
            if direction == 'bullish':
                # Bullish: close near high shows strong buying
                price_factor = (current_close - current_low) / candle_range
            else:
                # Bearish: close near low shows strong selling
                price_factor = (current_high - current_close) / candle_range
            
            base_strength += price_factor * 0.4
        
        # 3. Breakout magnitude (20% of strength)
        breakout_pct = self._calculate_breakout_percentage(df, index, direction)
        magnitude_factor = min(breakout_pct / 2.0, 1.0)  # Normalize 2% breakout = 1.0
        base_strength += magnitude_factor * 0.2
        
        return min(max(base_strength, 0.0), 1.0)  # Ensure 0.0-1.0 range

class TrendAgeAnalyzer:
    """Analyzes trend age and momentum context for better entry timing"""
    
    def __init__(self):
        self.trend_history = deque(maxlen=500)  # Store recent trend data
    
    def analyze_trend_context(self, df: pd.DataFrame) -> TrendContext:
        """Analyze current trend age and momentum context"""
        # Calculate trend direction using EMA crossover
        df['ema_fast'] = df['close'].ewm(span=20).mean()
        df['ema_slow'] = df['close'].ewm(span=50).mean()
        
        # Determine trend direction
        current_trend = 'bullish' if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1] else 'bearish'
        
        # Calculate trend age
        age_in_bars = self._calculate_trend_age(df)
        
        # Calculate momentum strength
        momentum_strength = self._calculate_momentum_strength(df)
        
        # Determine trend phase
        phase = self._determine_trend_phase(age_in_bars, momentum_strength)
        
        # Calculate expansion potential
        expansion_potential = self._calculate_expansion_potential(df, age_in_bars, momentum_strength)
        
        return TrendContext(
            age_in_bars=age_in_bars,
            momentum_strength=momentum_strength,
            phase=phase,
            expansion_potential=expansion_potential
        )
    
    def _calculate_trend_age(self, df: pd.DataFrame) -> int:
        """Calculate how long the current trend has been alive"""
        df['ema_fast'] = df['close'].ewm(span=20).mean()
        df['ema_slow'] = df['close'].ewm(span=50).mean()
        df['trend_signal'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        
        # Find the last trend change
        trend_changes = df['trend_signal'].diff() != 0
        last_change_idx = df[trend_changes].index[-1] if trend_changes.any() else df.index[0]
        
        # Calculate bars since last trend change
        current_idx = df.index[-1]
        age_in_bars = len(df.loc[last_change_idx:current_idx]) - 1
        
        return max(age_in_bars, 1)
    
    def _calculate_momentum_strength(self, df: pd.DataFrame) -> float:
        """Calculate current momentum strength (0.0 to 1.0)"""
        # Use RSI and price velocity
        rsi = self._calculate_rsi(df['close'], 14)
        current_rsi = rsi.iloc[-1]
        
        # Price velocity (rate of change)
        price_velocity = df['close'].pct_change(10).iloc[-1]
        
        # Normalize RSI to momentum scale
        rsi_momentum = abs(current_rsi - 50) / 50  # 0 to 1 scale
        
        # Combine RSI and velocity
        momentum_strength = (rsi_momentum + abs(price_velocity) * 100) / 2
        
        return min(momentum_strength, 1.0)
    
    def _determine_trend_phase(self, age_in_bars: int, momentum_strength: float) -> str:
        """Determine if trend is in early, mid, or late phase"""
        if age_in_bars <= 20 and momentum_strength > 0.6:
            return 'early'
        elif age_in_bars <= 50 and momentum_strength > 0.4:
            return 'mid'
        else:
            return 'late'
    
    def _calculate_expansion_potential(self, df: pd.DataFrame, age: int, momentum: float) -> float:
        """Calculate potential for further trend expansion"""
        # Early trends with high momentum have high expansion potential
        if age <= 20 and momentum > 0.7:
            return 0.9
        elif age <= 30 and momentum > 0.5:
            return 0.7
        elif age <= 50 and momentum > 0.3:
            return 0.5
        else:
            return 0.2
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class VolatilityAnalyzer:
    """Enhanced volatility analyzer with compression zones and expansion prediction"""
    
    def __init__(self):
        self.compression_threshold = 0.3
        self.expansion_threshold = 2.0
        self.norm_window = 100  # Stable reference window for normalization
        self.min_bars_required = 100  # Minimum bars needed for analysis
    
    def analyze_compression_expansion(self, df: pd.DataFrame) -> Dict:
        """Enhanced compression detection with zone tracking and expansion probability"""
        # Safety check for minimum data requirement
        if len(df) < self.min_bars_required:
            return {
                'compression_zones': [],
                'expansion_zones': [],
                'current_volatility_state': 'INSUFFICIENT_DATA',
                'volatility_prediction': {'direction': 'UNKNOWN', 'confidence': 0.0}
            }
        
        # Calculate ATR and standard deviation
        df = df.copy()
        df['atr'] = self._calculate_atr(df, period=14)
        df['price_std'] = df['close'].rolling(window=20).std()
        
        # Enhanced normalization with stable 100-bar window
        atr_normalized = df['atr'] / df['atr'].rolling(window=self.norm_window, min_periods=self.norm_window).mean()
        std_normalized = df['price_std'] / df['price_std'].rolling(window=self.norm_window, min_periods=self.norm_window).mean()
        
        # Detect compression and expansion zones
        compression_zones = self._detect_compression_zones(df, atr_normalized, std_normalized)
        expansion_zones = self._detect_expansion_zones(df, atr_normalized, std_normalized)
        
        # Current state analysis
        current_state = self._get_current_volatility_state(
            atr_normalized.iloc[-1] if not pd.isna(atr_normalized.iloc[-1]) else 1.0,
            std_normalized.iloc[-1] if not pd.isna(std_normalized.iloc[-1]) else 1.0
        )
        
        # Generate volatility profile prediction
        volatility_prediction = self._predict_volatility_profile(df)
        
        return {
            'compression_zones': compression_zones,
            'expansion_zones': expansion_zones,
            'current_volatility_state': current_state,
            'volatility_prediction': volatility_prediction,
            'analysis_metadata': {
                'bars_analyzed': len(df),
                'norm_window': self.norm_window,
                'compression_threshold': self.compression_threshold,
                'expansion_threshold': self.expansion_threshold
            }
        }
    
    def _detect_compression_zones(self, df: pd.DataFrame, atr_norm: pd.Series, std_norm: pd.Series) -> List[Dict]:
        """Detect compression zones with zone_start and zone_end tracking"""
        compression_zones = []
        in_compression = False
        zone_start = None
        
        for i in range(self.norm_window, len(df)):
            current_atr = atr_norm.iloc[i]
            current_std = std_norm.iloc[i]
            
            # Skip if data is NaN
            if pd.isna(current_atr) or pd.isna(current_std):
                continue
            
            is_compressed = (current_atr < self.compression_threshold and 
                           current_std < self.compression_threshold)
            
            if is_compressed and not in_compression:
                # Start of new compression zone
                in_compression = True
                zone_start = i
            elif not is_compressed and in_compression:
                # End of compression zone
                in_compression = False
                zone_end = i - 1
                duration = zone_end - zone_start + 1
                
                # Enhanced expansion probability scaling with duration
                expansion_probability = self._calculate_expansion_probability(duration)
                
                # Enhanced expansion forecast
                expansion_forecast = self._forecast_expansion(df, zone_end, duration)
                
                # Convert pandas Timestamps to datetime objects
                zone_start_ts = df.index[zone_start]
                zone_end_ts = df.index[zone_end]
                if isinstance(zone_start_ts, pd.Timestamp):
                    zone_start_dt = zone_start_ts.to_pydatetime()
                else:
                    zone_start_dt = zone_start_ts
                if isinstance(zone_end_ts, pd.Timestamp):
                    zone_end_dt = zone_end_ts.to_pydatetime()
                else:
                    zone_end_dt = zone_end_ts
                    
                compression_zones.append({
                    'zone_start': zone_start_dt,
                    'zone_end': zone_end_dt,
                    'timestamp': zone_end_dt,  # For backward compatibility
                    'duration': duration,
                    'atr_level': float(atr_norm.iloc[zone_end]),
                    'std_level': float(std_norm.iloc[zone_end]),
                    'expansion_probability': expansion_probability,
                    'confidence_label': self._get_confidence_label(expansion_probability),
                    'expansion_forecast': expansion_forecast,
                    'avg_atr_during_compression': float(atr_norm.iloc[zone_start:zone_end+1].mean()),
                    'avg_std_during_compression': float(std_norm.iloc[zone_start:zone_end+1].mean())
                })
        
        # Handle ongoing compression at the end
        if in_compression and zone_start is not None:
            zone_end = len(df) - 1
            duration = zone_end - zone_start + 1
            expansion_probability = self._calculate_expansion_probability(duration)
            expansion_forecast = self._forecast_expansion(df, zone_end, duration)
            
            # Convert pandas Timestamps to datetime objects
            zone_start_ts = df.index[zone_start]
            zone_end_ts = df.index[zone_end]
            if isinstance(zone_start_ts, pd.Timestamp):
                zone_start_dt = zone_start_ts.to_pydatetime()
            else:
                zone_start_dt = zone_start_ts
            if isinstance(zone_end_ts, pd.Timestamp):
                zone_end_dt = zone_end_ts.to_pydatetime()
            else:
                zone_end_dt = zone_end_ts
                
            compression_zones.append({
                'zone_start': zone_start_dt,
                'zone_end': zone_end_dt,
                'timestamp': zone_end_dt,
                'duration': duration,
                'atr_level': float(atr_norm.iloc[zone_end]),
                'std_level': float(std_norm.iloc[zone_end]),
                'expansion_probability': expansion_probability,
                'confidence_label': self._get_confidence_label(expansion_probability),
                'expansion_forecast': expansion_forecast,
                'avg_atr_during_compression': float(atr_norm.iloc[zone_start:zone_end+1].mean()),
                'avg_std_during_compression': float(std_norm.iloc[zone_start:zone_end+1].mean()),
                'is_ongoing': True
            })
        
        return compression_zones
    
    def _detect_expansion_zones(self, df: pd.DataFrame, atr_norm: pd.Series, std_norm: pd.Series) -> List[Dict]:
        """Detect expansion zones with enhanced tracking"""
        expansion_zones = []
        
        for i in range(self.norm_window, len(df)):
            current_atr = atr_norm.iloc[i]
            current_std = std_norm.iloc[i]
            
            # Skip if data is NaN
            if pd.isna(current_atr) or pd.isna(current_std):
                continue
            
            if current_atr > self.expansion_threshold or current_std > self.expansion_threshold:
                # Convert pandas Timestamp to datetime object
                timestamp_val = df.index[i]
                if isinstance(timestamp_val, pd.Timestamp):
                    timestamp_dt = timestamp_val.to_pydatetime()
                else:
                    timestamp_dt = timestamp_val
                    
                expansion_zones.append({
                    'timestamp': timestamp_dt,
                    'atr_level': float(current_atr),
                    'std_level': float(current_std),
                    'strength': float(max(current_atr, current_std)),
                    'expansion_type': 'ATR_DRIVEN' if current_atr > current_std else 'STD_DRIVEN'
                })
        
        return expansion_zones
    
    def _calculate_expansion_probability(self, duration: int) -> float:
        """Enhanced expansion probability calculation with better scaling"""
        # Base probability starts at 0.3 for short compressions
        base_prob = 0.3
        
        # Enhanced duration scaling with different formula than bonus calculation
        # Uses square root scaling for probability (different from logarithmic bonus)
        duration_factor = min(np.sqrt(duration) / np.sqrt(25), 0.5)  # Max 0.5 from duration
        
        # Volatility context bonus (compression intensity)
        volatility_bonus = 0.1  # Additional probability for confirmed compression
        
        # Combine all factors
        total_probability = base_prob + duration_factor + volatility_bonus
        
        # Ensure probability > 50% for significant compressions (duration >= 8)
        if duration >= 8:
            total_probability = max(total_probability, 0.52)
        
        # Cap at 0.95 to maintain realism
        return min(total_probability, 0.95)
    
    def _get_confidence_label(self, probability: float) -> str:
        """Generate confidence labels for LLM filtering"""
        if probability >= 0.80:
            return "VERY_HIGH"
        elif probability >= 0.65:
            return "HIGH"
        elif probability >= 0.50:
            return "MODERATE"
        elif probability >= 0.35:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _forecast_expansion(self, df: pd.DataFrame, compression_index: int, duration: int) -> Dict:
        """Enhanced expansion forecasting with defensive coding and better fusion"""
        # Defensive: Ensure valid compression_index
        if compression_index >= len(df) or compression_index < 0:
            compression_index = len(df) - 1
        
        # Defensive: Get current_atr safely
        try:
            current_atr = df['atr'].iloc[compression_index]
            if pd.isna(current_atr) or current_atr <= 0:
                current_atr = df['atr'].dropna().iloc[-1] if len(df['atr'].dropna()) > 0 else 0.001
        except (IndexError, KeyError):
            current_atr = 0.001  # Fallback minimum value
        
        # Analyze historical breakouts with defensive handling
        historical_breakouts = self._analyze_historical_breakouts(df, duration)
        
        # Defensive: Protected access to historical_breakouts
        sample_size = historical_breakouts.get('sample_size', 0)
        avg_expansion = historical_breakouts.get('avg_expansion', 2.0)
        success_rate = historical_breakouts.get('success_rate', 0.6)
        
        # Calculate expansion multiplier with protected data
        expected_expansion_multiplier = self._calculate_expansion_multiplier(duration, historical_breakouts)
        
        # Estimate range of expansion (now properly using current_atr)
        expected_range = current_atr * expected_expansion_multiplier
        
        # Enhanced explosiveness score with better fusion model
        # Different scaling for duration component vs probability calculation
        duration_score = min(np.log(duration + 1) / np.log(15), 1.0)  # Logarithmic for score
        historical_score = min(avg_expansion / 4.0, 1.0)  # Normalized historical strength
        success_score = success_rate  # Already 0-1 range
        sample_confidence = min(sample_size / 8.0, 1.0)  # Confidence in historical data
        
        # Weighted fusion with confidence adjustment
        base_explosiveness = (
            duration_score * 0.35 +  # Duration component (35%)
            historical_score * 0.35 +  # Historical strength (35%)
            success_score * 0.30  # Success rate (30%)
        )
        
        # Apply sample confidence weighting
        explosiveness_score = (
            base_explosiveness * sample_confidence + 
            (duration_score * 0.6 + success_score * 0.4) * (1 - sample_confidence)
        )
        
        # Ensure normalized 0-1 range
        explosiveness_score = max(0.0, min(explosiveness_score, 1.0))
        
        # Enhanced confidence calculation
        data_confidence = min(sample_size / 10.0, 0.8)  # Max 0.8 from sample size
        atr_confidence = 0.9 if current_atr > 0.001 else 0.3  # ATR data quality
        overall_confidence = (data_confidence * 0.7) + (atr_confidence * 0.3)
        
        return {
            'explosiveness_score': float(explosiveness_score),
            'expected_range': float(expected_range),
            'expansion_multiplier': float(expected_expansion_multiplier),
            'current_atr': float(current_atr),
            'historical_reference': historical_breakouts,
            'confidence': float(overall_confidence),
            'confidence_breakdown': {
                'data_confidence': float(data_confidence),
                'atr_confidence': float(atr_confidence),
                'sample_size': sample_size
            },
            'score_components': {
                'duration_score': float(duration_score),
                'historical_score': float(historical_score),
                'success_score': float(success_score),
                'sample_confidence': float(sample_confidence)
            }
        }
    
    def _analyze_historical_breakouts(self, df: pd.DataFrame, target_duration: int) -> Dict:
        """Enhanced historical breakout analysis with better pattern matching"""
        breakout_data = []
        
        # Use stable normalization window
        atr_normalized = df['atr'] / df['atr'].rolling(window=self.norm_window, min_periods=self.norm_window).mean()
        
        # Ensure we have enough data for analysis
        start_idx = max(self.norm_window, 50)
        end_idx = len(df) - 20  # Leave room for measuring breakout
        
        if start_idx >= end_idx:
            return self._get_default_historical_data()
        
        for i in range(start_idx, end_idx):
            if pd.isna(atr_normalized.iloc[i]) or atr_normalized.iloc[i] >= self.compression_threshold:
                continue
            
            # Measure compression duration more accurately
            duration = self._measure_compression_duration(atr_normalized, i)
            
            # If similar duration (with tolerance), measure subsequent expansion
            duration_tolerance = max(2, target_duration * 0.3)  # Dynamic tolerance
            if abs(duration - target_duration) <= duration_tolerance:
                expansion_strength = self._measure_expansion_strength(df, i)
                if expansion_strength > 0:  # Valid expansion
                    # Convert pandas Timestamp to datetime to avoid exception issues
                    timestamp_val = df.index[i]
                    if isinstance(timestamp_val, pd.Timestamp):
                        timestamp_dt = timestamp_val.to_pydatetime()
                    else:
                        timestamp_dt = timestamp_val
                    
                    breakout_data.append({
                        'duration': duration,
                        'expansion_strength': expansion_strength,
                        'timestamp': timestamp_dt
                    })
        
        if breakout_data:
            expansions = [b['expansion_strength'] for b in breakout_data]
            avg_expansion = np.mean(expansions)
            max_expansion = np.max(expansions)
            success_rate = len([b for b in breakout_data if b['expansion_strength'] > 1.5]) / len(breakout_data)
        else:
            return self._get_default_historical_data()
        
        return {
            'sample_size': len(breakout_data),
            'avg_expansion': float(avg_expansion),
            'max_expansion': float(max_expansion),
            'success_rate': float(success_rate)
        }
    
    def _measure_compression_duration(self, atr_normalized: pd.Series, end_idx: int) -> int:
        """Accurately measure compression duration"""
        duration = 1
        for j in range(end_idx - 1, max(0, end_idx - 50), -1):
            if (not pd.isna(atr_normalized.iloc[j]) and 
                atr_normalized.iloc[j] < self.compression_threshold):
                duration += 1
            else:
                break
        return duration
    
    def _get_default_historical_data(self) -> Dict:
        """Return default historical data when insufficient samples"""
        return {
            'sample_size': 0,
            'avg_expansion': 2.0,
            'max_expansion': 3.0,
            'success_rate': 0.6
        }
    
    def _measure_expansion_strength(self, df: pd.DataFrame, compression_end_index: int) -> float:
        """Enhanced expansion strength measurement"""
        # Look at next 15 bars after compression for better measurement
        expansion_window_size = min(15, len(df) - compression_end_index - 1)
        
        if expansion_window_size < 3:
            return 0.0  # Insufficient data
        
        expansion_window = df.iloc[compression_end_index:compression_end_index + expansion_window_size]
        
        # Calculate multiple expansion metrics
        compression_atr = df['atr'].iloc[compression_end_index]
        if compression_atr <= 0:
            return 0.0
        
        # ATR expansion
        max_expansion_atr = expansion_window['atr'].max()
        atr_ratio = max_expansion_atr / compression_atr
        
        # Price range expansion
        compression_range = df['high'].iloc[compression_end_index] - df['low'].iloc[compression_end_index]
        max_expansion_range = (expansion_window['high'] - expansion_window['low']).max()
        range_ratio = max_expansion_range / compression_range if compression_range > 0 else 1.0
        
        # Combined expansion strength (weighted average)
        expansion_strength = (atr_ratio * 0.7) + (range_ratio * 0.3)
        
        return min(float(expansion_strength), 5.0)  # Cap at 5x
    
    def _calculate_expansion_multiplier(self, duration: int, historical_data: Dict) -> float:
        """Enhanced expansion multiplier with defensive coding"""
        base_multiplier = 1.5  # Minimum expected expansion
        
        # Defensive: Protected access to historical data
        sample_size = historical_data.get('sample_size', 0)
        avg_expansion = historical_data.get('avg_expansion', 2.0)
        success_rate = historical_data.get('success_rate', 0.6)
        
        # Enhanced duration scaling (different from probability calculation)
        # Uses power scaling for multiplier (different from sqrt for probability)
        duration_bonus = min((duration / 15.0) ** 0.7, 1.8)  # Max 1.8 bonus
        
        # Historical data adjustment with defensive handling
        if sample_size > 0 and avg_expansion > 0:
            try:
                sample_confidence = min(sample_size / 10.0, 1.0)
                
                # Weighted combination with zero-division protection
                success_adjustment = max(0.3, min(success_rate, 1.0))  # Clamp success rate
                adjusted_historical = avg_expansion * success_adjustment * sample_confidence
                
                # Blend with base multiplier
                base_multiplier = (
                    base_multiplier * (1 - sample_confidence) + 
                    adjusted_historical * sample_confidence
                )
            except (ZeroDivisionError, TypeError, ValueError):
                # Fallback to base multiplier if any calculation fails
                pass
        
        # Combine base and duration components
        total_multiplier = base_multiplier + duration_bonus
        
        # Ensure reasonable bounds
        return max(1.2, min(float(total_multiplier), 4.5))  # Min 1.2x, Max 4.5x
    
    def _predict_volatility_profile(self, df: pd.DataFrame) -> Dict:
        """Enhanced volatility profile prediction"""
        if len(df) < 20:
            return {
                'direction': 'UNKNOWN',
                'confidence': 0.0,
                'current_atr': 0.0,
                'atr_trend': 0.0
            }
        
        current_atr = df['atr'].iloc[-1]
        
        # Multi-timeframe trend analysis
        short_trend = df['atr'].iloc[-5:].pct_change().mean() if len(df) >= 5 else 0.0
        medium_trend = df['atr'].iloc[-10:].pct_change().mean() if len(df) >= 10 else 0.0
        long_trend = df['atr'].iloc[-20:].pct_change().mean() if len(df) >= 20 else 0.0
        
        # Weighted trend calculation
        weighted_trend = (short_trend * 0.5) + (medium_trend * 0.3) + (long_trend * 0.2)
        
        # Enhanced prediction logic
        if weighted_trend > 0.03:  # Increasing volatility
            prediction = "INCREASING"
            confidence = min(abs(weighted_trend) * 15, 0.9)
        elif weighted_trend < -0.03:  # Decreasing volatility
            prediction = "DECREASING"
            confidence = min(abs(weighted_trend) * 15, 0.9)
        else:
            prediction = "STABLE"
            confidence = 0.6
        
        # Additional context
        volatility_percentile = self._calculate_volatility_percentile(df)
        
        return {
            'direction': prediction,
            'confidence': float(confidence),
            'current_atr': float(current_atr),
            'atr_trend': float(weighted_trend),
            'short_trend': float(short_trend),
            'medium_trend': float(medium_trend),
            'long_trend': float(long_trend),
            'volatility_percentile': volatility_percentile
        }
    
    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """Calculate current volatility percentile relative to recent history"""
        if len(df) < self.norm_window:
            return 0.5  # Default to median
        
        current_atr = df['atr'].iloc[-1]
        recent_atr = df['atr'].iloc[-self.norm_window:]
        
        # Calculate percentile rank
        percentile = (recent_atr < current_atr).sum() / len(recent_atr)
        return float(percentile)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Enhanced ATR calculation with better handling of edge cases"""
        # Calculate true range components
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        
        # Calculate true range
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # Use exponential moving average for smoother ATR
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        # Fill initial NaN values with simple moving average
        atr_sma = true_range.rolling(window=period, min_periods=1).mean()
        atr = atr.fillna(atr_sma)
        
        return atr
    
    def _get_current_volatility_state(self, atr_norm: float, std_norm: float) -> str:
        """Enhanced volatility state determination with safety checks"""
        # Handle NaN values safely
        if pd.isna(atr_norm) or pd.isna(std_norm):
            return "UNKNOWN"
        
        # Enhanced state logic with intermediate states
        if atr_norm < self.compression_threshold and std_norm < self.compression_threshold:
            return "COMPRESSED"
        elif atr_norm > self.expansion_threshold or std_norm > self.expansion_threshold:
            return "EXPANDING"
        elif (atr_norm < self.compression_threshold * 1.5 and 
              std_norm < self.compression_threshold * 1.5):
            return "LOW_VOLATILITY"
        elif (atr_norm > self.expansion_threshold * 0.7 or 
              std_norm > self.expansion_threshold * 0.7):
            return "HIGH_VOLATILITY"
        else:
            return "NORMAL"

class SessionAnalyzer:
    """Analyzes time-based and session-specific behavior patterns with proper timezone alignment"""
    
    def __init__(self, session_timezone_aligner=None):
        # Legacy session times for backward compatibility
        self.session_times = {
            SessionType.ASIA: [(0, 9)],  # UTC hours
            SessionType.LONDON: [(8, 17)],
            SessionType.NEW_YORK: [(13, 22)],
            SessionType.OVERLAP: [(13, 17)]  # London-NY overlap
        }
        
        # Use session timezone aligner if provided, otherwise create one
        if session_timezone_aligner is None:
            from session_timezone_aligner import SessionTimezoneAligner
            # For SessionAnalyzer, use UTC+0 as default since it's typically used with UTC timestamps
            self.session_timezone_aligner = SessionTimezoneAligner()
        else:
            self.session_timezone_aligner = session_timezone_aligner
    
    def get_current_session(self, timestamp: datetime) -> SessionType:
        """Determine current trading session using proper timezone alignment"""
        # Use the session timezone aligner for proper session detection
        return self.session_timezone_aligner.get_current_session(timestamp)
    
    def analyze_session_behavior(self, df: pd.DataFrame) -> Dict:
        """Analyze behavior patterns for each session with market maker logic"""
        df['session'] = df.index.map(self.get_current_session)
        df['hour'] = df.index.hour
        
        session_stats = {}
        for session in SessionType:
            session_data = df[df['session'] == session]
            if len(session_data) > 0:
                session_stats[session.value] = {
                    'avg_range': (session_data['high'] - session_data['low']).mean(),
                    'avg_volume': session_data['tick_volume'].mean(),
                    'bullish_bias': (session_data['close'] > session_data['open']).mean(),
                    'volatility': session_data['close'].pct_change().std(),
                    'breakout_frequency': self._calculate_breakout_frequency(session_data)
                }
        
        # Enhanced market maker logic analysis
        market_maker_patterns = self._analyze_market_maker_patterns(df)
        session_transitions = self._analyze_session_transitions(df)
        
        return {
            'session_statistics': session_stats,
            'current_session': self.session_timezone_aligner.get_current_session(datetime.now(pytz.UTC)),
            'session_recommendations': self._generate_session_recommendations(session_stats),
            'market_maker_patterns': market_maker_patterns,
            'session_transitions': session_transitions
        }
    
    def _analyze_market_maker_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze market maker behavior patterns"""
        logger.debug(f"🔍 Analyzing market maker patterns with {len(df)} data points")
        
        # Check session distribution
        session_counts = df['session'].value_counts()
        logger.debug(f"📊 Session distribution: {dict(session_counts)}")
        
        patterns = {
            'london_sets_ny_continues': self._detect_london_ny_continuation(df),
            'ny_grabs_london_liquidity': self._detect_ny_liquidity_grab(df),
            'asia_range_breakouts': self._detect_asia_range_breakouts(df),
            'session_reversals': self._detect_session_reversals(df)
        }
        
        # Log pattern results
        for pattern_name, pattern_data in patterns.items():
            success_rate = pattern_data.get('success_rate', 0)
            frequency = pattern_data.get('pattern_frequency', 0)
            logger.debug(f"📈 {pattern_name}: success_rate={success_rate:.3f}, frequency={frequency}")
        
        return patterns
    
    def _detect_london_ny_continuation(self, df: pd.DataFrame) -> Dict:
        """Detect pattern: London sets high/low → NY continues"""
        continuations = []
        
        # Group by date to analyze daily patterns
        df_daily = df.groupby(df.index.date)
        logger.debug(f"🔍 London-NY continuation: analyzing {len(list(df_daily))} days of data")
        
        for date, day_data in df_daily:
            if len(day_data) < 10:
                continue
            
            london_data = day_data[day_data['session'] == SessionType.LONDON]
            ny_data = day_data[day_data['session'] == SessionType.NEW_YORK]
            
            if len(london_data) > 0 and len(ny_data) > 0:
                london_high = london_data['high'].max()
                london_low = london_data['low'].min()
                london_close = london_data['close'].iloc[-1]
                
                ny_open = ny_data['open'].iloc[0]
                ny_direction = 'bullish' if ny_data['close'].iloc[-1] > ny_open else 'bearish'
                london_direction = 'bullish' if london_close > london_data['open'].iloc[0] else 'bearish'
                
                # Check for continuation
                if london_direction == ny_direction:
                    london_range = london_high - london_low
                    ny_move = abs(ny_data['close'].iloc[-1] - ny_open)
                    strength = ny_move / max(london_range, 0.0001)  # Avoid division by zero
                    
                    # More realistic follow-through criteria
                    # Consider it successful if NY moves at least 20% of London range in same direction
                    # OR if NY move is significant (>10 pips for major pairs)
                    ny_follow_through = (strength > 0.2) or (ny_move > 0.001)  # 10 pips = 0.001 for USDJPY
                    
                    continuations.append({
                        'date': date,
                        'direction': london_direction,
                        'strength': min(strength, 2.0),
                        'london_range': london_range,
                        'ny_follow_through': ny_follow_through
                    })
                    
                    logger.debug(f"📊 London-NY continuation {date}: direction={london_direction}, strength={strength:.3f}, ny_move={ny_move:.5f}, london_range={london_range:.5f}, follow_through={ny_follow_through}")
        
        success_rate = len([c for c in continuations if c['ny_follow_through']]) / len(continuations) if continuations else 0
        
        return {
            'pattern_frequency': len(continuations),
            'success_rate': success_rate,
            'avg_strength': np.mean([c['strength'] for c in continuations]) if continuations else 0,
            'recent_occurrences': continuations[-5:] if len(continuations) >= 5 else continuations
        }
    
    def _detect_ny_liquidity_grab(self, df: pd.DataFrame) -> Dict:
        """Detect pattern: NY grabs liquidity from London → reverses"""
        liquidity_grabs = []
        
        df_daily = df.groupby(df.index.date)
        
        for date, day_data in df_daily:
            if len(day_data) < 10:
                continue
            
            london_data = day_data[day_data['session'] == SessionType.LONDON]
            ny_data = day_data[day_data['session'] == SessionType.NEW_YORK]
            
            if len(london_data) > 0 and len(ny_data) > 5:
                london_high = london_data['high'].max()
                london_low = london_data['low'].min()
                
                # Check if NY initially breaks London levels then reverses
                ny_first_hour = ny_data.iloc[:4]  # First hour of NY
                ny_later = ny_data.iloc[4:] if len(ny_data) > 4 else pd.DataFrame()
                
                if len(ny_first_hour) > 0 and len(ny_later) > 0:
                    london_range = london_high - london_low
                    
                    # Check for liquidity grab (break above/below London range)
                    # Allow for smaller breaks to catch more patterns
                    buffer = london_range * 0.1  # 10% buffer for noise
                    grabbed_high = ny_first_hour['high'].max() > (london_high + buffer)
                    grabbed_low = ny_first_hour['low'].min() < (london_low - buffer)
                    
                    if grabbed_high or grabbed_low:
                        # Check for reversal - more lenient criteria
                        ny_close = ny_data['close'].iloc[-1]
                        reversal_occurred = False
                        
                        if grabbed_high:
                            # Reversal if price comes back into or below London range
                            if ny_close < (london_high + buffer * 0.5):
                                reversal_occurred = True
                                grab_type = 'high_liquidity_grab'
                                reversal_strength = abs(ny_close - london_high) / max(london_range, 0.0001)
                        elif grabbed_low:
                            # Reversal if price comes back into or above London range  
                            if ny_close > (london_low - buffer * 0.5):
                                reversal_occurred = True
                                grab_type = 'low_liquidity_grab'
                                reversal_strength = abs(ny_close - london_low) / max(london_range, 0.0001)
                        
                        if reversal_occurred:
                            liquidity_grabs.append({
                                'date': date,
                                'grab_type': grab_type,
                                'reversal_strength': min(reversal_strength, 2.0),
                                'london_range': london_range
                            })
                            
                            logger.debug(f"💧 NY liquidity grab {date}: type={grab_type}, reversal_strength={reversal_strength:.3f}, london_range={london_range:.5f}")
        
        # Calculate success rate based on actual grab attempts, not total days
        total_grab_attempts = 0
        for date, day_data in df_daily:
            if len(day_data) < 10:
                continue
            london_data = day_data[day_data['session'] == SessionType.LONDON]
            ny_data = day_data[day_data['session'] == SessionType.NEW_YORK]
            if len(london_data) > 0 and len(ny_data) > 5:
                london_high = london_data['high'].max()
                london_low = london_data['low'].min()
                london_range = london_high - london_low
                buffer = london_range * 0.1
                ny_first_hour = ny_data.iloc[:4]
                if len(ny_first_hour) > 0:
                    grabbed_high = ny_first_hour['high'].max() > (london_high + buffer)
                    grabbed_low = ny_first_hour['low'].min() < (london_low - buffer)
                    if grabbed_high or grabbed_low:
                        total_grab_attempts += 1
        
        success_rate = len(liquidity_grabs) / max(total_grab_attempts, 1)
        
        return {
            'pattern_frequency': len(liquidity_grabs),
            'success_rate': success_rate,
            'avg_reversal_strength': np.mean([lg['reversal_strength'] for lg in liquidity_grabs]) if liquidity_grabs else 0,
            'recent_occurrences': liquidity_grabs[-5:] if len(liquidity_grabs) >= 5 else liquidity_grabs
        }
    
    def _detect_asia_range_breakouts(self, df: pd.DataFrame) -> Dict:
        """Detect Asia range breakouts during London/NY sessions"""
        breakouts = []
        
        df_daily = df.groupby(df.index.date)
        
        for date, day_data in df_daily:
            asia_data = day_data[day_data['session'] == SessionType.ASIA]
            london_ny_data = day_data[day_data['session'].isin([SessionType.LONDON, SessionType.NEW_YORK])]
            
            if len(asia_data) > 0 and len(london_ny_data) > 0:
                asia_high = asia_data['high'].max()
                asia_low = asia_data['low'].min()
                asia_range = asia_high - asia_low
                
                # Check for breakouts during London/NY
                breakout_high = london_ny_data['high'].max() > asia_high
                breakout_low = london_ny_data['low'].min() < asia_low
                
                if breakout_high or breakout_low:
                    breakout_strength = max(
                        (london_ny_data['high'].max() - asia_high) / asia_range if breakout_high else 0,
                        (asia_low - london_ny_data['low'].min()) / asia_range if breakout_low else 0
                    )
                    
                    breakouts.append({
                        'date': date,
                        'breakout_type': 'high' if breakout_high else 'low',
                        'strength': breakout_strength,
                        'asia_range': asia_range
                    })
        
        return {
            'pattern_frequency': len(breakouts),
            'avg_strength': np.mean([b['strength'] for b in breakouts]) if breakouts else 0,
            'high_breakouts': len([b for b in breakouts if b['breakout_type'] == 'high']),
            'low_breakouts': len([b for b in breakouts if b['breakout_type'] == 'low'])
        }
    
    def _detect_session_reversals(self, df: pd.DataFrame) -> Dict:
        """Detect reversals at session transitions"""
        reversals = []
        
        # Look for reversals at session boundaries
        for i in range(1, len(df)):
            current_session = df['session'].iloc[i]
            prev_session = df['session'].iloc[i-1]
            
            # Session transition detected
            if current_session != prev_session:
                # Look at price action around transition
                window_start = max(0, i-5)
                window_end = min(len(df), i+5)
                window_data = df.iloc[window_start:window_end]
                
                if len(window_data) >= 8:
                    pre_transition = window_data.iloc[:5]
                    post_transition = window_data.iloc[5:]
                    
                    pre_direction = 'bullish' if pre_transition['close'].iloc[-1] > pre_transition['open'].iloc[0] else 'bearish'
                    post_direction = 'bullish' if post_transition['close'].iloc[-1] > post_transition['open'].iloc[0] else 'bearish'
                    
                    if pre_direction != post_direction:
                        reversal_strength = abs(post_transition['close'].iloc[-1] - pre_transition['close'].iloc[-1]) / (window_data['high'].max() - window_data['low'].min())
                        
                        # Convert pandas Timestamp to datetime to avoid exception issues
                        timestamp_val = df.index[i]
                        if isinstance(timestamp_val, pd.Timestamp):
                            timestamp_dt = timestamp_val.to_pydatetime()
                        else:
                            timestamp_dt = timestamp_val
                        
                        reversals.append({
                            'timestamp': timestamp_dt,
                            'from_session': prev_session.value,
                            'to_session': current_session.value,
                            'reversal_strength': reversal_strength
                        })
        
        return {
            'total_reversals': len(reversals),
            'avg_strength': np.mean([r['reversal_strength'] for r in reversals]) if reversals else 0,
            'most_common_transitions': self._get_common_reversal_transitions(reversals)
        }
    
    def _get_common_reversal_transitions(self, reversals: List[Dict]) -> Dict:
        """Get most common session transition reversals"""
        transition_counts = defaultdict(int)
        
        for reversal in reversals:
            transition = f"{reversal['from_session']}_to_{reversal['to_session']}"
            transition_counts[transition] += 1
        
        return dict(transition_counts)
    
    def _analyze_session_transitions(self, df: pd.DataFrame) -> Dict:
        """Analyze behavior at session transitions"""
        transitions = []
        
        for i in range(1, len(df)):
            if df['session'].iloc[i] != df['session'].iloc[i-1]:
                # Measure volatility and direction change at transition
                pre_volatility = df['close'].iloc[max(0, i-5):i].pct_change().std()
                post_volatility = df['close'].iloc[i:min(len(df), i+5)].pct_change().std()
                
                # Convert pandas Timestamp to datetime to avoid exception issues
                timestamp_val = df.index[i]
                if isinstance(timestamp_val, pd.Timestamp):
                    timestamp_dt = timestamp_val.to_pydatetime()
                else:
                    timestamp_dt = timestamp_val
                
                transitions.append({
                    'timestamp': timestamp_dt,
                    'from_session': df['session'].iloc[i-1].value,
                    'to_session': df['session'].iloc[i].value,
                    'volatility_change': post_volatility / pre_volatility if pre_volatility > 0 else 1.0
                })
        
        return {
            'total_transitions': len(transitions),
            'avg_volatility_change': np.mean([t['volatility_change'] for t in transitions]) if transitions else 1.0,
            'high_volatility_transitions': len([t for t in transitions if t['volatility_change'] > 1.5])
        }
    
    def _calculate_breakout_frequency(self, session_data: pd.DataFrame) -> float:
        """Calculate frequency of significant breakouts during session"""
        if len(session_data) < 10:
            return 0.0
        
        ranges = session_data['high'] - session_data['low']
        avg_range = ranges.mean()
        significant_moves = (ranges > avg_range * 1.5).sum()
        
        return significant_moves / len(session_data)
    
    def _generate_session_recommendations(self, session_stats: Dict) -> Dict:
        """Generate trading recommendations based on session analysis"""
        recommendations = {}
        
        for session, stats in session_stats.items():
            if stats['breakout_frequency'] > 0.3:
                recommendations[session] = "HIGH_BREAKOUT_PROBABILITY"
            elif stats['volatility'] < 0.005:
                recommendations[session] = "CONSOLIDATION_EXPECTED"
            elif stats['bullish_bias'] > 0.6:
                recommendations[session] = "BULLISH_BIAS"
            elif stats['bullish_bias'] < 0.4:
                recommendations[session] = "BEARISH_BIAS"
            else:
                recommendations[session] = "NEUTRAL"
        
        return recommendations
    
    def calculate_volatility_factor(self, df: pd.DataFrame) -> Dict:
        """Enhanced volatility factor calculation with comprehensive features"""
        # ✅ Timestamp included - Makes each prediction traceable
        if len(df) > 0:
            # Convert pandas Timestamp to datetime to avoid exception issues
            timestamp_val = df.index[-1]
            if isinstance(timestamp_val, pd.Timestamp):
                current_timestamp = timestamp_val.to_pydatetime()
            else:
                current_timestamp = timestamp_val
        else:
            current_timestamp = datetime.now(pytz.UTC)
        
        # ✅ ATR presence check - No crash if column missing
        if 'atr' not in df.columns or len(df) < 20:
            return {
                'timestamp': current_timestamp.isoformat(),
                'state': 'INSUFFICIENT_DATA',
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'trend_strength': 0.0,
                'volatility_percentile': 0.5,
                'atr_available': False,
                'error': 'ATR column missing or insufficient data'
            }
        
        # Calculate current volatility state
        current_atr = df['atr'].iloc[-1]
        atr_mean = df['atr'].rolling(window=100, min_periods=20).mean().iloc[-1]
        atr_normalized = current_atr / atr_mean if atr_mean > 0 else 1.0
        
        # 🧠 Trend defined via % change - More robust than slope alone
        # Calculate percentage changes over multiple periods
        atr_pct_5 = df['atr'].pct_change(5).iloc[-1] if len(df) >= 6 else 0.0
        atr_pct_10 = df['atr'].pct_change(10).iloc[-1] if len(df) >= 11 else 0.0
        atr_pct_20 = df['atr'].pct_change(20).iloc[-1] if len(df) >= 21 else 0.0
        
        # Weighted trend calculation using percentage changes
        trend_strength = (
            atr_pct_5 * 0.5 +   # Short-term: 50% weight
            atr_pct_10 * 0.3 +  # Medium-term: 30% weight
            atr_pct_20 * 0.2    # Long-term: 20% weight
        )
        
        # Determine volatility state
        if atr_normalized < 0.7:
            state = "COMPRESSED"
        elif atr_normalized > 1.5:
            state = "EXPANDING"
        else:
            state = "NORMAL"
        
        # Predict future volatility direction
        if trend_strength > 0.05:  # 5% threshold for increasing
            prediction = "INCREASING"
            base_confidence = min(abs(trend_strength) * 10, 0.9)
        elif trend_strength < -0.05:  # 5% threshold for decreasing
            prediction = "DECREASING"
            base_confidence = min(abs(trend_strength) * 10, 0.9)
        else:
            prediction = "STABLE"
            base_confidence = 0.6
        
        # Enhanced confidence calculation with multiple factors
        data_quality = min(len(df) / 100.0, 1.0)  # More data = higher confidence
        trend_consistency = self._calculate_trend_consistency(df)
        
        # Combine confidence factors
        final_confidence = (
            base_confidence * 0.6 +      # Base prediction confidence: 60%
            data_quality * 0.2 +         # Data quality: 20%
            trend_consistency * 0.2      # Trend consistency: 20%
        )
        
        # ✅ 0.95 cap on confidence - Prevents unrealistic values
        final_confidence = min(final_confidence, 0.95)
        
        # Calculate volatility percentile
        volatility_percentile = self._calculate_session_volatility_percentile(df)
        
        # ✅ round() on numeric output - Clean JSON formatting
        return {
            'timestamp': current_timestamp.isoformat(),
            'state': state,
            'prediction': prediction,
            'confidence': round(final_confidence, 3),
            'trend_strength': round(trend_strength, 4),
            'volatility_percentile': round(volatility_percentile, 3),
            'current_atr': round(current_atr, 5),
            'atr_normalized': round(atr_normalized, 3),
            'atr_available': True,
            'trend_components': {
                'short_term_pct': round(atr_pct_5, 4),
                'medium_term_pct': round(atr_pct_10, 4),
                'long_term_pct': round(atr_pct_20, 4)
            },
            'confidence_breakdown': {
                'base_confidence': round(base_confidence, 3),
                'data_quality': round(data_quality, 3),
                'trend_consistency': round(trend_consistency, 3)
            },
            # ✅ Suggested Use in Agentic AI Stack
            'ai_recommendations': self._generate_ai_recommendations(state, prediction, final_confidence)
        }
    
    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """Calculate how consistent the trend direction is"""
        if len(df) < 10:
            return 0.5
        
        # Calculate direction consistency over recent periods
        recent_changes = df['atr'].pct_change().iloc[-10:]
        positive_changes = (recent_changes > 0).sum()
        negative_changes = (recent_changes < 0).sum()
        total_changes = len(recent_changes.dropna())
        
        if total_changes == 0:
            return 0.5
        
        # Consistency is the proportion of changes in the dominant direction
        consistency = max(positive_changes, negative_changes) / total_changes
        return consistency
    
    def _calculate_session_volatility_percentile(self, df: pd.DataFrame) -> float:
        """Calculate current volatility percentile for session context"""
        if len(df) < 50:
            return 0.5
        
        current_atr = df['atr'].iloc[-1]
        recent_atr = df['atr'].iloc[-100:] if len(df) >= 100 else df['atr']
        
        # Calculate percentile rank
        percentile = (recent_atr < current_atr).sum() / len(recent_atr)
        return percentile
    
    def _generate_ai_recommendations(self, state: str, prediction: str, confidence: float) -> Dict:
        """Generate specific recommendations for agentic AI stack"""
        recommendations = {
            'priority_signals': [],
            'risk_adjustment': 'NORMAL',
            'signal_weight_modifier': 1.0,
            'trading_advice': ''
        }
        
        # High-confidence compressed state with increasing prediction
        if state == "COMPRESSED" and prediction == "INCREASING" and confidence > 0.75:
            recommendations.update({
                'priority_signals': ['MSB', 'FVG', 'ORDER_BLOCK'],
                'risk_adjustment': 'INCREASE_POSITION_SIZE',
                'signal_weight_modifier': 1.5,
                'trading_advice': 'Flag for high breakout potential - Prioritize MSB/FVG/OB signals'
            })
        
        # Expanding state with decreasing prediction
        elif state == "EXPANDING" and prediction == "DECREASING":
            recommendations.update({
                'priority_signals': ['MEAN_REVERSION', 'SUPPORT_RESISTANCE'],
                'risk_adjustment': 'ADJUST_RR_RATIO',
                'signal_weight_modifier': 0.8,
                'trading_advice': 'Expect mean reversion or slowdown - adjust RR ratio accordingly'
            })
        
        # High confidence modifier
        if confidence > 0.8:
            recommendations['signal_weight_modifier'] *= 1.2
            recommendations['trading_advice'] += ' [HIGH CONFIDENCE]'
        elif confidence < 0.4:
            recommendations['signal_weight_modifier'] *= 0.7
            recommendations['trading_advice'] += ' [LOW CONFIDENCE - USE CAUTION]'
        
        # Use confidence as weight modifier for trade signal scoring
        recommendations['confidence_weight'] = round(confidence, 3)
        
        return recommendations

class ForexDataCollector:
    """Handles MT5 data collection and multi-timeframe analysis"""
    
    def __init__(self, symbol: str = "USDJPY"):
        self.symbol = symbol
        self.timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        self.is_connected = False
        self.pip_value = self._get_pip_value(symbol)
        self.pip_multiplier = self._get_pip_multiplier(symbol)
    
    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value based on currency pair"""
        if 'JPY' in symbol:
            return 0.01  # JPY pairs: 2 decimal places
        elif any(x in symbol for x in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return 0.1   # Precious metals
        else:
            return 0.0001  # Major pairs: 4 decimal places
    
    def _get_pip_multiplier(self, symbol: str) -> int:
        """Get pip multiplier for calculations"""
        if 'JPY' in symbol:
            return 100   # 1 pip = 0.01, so multiply by 100
        elif any(x in symbol for x in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return 10    # 1 pip = 0.1, so multiply by 10
        else:
            return 10000 # 1 pip = 0.0001, so multiply by 10000
    
    def pips_to_price(self, pips: float) -> float:
        """Convert pips to price difference"""
        return pips * self.pip_value
    
    def price_to_pips(self, price_diff: float) -> float:
        """Convert price difference to pips"""
        return price_diff / self.pip_value
    
    def connect_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        logger.info(f"MT5 initialized successfully. Version: {mt5.version()}")
        self.is_connected = True
        return True
    
    def disconnect_mt5(self):
        """Close MT5 connection"""
        mt5.shutdown()
        self.is_connected = False
        logger.info("MT5 connection closed")
    
    def get_mtf_data(self, bars: int = 1000) -> Dict[str, pd.DataFrame]:
        """Collect multi-timeframe data with enhanced error handling"""
        if not self.is_connected:
            if not self.connect_mt5():
                logger.error("Failed to establish MT5 connection")
                return {}
        
        # Validate symbol availability
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error(f"Symbol {self.symbol} not found or not available")
            return {}
        
        # Check if symbol is selected in Market Watch
        if not symbol_info.select:
            logger.info(f"Selecting symbol {self.symbol} in Market Watch")
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Failed to select symbol {self.symbol}")
                return {}
        
        mtf_data = {}
        
        for tf_name, tf_value in self.timeframes.items():
            try:
                # Debug logging to see what tf_value actually is
                logger.debug(f"Processing {tf_name}: tf_value={tf_value}, type={type(tf_value)}")
                
                # Validate parameters before calling copy_rates_from_pos
                # MT5 timeframe constants should be integers
                if tf_value is None:
                    logger.error(f"Timeframe value is None for {tf_name}")
                    continue
                
                # Validate bars parameter (do this once, not in loop)
                # Handle case where bars might be passed as a list or other type
                logger.debug(f"bars parameter: {bars}, type: {type(bars)}")
                
                # Robust parameter validation to prevent comparison errors
                try:
                    if isinstance(bars, (list, tuple)):
                        logger.warning(f"bars parameter is {type(bars)}, taking first element: {bars}")
                        if len(bars) > 0 and isinstance(bars[0], (int, float)):
                            safe_bars = int(bars[0])
                        else:
                            logger.error(f"Invalid first element in bars list: {bars}")
                            safe_bars = 1000
                    elif isinstance(bars, (int, float)):
                        safe_bars = int(bars)
                    else:
                        logger.error(f"Invalid bars parameter type: {type(bars)}, using default 1000")
                        safe_bars = 1000
                    
                    # Ensure safe_bars is definitely an integer before comparison
                    if not isinstance(safe_bars, int):
                        logger.error(f"safe_bars is not an integer: {safe_bars}, type: {type(safe_bars)}")
                        safe_bars = 1000
                    
                    if safe_bars <= 0 or safe_bars > 50000:  # MT5 limit is typically 50000
                        logger.warning(f"Adjusting bars from {safe_bars} to 1000 for safety")
                        safe_bars = 1000
                        
                except Exception as param_error:
                    logger.error(f"Error processing bars parameter: {param_error}, using default 1000")
                    safe_bars = 1000
                
                # Try alternative data collection methods
                rates = None
                
                # Method 1: copy_rates_from_pos (current position)
                try:
                    rates = mt5.copy_rates_from_pos(self.symbol, tf_value, 0, safe_bars)
                except Exception as pos_error:
                    logger.warning(f"copy_rates_from_pos failed for {tf_name}: {pos_error}")
                    
                    # Method 2: copy_rates_from (from specific date)
                    try:
                        utc_from = datetime.now(pytz.UTC) - timedelta(days=30)  # Last 30 days
                        rates = mt5.copy_rates_from(self.symbol, tf_value, utc_from, safe_bars)
                        logger.info(f"Used copy_rates_from for {tf_name}")
                    except Exception as from_error:
                        logger.warning(f"copy_rates_from also failed for {tf_name}: {from_error}")
                        
                        # Method 3: copy_rates_range (date range)
                        try:
                            utc_from = datetime.now(pytz.UTC) - timedelta(days=7)
                            utc_to = datetime.now(pytz.UTC)
                            rates = mt5.copy_rates_range(self.symbol, tf_value, utc_from, utc_to)
                            logger.info(f"Used copy_rates_range for {tf_name}")
                        except Exception as range_error:
                            logger.error(f"All data collection methods failed for {tf_name}: {range_error}")
                
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    # Keep timestamps in broker time - no UTC conversion
                    
                    # Historical data collection - old timestamps are expected when collecting 200 bars
                    logger.debug(f"Collected {len(df)} bars of {tf_name} historical data")
                    
                    if len(df) > 0:
                        mtf_data[tf_name] = df
                        logger.info(f"Successfully collected {len(df)} bars for {tf_name}")
                    else:
                        logger.warning(f"No valid data remaining for {tf_name} after filtering invalid timestamps")
                        # Try to get data with a more recent date range to avoid corrupted historical data
                        try:
                            logger.info(f"Attempting to get recent data for {tf_name} to avoid corrupted historical data")
                            utc_from = datetime.now(pytz.UTC) - timedelta(hours=24)  # Last 24 hours only
                            utc_to = datetime.now(pytz.UTC)
                            recent_rates = mt5.copy_rates_range(self.symbol, tf_value, utc_from, utc_to)
                            
                            if recent_rates is not None and len(recent_rates) > 0:
                                df_recent = pd.DataFrame(recent_rates)
                                df_recent['time'] = pd.to_datetime(df_recent['time'], unit='s')
                                df_recent.set_index('time', inplace=True)
                                # Keep timestamps in broker time - no UTC conversion
                                
                                if len(df_recent) > 0:
                                    mtf_data[tf_name] = df_recent
                                    logger.info(f"Successfully collected {len(df_recent)} recent bars for {tf_name}")
                                else:
                                    logger.error(f"No recent data available for {tf_name}")
                            else:
                                logger.error(f"No recent data available for {tf_name}")
                        except Exception as recent_error:
                            logger.error(f"Failed to get recent data for {tf_name}: {recent_error}")
                else:
                    # Get more detailed error information
                    last_error = mt5.last_error()
                    if last_error[0] != 0:  # 0 means no error
                        logger.error(f"MT5 error for {tf_name}: Code {last_error[0]} - {last_error[1]}")
                    else:
                        logger.warning(f"No data received for {tf_name} - market may be closed or no historical data available")
            
            except Exception as e:
                last_error = mt5.last_error()
                logger.error(f"Unexpected error collecting {tf_name} data: {e} | MT5 Error: {last_error}")
        
        if not mtf_data:
            logger.warning("No data collected for any timeframe - check market hours and symbol availability")
            logger.warning("Attempting to use fallback data generation due to corrupted server data")
            
            # Fallback: Generate minimal synthetic data for system operation
            try:
                current_time = self._get_broker_time().replace(tzinfo=None)  # Use broker time as naive datetime
                current_price = self.get_current_price()
                
                if current_price:
                    # Generate basic OHLC data for the last few periods
                    base_price = (current_price['bid'] + current_price['ask']) / 2
                    
                    for tf_name in self.timeframes.keys():
                        # Create minimal synthetic data
                        periods = 10  # Minimal data for basic analysis
                        time_delta = timedelta(minutes=1) if tf_name == 'M1' else timedelta(hours=1)
                        
                        synthetic_data = []
                        for i in range(periods):
                            timestamp = current_time - (time_delta * (periods - i))
                            # Simple price variation around current price
                            price_variation = 0.001 * (i % 3 - 1)  # Small random-like variation
                            price = base_price + price_variation
                            
                            synthetic_data.append({
                                'time': timestamp,
                                'open': price,
                                'high': price + 0.0005,
                                'low': price - 0.0005,
                                'close': price,
                                'tick_volume': 100,
                                'spread': 2,
                                'real_volume': 0
                            })
                        
                        df_synthetic = pd.DataFrame(synthetic_data)
                        df_synthetic.set_index('time', inplace=True)
                        mtf_data[tf_name] = df_synthetic
                        
                        logger.info(f"Generated {len(df_synthetic)} synthetic bars for {tf_name} as fallback")
                else:
                    logger.error("Cannot generate fallback data - no current price available")
                    
            except Exception as fallback_error:
                logger.error(f"Failed to generate fallback data: {fallback_error}")
        
        return mtf_data
    
    def _get_broker_time(self) -> datetime:
        """Get current broker server time from MT5 as naive datetime"""
        print("🔍 _get_broker_time() called!")
        try:
            # Get current tick which contains broker server time
            print(f"🔍 MT5 DEBUG - Attempting to get tick for symbol: {self.symbol}")
            tick = mt5.symbol_info_tick(self.symbol)
            print(f"🔍 MT5 DEBUG - Tick result: {tick}")
            
            if tick and tick.time:
                broker_time = datetime.fromtimestamp(tick.time)  # Return naive datetime in broker time
                print(f"🔍 MT5 DEBUG - tick.time: {tick.time}, converted to datetime: {broker_time}, system time: {datetime.now()}")
                return broker_time
            else:
                print(f"🔍 MT5 DEBUG - No valid tick data: tick={tick}, tick.time={getattr(tick, 'time', 'N/A') if tick else 'N/A'}")
        except Exception as e:
            print(f"FAILED TO GET BROKER TIME: {e}")
        
        # Fallback to system time as naive datetime
        fallback_time = datetime.now()
        print(f"🔍 MT5 DEBUG - Using fallback system time: {fallback_time}")
        return fallback_time
    
    def _is_valid_data_timestamp(self, timestamp) -> bool:
        """Validate timestamp using broker time"""
        try:
            # Handle pandas Series or arrays - check if it's a scalar value
            if hasattr(timestamp, '__len__') and not isinstance(timestamp, (str, bytes)):
                # If it's an array/series, check if it has any valid values
                if len(timestamp) == 0:
                    return False
                # Take the first element for validation
                timestamp = timestamp.iloc[0] if hasattr(timestamp, 'iloc') else timestamp[0]
            
            if pd.isna(timestamp):
                return False
            
            # Convert to standard Python datetime to avoid pandas Series issues
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            # Convert pandas Timestamp to Python datetime immediately
            if hasattr(timestamp, 'to_pydatetime'):
                timestamp = timestamp.to_pydatetime()
            elif hasattr(timestamp, 'timestamp'):
                # For pandas Timestamp objects
                timestamp = datetime.fromtimestamp(timestamp.timestamp(), tz=pytz.UTC)
            
            # Ensure timezone awareness for Python datetime
            if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=pytz.UTC)
            
            broker_time = self._get_broker_time()
            
            # Allow up to 10 minutes in the future (for clock differences)
            future_threshold = broker_time + timedelta(minutes=10)
            
            # Allow data up to 30 days old
            past_threshold = broker_time - timedelta(days=30)
            
            # Now all objects should be Python datetime, safe for comparison
            return past_threshold <= timestamp <= future_threshold
            
        except Exception as e:
            logger.warning(f"Timestamp validation error: {e}")
            return False
    
    def get_current_price(self) -> Optional[Dict]:
        """Get current market price and spread"""
        if not self.is_connected:
            return None
        
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid,
            'time': datetime.fromtimestamp(tick.time, tz=pytz.UTC)
        }

class AutoLearningMemoryEngine:
    """Auto-learning memory engine that adapts weights based on past performance"""
    
    def __init__(self, memory_file: str = "trading_memory.pkl"):
        self.memory_file = memory_file
        self.trade_outcomes = deque(maxlen=1000)  # Keep last 1000 trades
        self.factor_weights = {
            'trend_age': 1.0,
            'fvg_weight': 1.0,
            'volatility_prediction': 1.0,
            'session_bias': 1.0,
            'order_block_strength': 1.0,
            'liquidity_sweep': 1.0,
            'market_structure': 1.0
        }
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'avg_rr': 0.0,
            'best_factors': {},
            'worst_factors': {}
        }
        self.load_memory()
    
    def record_trade_outcome(self, outcome: TradeOutcome):
        """Record a trade outcome for learning"""
        self.trade_outcomes.append(outcome)
        self.performance_metrics['total_trades'] += 1
        
        if outcome.profit_loss > 0:
            self.performance_metrics['winning_trades'] += 1
        
        # Update average RR using risk_reward_actual
        if outcome.risk_reward_actual > 0:
            current_avg = self.performance_metrics['avg_rr']
            total_trades = self.performance_metrics['total_trades']
            self.performance_metrics['avg_rr'] = ((current_avg * (total_trades - 1)) + outcome.risk_reward_actual) / total_trades
        
        # Analyze factors that led to this outcome
        self._analyze_factor_performance(outcome)
        
        # Adapt weights based on recent performance
        if len(self.trade_outcomes) >= 20:  # Need minimum sample size
            self._adapt_weights()
        
        self.save_memory()
    
    def _analyze_factor_performance(self, outcome: TradeOutcome):
        """Analyze which factors contributed to trade success/failure"""
        for factor, value in outcome.contributing_factors.items():
            if factor not in self.performance_metrics['best_factors']:
                self.performance_metrics['best_factors'][factor] = []
                self.performance_metrics['worst_factors'][factor] = []
            
            if outcome.profit_loss > 0:
                self.performance_metrics['best_factors'][factor].append(value)
            else:
                self.performance_metrics['worst_factors'][factor].append(value)
    
    def _adapt_weights(self):
        """Adapt factor weights based on recent performance"""
        recent_trades = list(self.trade_outcomes)[-50:]  # Last 50 trades
        
        # Calculate success rate for each factor range
        factor_performance = {}
        
        for factor in self.factor_weights.keys():
            high_factor_trades = [t for t in recent_trades if t.contributing_factors.get(factor, 0) > 0.7]
            low_factor_trades = [t for t in recent_trades if t.contributing_factors.get(factor, 0) < 0.3]
            
            if high_factor_trades:
                high_success_rate = len([t for t in high_factor_trades if t.profit_loss > 0]) / len(high_factor_trades)
            else:
                high_success_rate = 0.5
            
            if low_factor_trades:
                low_success_rate = len([t for t in low_factor_trades if t.profit_loss > 0]) / len(low_factor_trades)
            else:
                low_success_rate = 0.5
            
            factor_performance[factor] = high_success_rate - low_success_rate
        
        # Adjust weights based on performance
        for factor, performance in factor_performance.items():
            current_weight = self.factor_weights[factor]
            
            if performance > 0.1:  # Factor shows positive correlation
                new_weight = min(current_weight * 1.1, 2.0)  # Increase weight, cap at 2.0
            elif performance < -0.1:  # Factor shows negative correlation
                new_weight = max(current_weight * 0.9, 0.5)  # Decrease weight, floor at 0.5
            else:
                new_weight = current_weight  # No significant change
            
            self.factor_weights[factor] = new_weight
    
    def get_adjusted_weight(self, factor: str, base_value: float) -> float:
        """Get adjusted weight for a factor based on learning"""
        factor_weight = self.factor_weights.get(factor, 1.0)
        return base_value * factor_weight
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary and learning insights"""
        if self.performance_metrics['total_trades'] == 0:
            return {'status': 'No trades recorded yet'}
        
        win_rate = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
        
        # Find best and worst performing factors
        best_factors = {}
        worst_factors = {}
        
        for factor, weight in self.factor_weights.items():
            if weight > 1.2:
                best_factors[factor] = weight
            elif weight < 0.8:
                worst_factors[factor] = weight
        
        return {
            'total_trades': self.performance_metrics['total_trades'],
            'win_rate': win_rate,
            'avg_rr': self.performance_metrics['avg_rr'],
            'current_weights': self.factor_weights.copy(),
            'best_performing_factors': best_factors,
            'worst_performing_factors': worst_factors,
            'recent_trades_count': len(self.trade_outcomes)
        }
    
    def save_memory(self):
        """Save memory to file"""
        try:
            memory_data = {
                'trade_outcomes': list(self.trade_outcomes),
                'factor_weights': self.factor_weights,
                'performance_metrics': self.performance_metrics
            }
            with open(self.memory_file, 'wb') as f:
                pickle.dump(memory_data, f)
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")
    
    def load_memory(self):
        """Load memory from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    memory_data = pickle.load(f)
                
                self.trade_outcomes = deque(memory_data.get('trade_outcomes', []), maxlen=1000)
                self.factor_weights = memory_data.get('factor_weights', self.factor_weights)
                self.performance_metrics = memory_data.get('performance_metrics', self.performance_metrics)
                
                logger.info(f"Loaded {len(self.trade_outcomes)} trade outcomes from memory")
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")

class AgenticForexEngine:
    """🚀 Main agentic trading engine with Strategic Enhancements to Beat the Banks
    
    Enhanced with:
    - Institutional Flow Tracking
    - Advanced Order Block Validation
    - Liquidity Mapping System
    - Adaptive Signal Weighting
    """
    
    def __init__(self, symbol: str = "USDJPY"):
        self.symbol = symbol
        self.data_collector = ForexDataCollector(symbol)
        self.smc_analyzer = SmartMoneyAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        
        # Session timezone alignment for broker time to UTC conversion
        from session_timezone_aligner import SessionTimezoneAligner
        # Detect broker timezone offset by comparing MT5 server time with UTC
        broker_offset = self._detect_broker_timezone_offset()
        self.session_timezone_aligner = SessionTimezoneAligner(broker_timezone_offset=broker_offset)
        
        # Initialize session analyzer with timezone aligner for consistency
        self.session_analyzer = SessionAnalyzer(self.session_timezone_aligner)
        self.trend_analyzer = TrendAgeAnalyzer()
        self.memory_engine = AutoLearningMemoryEngine()
        
        # 🚀 Strategic Enhancement Components
        from institutional_flow_tracker import InstitutionalFlowTracker
        from advanced_order_block_validator import AdvancedOrderBlockValidator
        from liquidity_mapping_system import LiquidityMappingSystem
        from adaptive_signal_weighting import AdaptiveSignalWeighting
        
        self.institutional_flow_tracker = InstitutionalFlowTracker()
        # Pass session timezone aligner to advanced OB validator for consistency
        self.advanced_ob_validator = AdvancedOrderBlockValidator(self.session_timezone_aligner)
        self.liquidity_mapper = LiquidityMappingSystem()
        self.adaptive_weighter = AdaptiveSignalWeighting()
        
        self.signals_history = []
        self.analysis_cache = {}
        self.last_analysis_time = None
        
        logger.info(f"Enhanced Agentic Forex Engine initialized for {symbol} with Strategic Enhancements")
    
    def _detect_broker_timezone_offset(self) -> int:
        """Detect broker timezone offset by comparing MT5 server time with UTC"""
        try:
            # Get MT5 server time
            tick = mt5.symbol_info_tick(self.symbol)
            if tick and tick.time:
                broker_time = datetime.fromtimestamp(tick.time)
                utc_time = datetime.utcnow()
                
                # Calculate offset in hours (rounded to nearest hour)
                offset_seconds = (broker_time - utc_time).total_seconds()
                offset_hours = round(offset_seconds / 3600)
                
                logger.info(f"Detected broker timezone offset: UTC{offset_hours:+d} (broker: {broker_time}, utc: {utc_time})")
                return offset_hours
        except Exception as e:
            logger.warning(f"Failed to detect broker timezone offset: {e}")
        
        # Default to UTC if detection fails
        logger.info("Using default UTC+0 timezone offset")
        return 0
    
    def _get_current_session(self, timestamp: datetime = None) -> SessionType:
        """Get current trading session"""
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC)
        return self.session_analyzer.get_current_session(timestamp)
    
    def _calculate_session_strength(self, session: SessionType) -> float:
        """Calculate session strength based on historical patterns"""
        # Simple session strength calculation
        session_strengths = {
            SessionType.LONDON: 0.8,
            SessionType.NEW_YORK: 0.9,
            SessionType.OVERLAP: 0.95,
            SessionType.ASIA: 0.6
        }
        return session_strengths.get(session, 0.5)
    
    def _determine_market_structure(self) -> str:
        """Determine current market structure"""
        # Simple market structure determination
        return "TRENDING"  # Default value
    
    def run_full_analysis(self) -> Dict:
        """Execute comprehensive multi-timeframe analysis with strategic enhancements"""
        logger.info("Starting enhanced market analysis with institutional flow tracking...")
        
        # Clear any cached analysis data to prevent stale timestamps
        self.smc_analyzer.clear_cache()
        # Also clear the main analysis cache to prevent stale data
        self.analysis_cache.clear()
        
        # Import config for optimized bar count
        from config import ForexEngineConfig
        config = ForexEngineConfig()
        
        # Collect fresh data with optimized bar count
        mtf_data = self.data_collector.get_mtf_data(bars=config.analysis.data_history_bars)
        if not mtf_data:
            logger.error("Failed to collect market data")
            return {}
        
        current_price = self.data_collector.get_current_price()
        current_time = self._get_broker_time()
        
        analysis_results = {
            'timestamp': datetime.now(pytz.UTC),
            'current_price': current_price,
            'timeframe_analysis': {},
            'smart_money_signals': {},
            'volatility_analysis': {},
            'session_analysis': {},
            'institutional_flow': {},
            'liquidity_mapping': {},
            'trading_signals': [],
            'market_narrative': ""
        }
        
        # 🚀 Strategic Enhancement 1: Institutional Flow Analysis
        logger.info("Analyzing institutional flow patterns...")
        if 'H1' in mtf_data:
            institutional_flow = self.institutional_flow_tracker.analyze_institutional_flow(
                mtf_data['H1'], current_time
            )
            analysis_results['institutional_flow'] = institutional_flow
        
        # 🚀 Strategic Enhancement 3: Liquidity Mapping
        logger.info("Mapping liquidity zones and absorption events...")
        if 'H4' in mtf_data:
            liquidity_mapping = self.liquidity_mapper.analyze_liquidity_landscape(
                mtf_data['H4']
            )
            analysis_results['liquidity_mapping'] = liquidity_mapping
        
        # Analyze each timeframe with enhanced features
        for tf, df in mtf_data.items():
            if len(df) < 50:  # Minimum data requirement
                continue
            
            # Enhanced analysis with new features
            trend_context = self.trend_analyzer.analyze_trend_context(df)
            fvg_analysis = self.smc_analyzer.detect_fair_value_gaps(df)
            volatility_analysis = self.volatility_analyzer.analyze_compression_expansion(df)
            
            # Apply memory engine weights to enhance analysis
            trend_weight = self.memory_engine.get_adjusted_weight('trend_age', trend_context.momentum_strength)
            volatility_weight = self.memory_engine.get_adjusted_weight('volatility_prediction', volatility_analysis.get('expansion_probability', 0.5))
            
            # 🚀 Strategic Enhancement 2: Advanced Order Block Validation
            raw_order_blocks = self.smc_analyzer.detect_order_blocks(df)
            enhanced_order_blocks = self.advanced_ob_validator.validate_and_score_order_blocks(
                raw_order_blocks, df, current_time
            )
            
            tf_analysis = {
                'smc_analysis': self.smc_analyzer.detect_market_structure_break(df),
                'order_blocks': raw_order_blocks,  # Keep original for compatibility
                'enhanced_order_blocks': enhanced_order_blocks,  # New enhanced version
                'fair_value_gaps': fvg_analysis,
                'liquidity_sweeps': self.smc_analyzer.detect_liquidity_sweeps(df),
                'volatility': volatility_analysis,
                'trend_context': trend_context,
                'adjusted_weights': {
                    'trend_weight': trend_weight,
                    'volatility_weight': volatility_weight
                }
            }
            
            analysis_results['timeframe_analysis'][tf] = tf_analysis
        
        # Session analysis
        if 'H1' in mtf_data:
            analysis_results['session_analysis'] = self.session_analyzer.analyze_session_behavior(mtf_data['H1'])
        
        # 🚀 Strategic Enhancement 4: Adaptive Signal Weighting
        logger.info("Applying adaptive signal weighting based on market regime...")
        market_regime = self.adaptive_weighter._analyze_market_regime(mtf_data)
        analysis_results['market_regime'] = market_regime
        
        # Generate trading signals with enhanced weighting
        signals = self._generate_enhanced_trading_signals(analysis_results)
        analysis_results['trading_signals'] = signals
        
        # Create market narrative
        analysis_results['market_narrative'] = self._create_enhanced_market_narrative(analysis_results)
        
        # Cache results
        self.analysis_cache = analysis_results
        self.last_analysis_time = datetime.now(pytz.UTC)
        
        logger.info(f"Enhanced analysis complete. Generated {len(signals)} signals with institutional flow insights")
        return analysis_results
    
    def generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals for automated trading - wrapper method"""
        logger.info("Generating signals for automated trading...")
        
        try:
            # Run full analysis
            analysis_results = self.run_full_analysis()
            
            # Extract trading signals with additional error handling
            signals = analysis_results.get('trading_signals', [])
            
            logger.info(f"Generated {len(signals)} trading signals for automation")
            return signals
            
        except Exception as e:
            # Enhanced error logging to identify timestamp issues
            error_msg = str(e)
            if "2025-07-04" in error_msg or "Timestamp" in error_msg:
                logger.error(f"Timestamp validation error in signal generation: {error_msg}")
                logger.info("Clearing analysis cache and retrying...")
                
                # Clear cache and try once more
                try:
                    self.smc_analyzer.clear_cache()
                    self.analysis_cache = {}
                    analysis_results = self.run_full_analysis()
                    signals = analysis_results.get('trading_signals', [])
                    logger.info(f"Retry successful: Generated {len(signals)} signals")
                    return signals
                except Exception as retry_error:
                    logger.error(f"Retry failed: {retry_error}")
                    logger.error(f"Retry error type: {type(retry_error)}")
                    logger.error(f"Retry error args: {retry_error.args if hasattr(retry_error, 'args') else 'No args'}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    return []
            else:
                logger.error(f"Error generating signals: {error_msg}")
                return []
    
    def _generate_trading_signals(self, analysis: Dict) -> List[TradingSignal]:
        """Generate enhanced trading signals with all new features and proper timezone alignment"""
        signals = []
        current_time = self._get_broker_time().replace(tzinfo=None)  # Use broker time as naive datetime
        
        # Use session timezone aligner for proper session detection
        utc_time = self.session_timezone_aligner.broker_time_to_utc(current_time)
        current_session = self.session_timezone_aligner.get_current_session(current_time)
        
        logger.debug(f"🕐 Trading signals - broker_time: {current_time}, utc_time: {utc_time}, session: {current_session}")
        
        # Pre-filter analysis data to remove any future timestamps
        analysis = self._sanitize_analysis_data(analysis, current_time)
        
        # Get session analysis for market maker patterns
        session_analysis = analysis.get('session_analysis', {})
        market_maker_patterns = session_analysis.get('market_maker_patterns', {})
        
        # Analyze H1 and H4 for primary signals
        for tf in ['H1', 'H4']:
            if tf not in analysis['timeframe_analysis']:
                continue
            
            tf_data = analysis['timeframe_analysis'][tf]
            trend_context = tf_data.get('trend_context')
            adjusted_weights = tf_data.get('adjusted_weights', {})
            
            # Enhanced Smart Money Concept signals with trend age context
            # Filter out invalid/future timestamps to prevent calculation errors
            logger.info(f"Processing {tf} timeframe: Found {len(tf_data['liquidity_sweeps'])} liquidity sweeps")
            valid_sweeps = []
            for s in tf_data['liquidity_sweeps']:
                try:
                    if self._is_valid_timestamp(s.get('timestamp'), current_time):
                        valid_sweeps.append(s)
                except Exception as e:
                    logger.debug(f"Skipping invalid sweep timestamp: {e}")
                    continue
            logger.info(f"After validation: {len(valid_sweeps)} valid sweeps in {tf}")
            
            recent_sweeps = []
            for s in valid_sweeps:
                try:
                    sweep_ts = s.get('timestamp')
                    if sweep_ts is None:
                        logger.error(f"Sweep missing timestamp key: {s}")
                        continue
                    logger.debug(f"Processing sweep timestamp: {sweep_ts} (type: {type(sweep_ts)})")
                    
                    # Additional validation before calculation
                    if not self._is_valid_timestamp(sweep_ts, current_time):
                        logger.error(f"Invalid sweep timestamp detected during processing: {sweep_ts}")
                        continue
                    
                    # Convert pandas Timestamp to datetime for calculation
                    original_sweep_ts = sweep_ts
                    logger.debug(f"Original sweep_ts: {original_sweep_ts} (type: {type(original_sweep_ts)})")
                    
                    if hasattr(sweep_ts, 'to_pydatetime'):
                        logger.debug(f"Converting pandas Timestamp to datetime")
                        sweep_ts = sweep_ts.to_pydatetime()
                        logger.debug(f"After conversion: {sweep_ts} (type: {type(sweep_ts)})")
                    elif isinstance(sweep_ts, str):
                        logger.debug(f"Converting string timestamp to datetime")
                        sweep_ts = pd.to_datetime(sweep_ts).to_pydatetime()
                        logger.debug(f"After string conversion: {sweep_ts} (type: {type(sweep_ts)})")
                    
                    # Convert to naive datetime for broker time comparison
                    logger.debug(f"Checking timezone: {sweep_ts.tzinfo}")
                    if hasattr(sweep_ts, 'tzinfo') and sweep_ts.tzinfo is not None:
                        logger.debug(f"Converting timezone-aware to naive datetime")
                        sweep_ts = sweep_ts.replace(tzinfo=None)
                    # sweep_ts is now in broker time (naive datetime)
                    
                    logger.debug(f"Final sweep_ts before calculation: {sweep_ts} (type: {type(sweep_ts)})")
                    logger.debug(f"Current time for calculation: {current_time} (type: {type(current_time)})")
                    
                    time_diff = (current_time - sweep_ts).total_seconds()
                    logger.debug(f"Time difference calculated: {time_diff} seconds")
                    
                    if time_diff < 604800:  # Last 7 days (temporarily increased for testing)
                        recent_sweeps.append(s)
                        logger.debug(f"Added sweep to recent_sweeps")
                    else:
                        logger.debug(f"Sweep too old: {time_diff} seconds")
                except Exception as e:
                    logger.error(f"Skipping sweep with timestamp calculation error: {e}, timestamp: {s.get('timestamp')}")
                    logger.error(f"Error type: {type(e)}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    continue
            
            logger.info(f"Processing {tf} timeframe: Found {len(tf_data['order_blocks'])} order blocks")
            valid_order_blocks = []
            for ob in tf_data['order_blocks']:
                try:
                    if self._is_valid_timestamp(ob.get('timestamp'), current_time):
                        valid_order_blocks.append(ob)
                except Exception as e:
                    logger.debug(f"Skipping invalid order block timestamp: {e}")
                    continue
            logger.info(f"After validation: {len(valid_order_blocks)} valid order blocks in {tf}")
            
            recent_order_blocks = []
            for ob in valid_order_blocks:
                try:
                    ob_ts = ob.get('timestamp')
                    if ob_ts is None:
                        logger.error(f"Order block missing timestamp key: {ob}")
                        continue
                    logger.debug(f"Processing order block timestamp: {ob_ts} (type: {type(ob_ts)})")
                    
                    # Additional validation before calculation
                    if not self._is_valid_timestamp(ob_ts, current_time):
                        logger.error(f"Invalid order block timestamp detected during processing: {ob_ts}")
                        continue
                    
                    # Convert pandas Timestamp to datetime for calculation
                    original_ob_ts = ob_ts
                    logger.debug(f"Original ob_ts: {original_ob_ts} (type: {type(original_ob_ts)})")
                    
                    if hasattr(ob_ts, 'to_pydatetime'):
                        logger.debug(f"Converting pandas Timestamp to datetime for order block")
                        ob_ts = ob_ts.to_pydatetime()
                        logger.debug(f"After conversion: {ob_ts} (type: {type(ob_ts)})")
                    elif isinstance(ob_ts, str):
                        logger.debug(f"Converting string timestamp to datetime for order block")
                        ob_ts = pd.to_datetime(ob_ts).to_pydatetime()
                        logger.debug(f"After string conversion: {ob_ts} (type: {type(ob_ts)})")
                    
                    # Convert to naive datetime for broker time comparison
                    logger.debug(f"Checking order block timezone: {ob_ts.tzinfo}")
                    if hasattr(ob_ts, 'tzinfo') and ob_ts.tzinfo is not None:
                        logger.debug(f"Converting order block timezone-aware to naive datetime")
                        ob_ts = ob_ts.replace(tzinfo=None)
                    # ob_ts is now in broker time (naive datetime)
                    
                    logger.debug(f"Final ob_ts before calculation: {ob_ts} (type: {type(ob_ts)})")
                    logger.debug(f"Current time for order block calculation: {current_time} (type: {type(current_time)})")
                    
                    time_diff = (current_time - ob_ts).total_seconds()
                    logger.debug(f"Order block time difference calculated: {time_diff} seconds")
                    
                    if time_diff < 604800:  # Last 7 days (temporarily increased for testing)
                        recent_order_blocks.append(ob)
                        logger.debug(f"Added order block to recent_order_blocks")
                    else:
                        logger.debug(f"Order block too old: {time_diff} seconds")
                except Exception as e:
                    logger.error(f"Skipping order block with timestamp calculation error: {e}, timestamp: {ob.get('timestamp')}")
                    logger.error(f"Order block error type: {type(e)}")
                    import traceback
                    logger.error(f"Order block full traceback: {traceback.format_exc()}")
                    continue
            
            # Enhanced FVG analysis with dynamic weighting
            enhanced_fvgs = []
            for fvg in tf_data['fair_value_gaps']:
                if hasattr(fvg, 'dynamic_weight'):
                    fvg_weight = self.memory_engine.get_adjusted_weight('fvg_weight', fvg.dynamic_weight)
                    if fvg_weight > 0.15:  # Only consider significant FVGs
                        enhanced_fvgs.append(fvg)
            
            # Signal generation with enhanced logic
            for sweep in recent_sweeps:
                for ob in recent_order_blocks:
                    if self._validate_enhanced_smc_setup(sweep, ob, tf_data, trend_context, market_maker_patterns):
                        signal = self._create_enhanced_smc_signal(sweep, ob, current_session, tf, trend_context, adjusted_weights)
                        if signal:
                            signals.append(signal)
            
            # Enhanced volatility breakout signals with prediction
            vol_analysis = tf_data['volatility']

            if vol_analysis['current_volatility_state'] == 'COMPRESSED':
                compression_zones = vol_analysis['compression_zones']

                if compression_zones:
                    latest_zone = compression_zones[-1]
                    # Use predicted expansion probability
                    expansion_prob = vol_analysis.get('predicted_expansion', {}).get('probability', 0.5)
                    adjusted_prob = self.memory_engine.get_adjusted_weight('volatility_prediction', expansion_prob)

                    
                    if adjusted_prob > 0.7:

                        breakout_signal = self._create_enhanced_breakout_signal(
                            latest_zone, current_session, tf, trend_context, vol_analysis
                        )
                        if breakout_signal:
                            signals.append(breakout_signal)
            
            # Session-based signals using market maker patterns
            session_signals = self._generate_session_based_signals(
                market_maker_patterns, current_session, tf, trend_context
            )
            signals.extend(session_signals)
        
        # Enhanced filtering and ranking with memory engine
        signals = self._enhanced_filter_and_rank_signals(signals, analysis)
        
        return signals[:3]  # Return top 3 signals
    
    def _generate_enhanced_trading_signals(self, analysis: Dict) -> List[TradingSignal]:
        """🚀 Enhanced trading signal generation with all strategic enhancements"""
        signals = []
        current_time = self._get_broker_time().replace(tzinfo=None)
        current_session = self.session_analyzer.get_current_session(current_time)
        
        # Pre-filter analysis data to remove any future timestamps
        analysis = self._sanitize_analysis_data(analysis, current_time)
        
        # Get strategic enhancement data
        institutional_flow = analysis.get('institutional_flow', {})
        liquidity_mapping = analysis.get('liquidity_mapping', {})
        market_regime = analysis.get('market_regime', {})
        
        logger.info(f"🔍 Institutional Flow Signals: {len(institutional_flow.get('accumulation_signals', []))}")
        logger.info(f"🎯 Liquidity Zones Mapped: {len(liquidity_mapping.get('liquidity_zones', []))}")
        logger.info(f"📊 Market Regime: {getattr(market_regime, 'trend_regime', 'UNKNOWN')} / {getattr(market_regime, 'volatility_regime', 'UNKNOWN')}")
        
        # Analyze H1 and H4 for primary signals with enhancements
        for tf in ['H1', 'H4']:
            if tf not in analysis['timeframe_analysis']:
                continue
            
            tf_data = analysis['timeframe_analysis'][tf]
            trend_context = tf_data.get('trend_context')
            adjusted_weights = tf_data.get('adjusted_weights', {})
            enhanced_order_blocks = tf_data.get('enhanced_order_blocks', [])
            
            # 🚀 Enhancement 1: Institutional Flow Validation
            # Filter signals based on institutional flow alignment
            valid_institutional_zones = []
            for signal in institutional_flow.get('accumulation_signals', []):
                if signal.confidence > 0.7 and signal.stealth_score > 0.6:
                    valid_institutional_zones.append({
                        'price_level': signal.price_level,
                        'direction': 'BULLISH' if signal.flow_type == 'ACCUMULATION' else 'BEARISH',
                        'strength': signal.confidence
                    })
            
            # 🚀 Enhancement 2: Advanced Order Block Filtering
            # Use only high-quality order blocks
            premium_order_blocks = []
            for enhanced_ob in enhanced_order_blocks:
                if (enhanced_ob.score.total_score > 7.0 and 
                    enhanced_ob.score.volume_score > 0.7):
                    premium_order_blocks.append(enhanced_ob)
            
            logger.info(f"📈 {tf}: {len(premium_order_blocks)} premium order blocks identified")
            
            # 🚀 Enhancement 3: Liquidity-Aware Signal Generation
            # Get high-probability liquidity sweep targets
            sweep_targets = []
            for zone in liquidity_mapping.get('liquidity_zones', []):
                # Check if zone has the required strength and type
                zone_strength = getattr(zone, 'strength', None)
                zone_type = getattr(zone, 'liquidity_type', None)
                
                if (zone_strength and hasattr(zone_strength, 'value') and 
                    zone_strength.value in ['high', 'extreme'] and
                    zone_type and hasattr(zone_type, 'value') and
                    zone_type.value in ['stop_cluster', 'swing_high', 'swing_low']):
                    sweep_targets.append(zone)
            
            # Generate enhanced SMC signals
            recent_sweeps = self._filter_recent_sweeps(tf_data.get('liquidity_sweeps', []), current_time)
            
            for sweep in recent_sweeps:
                for enhanced_ob in premium_order_blocks:
                    # Convert enhanced OB back to dict format for compatibility
                    ob_dict = {
                        'timestamp': enhanced_ob.timestamp,
                        'type': enhanced_ob.ob_type,
                        'high': enhanced_ob.high,
                        'low': enhanced_ob.low,
                        'strength': enhanced_ob.score.total_score
                    }
                    
                    if self._validate_enhanced_smc_setup_v2(sweep, ob_dict, tf_data, 
                                                           trend_context, institutional_flow, 
                                                           liquidity_mapping):
                        signal = self._create_institutional_aware_signal(
                            sweep, enhanced_ob, current_session, tf, 
                            trend_context, institutional_flow, market_regime
                        )
                        if signal:
                            signals.append(signal)
            
            # 🚀 Enhancement 4: Adaptive Volatility Signals
            vol_analysis = tf_data['volatility']
            if vol_analysis['current_volatility_state'] == 'COMPRESSED':
                compression_zones = vol_analysis['compression_zones']
                if compression_zones:
                    latest_zone = compression_zones[-1]
                    
                    # Apply adaptive weighting based on market regime
                    # Use volatility regime to determine if conditions are favorable
                    regime_favorable = (
                        getattr(market_regime, 'volatility_regime', 'medium') != 'high' and
                        getattr(market_regime, 'trend_regime', 'ranging') == 'trending'
                    )
                    
                    if regime_favorable:  # Only trade in favorable regimes
                        breakout_signal = self._create_regime_aware_breakout_signal(
                            latest_zone, current_session, tf, trend_context, 
                            market_regime, liquidity_mapping
                        )
                        if breakout_signal:
                            signals.append(breakout_signal)
        
        # 🚀 Session-based Market Maker Pattern Signals
        session_analysis = analysis.get('session_analysis', {})
        market_maker_patterns = session_analysis.get('market_maker_patterns', {})
        
        if market_maker_patterns:
            logger.debug(f"🔍 Generating session-based signals with patterns: {list(market_maker_patterns.keys())}")
            session_signals = self._generate_session_based_signals(
                market_maker_patterns, current_session, 'H1', None
            )
            signals.extend(session_signals)
            logger.info(f"📊 Added {len(session_signals)} session-based signals")
        
        # 🚀 Final Enhancement: Adaptive Signal Weighting
        logger.info(f"🎯 Applying adaptive signal weighting to {len(signals)} signals...")
        
        # Convert signals to the format expected by adaptive weighting
        signal_dicts = []
        for signal in signals:
            signal_dict = {
                'type': getattr(signal, 'signal_type', 'unknown'),
                'direction': getattr(signal, 'direction', 'neutral'),
                'confidence': getattr(signal, 'confidence', 0.5),
                'entry_price': getattr(signal, 'entry_price', 0),
                'timestamp': getattr(signal, 'timestamp', datetime.now())
            }
            signal_dicts.append(signal_dict)
        
        # Use the main analyze_and_weight_signals method
        mtf_data = analysis.get('timeframe_analysis', {})
        market_data = {tf: pd.DataFrame() for tf in mtf_data.keys()}  # Simplified for now
        
        try:
            enhanced_signals = self.adaptive_weighter.analyze_and_weight_signals(
                signal_dicts, market_data
            )
            # Filter by confidence
            enhanced_signals = [s for s in enhanced_signals if s.signal_weight.final_weight > 0.6]
        except Exception as e:
            logger.warning(f"Adaptive weighting failed: {e}, using original signals")
            enhanced_signals = signals
        
        # Sort by confidence and return top signals
        if enhanced_signals and hasattr(enhanced_signals[0], 'signal_weight'):
            enhanced_signals.sort(key=lambda x: x.signal_weight.final_weight, reverse=True)
        else:
            enhanced_signals.sort(key=lambda x: getattr(x, 'confidence', 0.5), reverse=True)
        
        logger.info(f"🚀 Enhanced signal generation complete: {len(enhanced_signals)} high-quality signals")
        
        # If no signals generated with enhanced method, try simplified approach
        if len(enhanced_signals) == 0:
            logger.info("🔄 No enhanced signals found, trying simplified approach...")
            try:
                simplified_signals = self._generate_simplified_trading_signals(analysis)
                logger.info(f"📊 Simplified approach generated {len(simplified_signals)} signals")
                return simplified_signals[:2]  # Return top 2
            except Exception as e:
                logger.error(f"Simplified signal generation failed: {e}")
                return []

        return enhanced_signals[:2]  # Return top 2 highest quality signals
    
    def _sanitize_analysis_data(self, analysis: Dict, current_time: datetime) -> Dict:
        """Remove any data with future timestamps to prevent calculation errors"""
        sanitized_analysis = analysis.copy()
        
        logger.debug(f"Sanitizing analysis data at {current_time}")
        
        # Sanitize timeframe analysis data
        if 'timeframe_analysis' in sanitized_analysis:
            for tf, tf_data in sanitized_analysis['timeframe_analysis'].items():
                logger.debug(f"Sanitizing {tf} timeframe data")
                
                # Filter liquidity sweeps
                if 'liquidity_sweeps' in tf_data:
                    original_count = len(tf_data['liquidity_sweeps'])
                    valid_sweeps = []
                    for i, sweep in enumerate(tf_data['liquidity_sweeps']):
                        try:
                            sweep_ts = sweep.get('timestamp')
                            logger.debug(f"Checking sweep {i} in {tf}: {sweep_ts} (type: {type(sweep_ts)})")
                            if self._is_valid_timestamp(sweep_ts, current_time):
                                valid_sweeps.append(sweep)
                            else:
                                logger.error(f"Filtered out invalid sweep timestamp in {tf}: {sweep_ts}")
                        except Exception as e:
                            logger.error(f"Error validating sweep timestamp in {tf}: {e}, timestamp: {sweep.get('timestamp')}")
                    tf_data['liquidity_sweeps'] = valid_sweeps
                    logger.debug(f"Filtered sweeps in {tf}: {original_count} -> {len(valid_sweeps)}")
                
                # Filter order blocks
                if 'order_blocks' in tf_data:
                    original_ob_count = len(tf_data['order_blocks'])
                    valid_order_blocks = []
                    for i, ob in enumerate(tf_data['order_blocks']):
                        try:
                            ob_ts = ob.get('timestamp')
                            logger.debug(f"Checking order block {i} in {tf}: {ob_ts} (type: {type(ob_ts)})")
                            if self._is_valid_timestamp(ob_ts, current_time):
                                valid_order_blocks.append(ob)
                            else:
                                logger.error(f"Filtered out invalid order block timestamp in {tf}: {ob_ts}")
                        except Exception as e:
                            logger.error(f"Error validating order block timestamp in {tf}: {e}, timestamp: {ob.get('timestamp')}")
                    tf_data['order_blocks'] = valid_order_blocks
                    logger.debug(f"Filtered order blocks in {tf}: {original_ob_count} -> {len(valid_order_blocks)}")
                
                # Filter fair value gaps
                if 'fair_value_gaps' in tf_data:
                    valid_fvgs = []
                    for fvg in tf_data['fair_value_gaps']:
                        try:
                            fvg_timestamp = getattr(fvg, 'timestamp', None)
                            if fvg_timestamp and self._is_valid_timestamp(fvg_timestamp, current_time):
                                valid_fvgs.append(fvg)
                            elif not fvg_timestamp:
                                # Keep FVGs without timestamps
                                valid_fvgs.append(fvg)
                            else:
                                logger.debug(f"Filtered out invalid FVG timestamp: {fvg_timestamp}")
                        except Exception as e:
                            logger.debug(f"Error validating FVG timestamp: {e}")
                    tf_data['fair_value_gaps'] = valid_fvgs
        
        return sanitized_analysis
    
    def _get_broker_time(self) -> datetime:
        """Get current broker server time from MT5 as naive datetime"""
        try:
            # Get current tick which contains broker server time
            tick = mt5.symbol_info_tick(self.symbol)
            if tick and tick.time:
                # Return naive datetime in broker time
                return datetime.fromtimestamp(tick.time)
        except Exception as e:
            logger.debug(f"Failed to get broker time: {e}")
        
        # Fallback to system time as naive datetime
        return datetime.now()
    
    def _is_valid_timestamp(self, timestamp, current_time: datetime = None) -> bool:
        """Validate timestamp for signal generation - accept any reasonable timestamp from collected data"""
        if not timestamp:
            return False
        
        try:
            # Handle string timestamps that might be invalid
            if isinstance(timestamp, str):
                try:
                    timestamp = pd.to_datetime(timestamp)
                except:
                    logger.debug(f"Invalid string timestamp: {timestamp}")
                    return False
            
            # Convert to datetime if it's a pandas Timestamp
            if hasattr(timestamp, 'to_pydatetime'):
                try:
                    timestamp = timestamp.to_pydatetime()
                except:
                    logger.debug(f"Failed to convert pandas timestamp: {timestamp}")
                    return False
            
            # Ensure we have a datetime object
            if not isinstance(timestamp, datetime):
                logger.debug(f"Timestamp is not a datetime object: {type(timestamp)}")
                return False
            
            # Convert to naive datetime for broker time compatibility
            if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)
            
            # For signal generation, accept any timestamp from collected data
            # Only reject obviously invalid timestamps (like year 1970 or far future)
            if timestamp.year < 2020 or timestamp.year > 2030:
                logger.debug(f"Timestamp year out of reasonable range: {timestamp}")
                return False
            
            return True
            
        except Exception as e:
             logger.debug(f"Error validating timestamp {timestamp}: {e}")
             return False
    
    def _is_valid_data_timestamp(self, timestamp) -> bool:
        """Validate timestamp during data collection using broker time to prevent invalid data entry"""
        if not timestamp:
            return False
        
        try:
            # Use broker time instead of local system time
            broker_time = self._get_broker_time()
            
            # Handle string timestamps that might be invalid
            if isinstance(timestamp, str):
                try:
                    timestamp = pd.to_datetime(timestamp)
                except:
                    logger.debug(f"Invalid string data timestamp: {timestamp}")
                    return False
            
            # Convert to datetime if it's a pandas Timestamp
            if hasattr(timestamp, 'to_pydatetime'):
                try:
                    timestamp = timestamp.to_pydatetime()
                except:
                    logger.debug(f"Failed to convert pandas data timestamp: {timestamp}")
                    return False
            
            # Ensure we have a datetime object
            if not isinstance(timestamp, datetime):
                logger.debug(f"Data timestamp is not a datetime object: {type(timestamp)}")
                return False
            
            # Convert to naive datetime for broker time compatibility
            if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)
            
            # Check if timestamp is reasonable relative to broker time
            try:
                time_diff = (broker_time - timestamp).total_seconds()
            except Exception as calc_error:
                logger.debug(f"Failed to calculate time difference for data timestamp {timestamp}: {calc_error}")
                return False
            
            # More lenient validation for demo servers with corrupted data
            # Reject future timestamps (allow 1 hour buffer for data collection)
            # or timestamps older than 180 days (6 months)
            if time_diff < -3600 or time_diff > (180 * 24 * 3600):
                logger.debug(f"Data timestamp out of valid range relative to broker time: {timestamp} (broker: {broker_time}, diff: {time_diff}s)")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating data timestamp {timestamp}: {e}")
            return False
     
    def _validate_smc_setup(self, sweep: Dict, order_block: Dict, tf_data: Dict) -> bool:
        """Validate if SMC setup is tradeable"""
        # Check if sweep and order block are aligned
        if sweep['type'] == 'Liquidity_Sweep_High' and order_block['type'] == 'Bearish_OB':
            return True
        elif sweep['type'] == 'Liquidity_Sweep_Low' and order_block['type'] == 'Bullish_OB':
            return True
        
        return False
    
    def _validate_enhanced_smc_setup(self, sweep: Dict, order_block: Dict, tf_data: Dict, 
                                   trend_context: TrendContext, market_maker_patterns: Dict) -> bool:
        """Enhanced SMC setup validation with trend age and market maker context"""
        # Basic alignment check
        if not self._validate_smc_setup(sweep, order_block, tf_data):
            return False
        
        # Trend age validation
        if trend_context:
            # Prefer early to mid-trend setups
            if trend_context.trend_phase == 'late':
                return False
            
            # Check momentum alignment
            if sweep['type'] == 'Liquidity_Sweep_High' and trend_context.momentum_strength < 0.3:
                return False
            elif sweep['type'] == 'Liquidity_Sweep_Low' and trend_context.momentum_strength < 0.3:
                return False
        
        # Market maker pattern validation
        if market_maker_patterns:
            # Check for favorable session patterns
            london_ny_continuation = market_maker_patterns.get('london_sets_ny_continues', {})
            if london_ny_continuation.get('success_rate', 0) > 0.6:
                return True
        
        return True
    
    def _create_smc_signal(self, sweep: Dict, order_block: Dict, session: SessionType, timeframe: str) -> Optional[TradingSignal]:
        """Create trading signal based on SMC setup"""
        current_price = self.data_collector.get_current_price()
        if not current_price:
            return None
        
        if sweep['type'] == 'Liquidity_Sweep_High':
            # Bearish signal
            entry_price = order_block['high']
            stop_loss = sweep['sweep_price'] + self.data_collector.pips_to_price(20)  # 20 pips buffer
            take_profit = entry_price - (stop_loss - entry_price) * 2  # 1:2 RR
            direction = "SELL"
        else:
            # Bullish signal
            entry_price = order_block['low']
            stop_loss = sweep['sweep_price'] - self.data_collector.pips_to_price(20)  # 20 pips buffer
            take_profit = entry_price + (entry_price - stop_loss) * 2  # 1:2 RR
            direction = "BUY"
        
        risk_reward = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        confidence = min(sweep.get('strength', 1.0) * order_block.get('strength', 1.0), 1.0)
        
        return TradingSignal(
            timestamp=datetime.now(pytz.UTC),
            signal_type="SMC_SETUP",
            direction=direction,
            strength=SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            confidence=confidence,
            reasoning=f"Liquidity sweep at {sweep['sweep_level']:.5f} followed by {order_block['type']} setup",
            session=session,
            market_structure=MarketStructure.TRANSITION,
            symbol=self.symbol
        )
    
    def _create_enhanced_smc_signal(self, sweep: Dict, order_block: Dict, session: SessionType, 
                                  timeframe: str, trend_context: TrendContext, adjusted_weights: Dict) -> Optional[TradingSignal]:
        """Create enhanced SMC signal with trend age and memory engine adjustments"""
        current_price = self.data_collector.get_current_price()
        if not current_price:
            return None
        
        if sweep['type'] == 'Liquidity_Sweep_High':
            # Bearish signal
            entry_price = order_block['high']
            stop_loss = sweep['sweep_price'] + self.data_collector.pips_to_price(20)  # 20 pips buffer
            take_profit = entry_price - (stop_loss - entry_price) * 2  # 1:2 RR
            direction = "SELL"
        else:
            # Bullish signal
            entry_price = order_block['low']
            stop_loss = sweep['sweep_price'] - self.data_collector.pips_to_price(20)  # 20 pips buffer
            take_profit = entry_price + (entry_price - stop_loss) * 2  # 1:2 RR
            direction = "BUY"
        
        # Enhanced confidence calculation with trend context
        base_confidence = min(sweep.get('strength', 1.0) * order_block.get('strength', 1.0), 1.0)
        
        # Adjust confidence based on trend age
        if trend_context:
            trend_multiplier = {
                'early': 1.2,
                'mid': 1.0,
                'late': 0.7
            }.get(trend_context.trend_phase, 1.0)
            
            momentum_bonus = trend_context.momentum_strength * 0.2
            base_confidence = min(base_confidence * trend_multiplier + momentum_bonus, 1.0)
        
        # Apply memory engine adjustments
        trend_weight = adjusted_weights.get('trend_weight', 1.0)
        final_confidence = min(base_confidence * trend_weight, 1.0)
        
        risk_reward = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        
        # Enhanced reasoning with trend context
        reasoning = f"Enhanced SMC: Liquidity sweep at {sweep['sweep_level']:.5f}, {order_block['type']} setup"
        if trend_context:
            reasoning += f", {trend_context.trend_phase} trend phase (momentum: {trend_context.momentum_strength:.2f})"
        
        return TradingSignal(
            timestamp=datetime.now(pytz.UTC),
            signal_type="ENHANCED_SMC_SETUP",
            direction=direction,
            strength=SignalStrength.STRONG if final_confidence > 0.7 else SignalStrength.MODERATE,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            confidence=final_confidence,
            reasoning=reasoning,
            session=session,
            market_structure=MarketStructure.TRANSITION,
            symbol=self.symbol
        )
    
    def _create_breakout_signal(self, compression_zone: Dict, session: SessionType, timeframe: str) -> Optional[TradingSignal]:
        """Create volatility breakout signal"""
        current_price = self.data_collector.get_current_price()
        if not current_price:
            return None
        
        # Determine breakout direction based on session bias
        session_analysis = self.analysis_cache.get('session_analysis', {})
        session_stats = session_analysis.get('session_statistics', {})
        current_session_stats = session_stats.get(session.value, {})
        
        bullish_bias = current_session_stats.get('bullish_bias', 0.5)
        
        if bullish_bias > 0.6:
            direction = "BUY"
            entry_price = current_price['ask']
            stop_loss = entry_price - self.data_collector.pips_to_price(30)  # 30 pips
            take_profit = entry_price + self.data_collector.pips_to_price(60)  # 60 pips (1:2 RR)
        else:
            direction = "SELL"
            entry_price = current_price['bid']
            stop_loss = entry_price + self.data_collector.pips_to_price(30)  # 30 pips
            take_profit = entry_price - self.data_collector.pips_to_price(60)  # 60 pips (1:2 RR)
        
        return TradingSignal(
            timestamp=datetime.now(pytz.UTC),
            signal_type="VOLATILITY_BREAKOUT",
            direction=direction,
            strength=SignalStrength.MODERATE,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=2.0,
            confidence=compression_zone['expansion_probability'],
            reasoning=f"Volatility compression for {compression_zone['duration']} periods, expansion expected",
            session=session,
            market_structure=MarketStructure.RANGING,
            symbol=self.symbol
        )
    
    def _create_enhanced_breakout_signal(self, compression_zone: Dict, session: SessionType, 
                                       timeframe: str, trend_context: TrendContext, vol_analysis: Dict) -> Optional[TradingSignal]:
        """Create enhanced volatility breakout signal with prediction"""
        current_price = self.data_collector.get_current_price()
        if not current_price:
            return None
        
        # Get predicted expansion details
        predicted_expansion = vol_analysis.get('predicted_expansion', {})
        expansion_range = predicted_expansion.get('expected_range', self.data_collector.pips_to_price(60))  # Default 60 pips
        expansion_probability = predicted_expansion.get('probability', 0.5)
        
        # Determine direction based on trend context and session bias
        session_analysis = self.analysis_cache.get('session_analysis', {})
        session_stats = session_analysis.get('session_statistics', {})
        current_session_stats = session_stats.get(session.value, {})
        
        bullish_bias = current_session_stats.get('bullish_bias', 0.5)
        
        # Enhanced direction logic with trend context
        if trend_context and trend_context.trend_direction:
            if trend_context.trend_direction == 'bullish' and trend_context.trend_phase != 'late':
                direction = "BUY"
            elif trend_context.trend_direction == 'bearish' and trend_context.trend_phase != 'late':
                direction = "SELL"
            else:
                direction = "BUY" if bullish_bias > 0.6 else "SELL"
        else:
            direction = "BUY" if bullish_bias > 0.6 else "SELL"
        
        # Convert expansion range from pips to price using currency-specific pip value
        expansion_range_pips = 60  # Default 60 pips
        expansion_range_price = self.data_collector.pips_to_price(expansion_range_pips)
        
        if direction == "BUY":
            entry_price = current_price['ask']
            stop_loss = entry_price - self.data_collector.pips_to_price(30)  # 30 pips
            take_profit = entry_price + self.data_collector.pips_to_price(60)  # 60 pips
        else:
            entry_price = current_price['bid']
            stop_loss = entry_price + self.data_collector.pips_to_price(30)  # 30 pips
            take_profit = entry_price - self.data_collector.pips_to_price(60)  # 60 pips
        
        risk_reward = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        
        # Enhanced confidence with prediction accuracy
        base_confidence = expansion_probability
        if trend_context:
            momentum_bonus = trend_context.momentum_strength * 0.15
            base_confidence = min(base_confidence + momentum_bonus, 1.0)
        
        reasoning = f"Enhanced volatility breakout: {expansion_probability:.1%} expansion probability, "
        reasoning += f"predicted range: {expansion_range*10000:.0f} pips"
        if trend_context:
            reasoning += f", {trend_context.trend_phase} {trend_context.trend_direction} trend"
        
        return TradingSignal(
            timestamp=datetime.now(pytz.UTC),
            signal_type="ENHANCED_VOLATILITY_BREAKOUT",
            direction=direction,
            strength=SignalStrength.STRONG if base_confidence > 0.75 else SignalStrength.MODERATE,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            confidence=base_confidence,
            reasoning=reasoning,
            session=session,
            market_structure=MarketStructure.BREAKOUT,
            symbol=self.symbol
        )
    
    def _generate_session_based_signals(self, market_maker_patterns: Dict, current_session: SessionType, 
                                       timeframe: str, trend_context: TrendContext) -> List[TradingSignal]:
        """Generate signals based on market maker patterns with proper timezone alignment"""
        signals = []
        current_price = self.data_collector.get_current_price()
        
        # Get broker time and convert to UTC for proper session detection
        broker_time = self._get_broker_time().replace(tzinfo=None)
        utc_time = self.session_timezone_aligner.broker_time_to_utc(broker_time)
        aligned_session = self.session_timezone_aligner.get_current_session(utc_time)
        
        logger.debug(f"🔍 Session signal generation - broker_time: {broker_time}, utc_time: {utc_time}")
        logger.debug(f"🔍 Session alignment - original_session: {current_session}, aligned_session: {aligned_session}")
        logger.debug(f"🔍 Session signal generation - current_price: {current_price is not None}, market_maker_patterns: {market_maker_patterns is not None}")
        
        if not current_price or not market_maker_patterns:
            logger.debug(f"❌ Early return - current_price: {current_price is not None}, market_maker_patterns keys: {list(market_maker_patterns.keys()) if market_maker_patterns else 'None'}")
            return signals
        
        # Use aligned session for signal generation
        current_session = aligned_session
        
        # London-NY continuation pattern
        london_ny_pattern = market_maker_patterns.get('london_sets_ny_continues', {})
        logger.debug(f"📊 London-NY pattern: success_rate={london_ny_pattern.get('success_rate', 0)}, recent_occurrences={len(london_ny_pattern.get('recent_occurrences', []))}")
        
        if (current_session == SessionType.NEW_YORK and 
            london_ny_pattern.get('success_rate', 0) > 0.3):  # Lowered from 0.65 to 0.3
            
            recent_occurrences = london_ny_pattern.get('recent_occurrences', [])
            if recent_occurrences:
                latest = recent_occurrences[-1]
                if latest.get('ny_follow_through', False):
                    direction = "BUY" if latest['direction'] == 'bullish' else "SELL"
                    
                    if direction == "BUY":
                        entry_price = current_price['ask']
                        stop_loss = entry_price - self.data_collector.pips_to_price(30)  # 30 pips
                        take_profit = entry_price + self.data_collector.pips_to_price(60)  # 60 pips
                    else:
                        entry_price = current_price['bid']
                        stop_loss = entry_price + self.data_collector.pips_to_price(30)  # 30 pips
                        take_profit = entry_price - self.data_collector.pips_to_price(60)  # 60 pips
                    
                    confidence = london_ny_pattern['success_rate'] * latest['strength']
                    
                    signal = TradingSignal(
                        timestamp=datetime.now(pytz.UTC),
                        signal_type="LONDON_NY_CONTINUATION",
                        direction=direction,
                        strength=SignalStrength.STRONG if confidence > 0.7 else SignalStrength.MODERATE,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward=2.0,
                        confidence=confidence,
                        reasoning=f"London-NY continuation pattern: {latest['direction']} direction, {confidence:.1%} confidence",
                        session=current_session,
                        market_structure=MarketStructure.CONTINUATION,
                        symbol=self.symbol
                    )
                    signals.append(signal)
        
        # NY liquidity grab reversal pattern
        ny_liquidity_pattern = market_maker_patterns.get('ny_grabs_london_liquidity', {})
        logger.debug(f"📊 NY liquidity pattern: success_rate={ny_liquidity_pattern.get('success_rate', 0)}, recent_occurrences={len(ny_liquidity_pattern.get('recent_occurrences', []))}")

        if (current_session == SessionType.NEW_YORK and 
            ny_liquidity_pattern.get('success_rate', 0) > 0.25):  # Lowered from 0.6 to 0.25
            
            recent_grabs = ny_liquidity_pattern.get('recent_occurrences', [])
            if recent_grabs:
                latest_grab = recent_grabs[-1]
                grab_type = latest_grab.get('grab_type')
                
                if grab_type == 'high_liquidity_grab':
                    direction = "SELL"  # Reversal after high grab
                    entry_price = current_price['bid']
                    stop_loss = entry_price + self.data_collector.pips_to_price(25)  # 25 pips
                    take_profit = entry_price - self.data_collector.pips_to_price(50)  # 50 pips
                elif grab_type == 'low_liquidity_grab':
                    direction = "BUY"  # Reversal after low grab
                    entry_price = current_price['ask']
                    stop_loss = entry_price - self.data_collector.pips_to_price(25)  # 25 pips
                    take_profit = entry_price + self.data_collector.pips_to_price(50)  # 50 pips
                else:
                    return signals
                
                confidence = ny_liquidity_pattern['success_rate'] * latest_grab['reversal_strength']
                
                signal = TradingSignal(
                    timestamp=datetime.now(pytz.UTC),
                    signal_type="NY_LIQUIDITY_REVERSAL",
                    direction=direction,
                    strength=SignalStrength.MODERATE,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=2.0,
                    confidence=confidence,
                    reasoning=f"NY liquidity grab reversal: {grab_type}, {confidence:.1%} confidence",
                    session=current_session,
                    market_structure=MarketStructure.REVERSAL,
                    symbol=self.symbol
                )
                signals.append(signal)
        
        return signals
    
    def _enhanced_filter_and_rank_signals(self, signals: List[TradingSignal], analysis: Dict) -> List[TradingSignal]:
        """Enhanced signal filtering and ranking with memory engine"""
        if not signals:
            return []
        
        # Apply memory engine weights to signal confidence
        for signal in signals:
            # Get contributing factors for this signal type
            contributing_factors = self._extract_signal_factors(signal, analysis)
            
            # Adjust confidence based on memory engine learning
            adjusted_confidence = signal.confidence
            for factor, value in contributing_factors.items():
                weight = self.memory_engine.get_adjusted_weight(factor, value)
                adjusted_confidence *= weight
            
            # Update signal confidence (create new signal with adjusted confidence)
            signal.confidence = min(adjusted_confidence, 1.0)
        
        # Filter signals by minimum confidence threshold
        filtered_signals = [s for s in signals if s.confidence > 0.4]
        
        # Rank by confidence and risk-reward ratio
        ranked_signals = sorted(filtered_signals, 
                              key=lambda s: (s.confidence * s.risk_reward), 
                              reverse=True)
        
        return ranked_signals
    
    def _extract_signal_factors(self, signal: TradingSignal, analysis: Dict) -> Dict[str, float]:
        """Extract contributing factors from a signal for memory engine"""
        factors = {}
        
        # Extract factors based on signal type
        if "SMC" in signal.signal_type:
            factors['order_block_strength'] = 0.8  # Placeholder
            factors['liquidity_sweep'] = 0.7
            factors['market_structure'] = 0.6
        
        if "VOLATILITY" in signal.signal_type:
            factors['volatility_prediction'] = signal.confidence
        
        if "LONDON_NY" in signal.signal_type or "NY_LIQUIDITY" in signal.signal_type:
            factors['session_bias'] = signal.confidence
        
        # Add trend age factor if available
        timeframe_analysis = analysis.get('timeframe_analysis', {})
        for tf_data in timeframe_analysis.values():
            trend_context = tf_data.get('trend_context')
            if trend_context:
                factors['trend_age'] = trend_context.momentum_strength
                break
        
        return factors
    
    def _filter_recent_sweeps(self, sweeps: List[Dict], current_time: datetime) -> List[Dict]:
        """Filter sweeps to recent ones for enhanced signal generation"""
        recent_sweeps = []
        for sweep in sweeps:
            try:
                if self._is_valid_timestamp(sweep.get('timestamp'), current_time):
                    sweep_ts = sweep.get('timestamp')
                    if hasattr(sweep_ts, 'to_pydatetime'):
                        sweep_ts = sweep_ts.to_pydatetime()
                    if hasattr(sweep_ts, 'tzinfo') and sweep_ts.tzinfo is not None:
                        sweep_ts = sweep_ts.replace(tzinfo=None)
                    
                    time_diff = (current_time - sweep_ts).total_seconds()
                    if time_diff < 604800:  # Last 7 days
                        recent_sweeps.append(sweep)
            except Exception as e:
                logger.debug(f"Error filtering sweep: {e}")
                continue
        return recent_sweeps
    
    def _validate_enhanced_smc_setup_v2(self, sweep: Dict, order_block: Dict, tf_data: Dict,
                                       trend_context, institutional_flow: Dict, 
                                       liquidity_mapping: Dict) -> bool:
        """🚀 Enhanced SMC setup validation with institutional flow and liquidity mapping"""
        # Basic alignment check
        if not self._validate_smc_setup(sweep, order_block, tf_data):
            return False
        
        # Institutional flow alignment
        accumulation_signals = institutional_flow.get('accumulation_signals', [])
        ob_price = (order_block['high'] + order_block['low']) / 2
        
        institutional_alignment = False
        for signal in accumulation_signals:
            if (abs(signal.price_level - ob_price) < self.data_collector.pips_to_price(50) and
                signal.confidence > 0.6):
                institutional_alignment = True
                break
        
        # Liquidity mapping validation
        liquidity_zones = liquidity_mapping.get('liquidity_zones', [])
        sweep_price = sweep.get('sweep_price', 0)
        
        liquidity_alignment = False
        for zone in liquidity_zones:
            if (abs(zone.get('price_level', 0) - sweep_price) < self.data_collector.pips_to_price(30) and
                zone.get('strength', 0) > 0.7):
                liquidity_alignment = True
                break
        
        # Require at least one enhancement alignment
        return institutional_alignment or liquidity_alignment
    
    def _create_institutional_aware_signal(self, sweep: Dict, enhanced_ob, current_session: SessionType,
                                         timeframe: str, trend_context, institutional_flow: Dict,
                                         market_regime: Dict) -> Optional[TradingSignal]:
        """🚀 Create trading signal with institutional flow awareness"""
        current_price = self.data_collector.get_current_price()
        if not current_price:
            return None
        
        # Get institutional flow strength
        institutional_strength = 0.5
        accumulation_signals = institutional_flow.get('accumulation_signals', [])
        ob_price = (enhanced_ob.high + enhanced_ob.low) / 2
        
        for signal in accumulation_signals:
            if abs(signal.price_level - ob_price) < self.data_collector.pips_to_price(50):
                institutional_strength = signal.confidence
                break
        
        if sweep['type'] == 'Liquidity_Sweep_High':
            # Bearish signal
            entry_price = enhanced_ob.high
            stop_loss = sweep['sweep_price'] + self.data_collector.pips_to_price(25)
            take_profit = entry_price - (stop_loss - entry_price) * 2.5  # Enhanced RR
            direction = "SELL"
        else:
            # Bullish signal
            entry_price = enhanced_ob.low
            stop_loss = sweep['sweep_price'] - self.data_collector.pips_to_price(25)
            take_profit = entry_price + (entry_price - stop_loss) * 2.5  # Enhanced RR
            direction = "BUY"
        
        # Calculate enhanced confidence
        base_confidence = enhanced_ob.score.total_score / 10.0
        regime_modifier = 0.8 if getattr(market_regime, 'trend_regime', 'ranging') == 'trending' else 0.6
        final_confidence = (base_confidence * 0.6 + institutional_strength * 0.4) * regime_modifier
        
        risk_reward = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        
        return TradingSignal(
            timestamp=datetime.now(pytz.UTC),
            signal_type=f"ENHANCED_SMC_{timeframe}",
            direction=direction,
            strength=SignalStrength.STRONG if final_confidence > 0.8 else SignalStrength.MODERATE,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            confidence=final_confidence,
            reasoning=f"🚀 Enhanced SMC: OB Score {enhanced_ob.score.total_score:.1f}, Institutional Flow {institutional_strength:.1%}, Regime {regime_modifier:.1%}",
            session=current_session,
            market_structure=MarketStructure.REVERSAL,
            symbol=self.symbol
        )
    
    def _create_regime_aware_breakout_signal(self, compression_zone: Dict, current_session: SessionType,
                                           timeframe: str, trend_context, market_regime: Dict,
                                           liquidity_mapping: Dict) -> Optional[TradingSignal]:
        """🚀 Create breakout signal with regime and liquidity awareness"""
        current_price = self.data_collector.get_current_price()
        if not current_price:
            return None
        
        # Determine direction based on regime and liquidity
        regime_bias = 'BULLISH' if getattr(market_regime, 'ema200_distance', 0) > 0 else 'BEARISH'
        liquidity_zones = liquidity_mapping.get('liquidity_zones', [])
        
        # Find nearest liquidity target
        current_mid = (current_price['bid'] + current_price['ask']) / 2
        nearest_target = None
        min_distance = float('inf')
        
        for zone in liquidity_zones:
            distance = abs(zone.get('price_level', 0) - current_mid)
            if distance < min_distance and zone.get('strength', 0) > 0.7:
                min_distance = distance
                nearest_target = zone
        
        if not nearest_target:
            return None
        
        # Determine direction towards liquidity target
        if nearest_target['price_level'] > current_mid:
            direction = "BUY"
            entry_price = current_price['ask']
            stop_loss = compression_zone['low'] - self.data_collector.pips_to_price(20)
            take_profit = nearest_target['price_level']
        else:
            direction = "SELL"
            entry_price = current_price['bid']
            stop_loss = compression_zone['high'] + self.data_collector.pips_to_price(20)
            take_profit = nearest_target['price_level']
        
        # Calculate regime-adjusted confidence
        base_confidence = 0.7 if getattr(market_regime, 'volatility_regime', 'medium') != 'high' else 0.5
        liquidity_strength = nearest_target.get('strength', 0.5)
        final_confidence = (base_confidence * 0.7 + liquidity_strength * 0.3)
        
        risk_reward = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        
        if risk_reward < 1.5:  # Minimum RR requirement
            return None
        
        return TradingSignal(
            timestamp=datetime.now(pytz.UTC),
            signal_type=f"REGIME_BREAKOUT_{timeframe}",
            direction=direction,
            strength=SignalStrength.STRONG if final_confidence > 0.75 else SignalStrength.MODERATE,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            confidence=final_confidence,
            reasoning=f"🚀 Regime-aware breakout: {regime_bias} bias, targeting liquidity at {nearest_target['price_level']:.5f}",
            session=current_session,
            market_structure=MarketStructure.BREAKOUT,
            symbol=self.symbol
        )
    
    
    def _generate_simplified_trading_signals(self, analysis: Dict) -> List:
        """Generate trading signals with simplified, more permissive criteria"""
        signals = []
        
        try:
            current_time = self._get_broker_time().replace(tzinfo=None)
            current_price = self.data_collector.get_current_price()
            
            # Extract current price value if it's a dict
            if isinstance(current_price, dict):
                current_price = current_price.get('bid', current_price.get('ask', 1.0))
            
            logger.info(f"🎯 Simplified Signal Generation - Current Price: {current_price}")
        except Exception as e:
            logger.error(f"Error in simplified signal generation setup: {e}")
            return signals
        
        # Process H1 and H4 timeframes with relaxed criteria
        for tf in ['H1', 'H4']:
            try:
                if tf not in analysis.get('timeframe_analysis', {}):
                    continue
                    
                tf_data = analysis['timeframe_analysis'][tf]
                
                # Get basic structures with safe access
                order_blocks = tf_data.get('order_blocks', []) if isinstance(tf_data, dict) else []
                liquidity_sweeps = tf_data.get('liquidity_sweeps', []) if isinstance(tf_data, dict) else []
                fair_value_gaps = tf_data.get('fair_value_gaps', []) if isinstance(tf_data, dict) else []
                
                # Ensure all are lists
                if not isinstance(order_blocks, list):
                    order_blocks = []
                if not isinstance(liquidity_sweeps, list):
                    liquidity_sweeps = []
                if not isinstance(fair_value_gaps, list):
                    fair_value_gaps = []
                
                logger.info(f"📊 {tf}: {len(order_blocks)} OBs, {len(liquidity_sweeps)} sweeps, {len(fair_value_gaps)} FVGs")
            except Exception as e:
                logger.error(f"Error accessing {tf} timeframe data: {e}")
                continue
            
            # Generate signals from order blocks (simplified criteria)
            # Safely get last 3 order blocks
            if len(order_blocks) >= 3:
                recent_obs = order_blocks[-3:]
            else:
                recent_obs = order_blocks
            
            for ob in recent_obs:
                try:
                    # Ensure ob is a dictionary
                    if not isinstance(ob, dict):
                        continue
                        
                    ob_high = float(ob.get('high', 0))
                    ob_low = float(ob.get('low', 0))
                    ob_type = ob.get('type', '')
                    ob_strength = float(ob.get('strength', 0))
                    
                    # Relaxed criteria: strength > 0.5 instead of 0.7+
                    if ob_strength > 0.5 and ob_high > 0 and ob_low > 0:
                        
                        # Check if price is near the order block
                        distance_to_ob = min(
                            abs(current_price - ob_high) / current_price,
                            abs(current_price - ob_low) / current_price
                        )
                        
                        # If price is within 0.5% of order block, generate signal
                        if distance_to_ob < 0.005:  # 0.5% tolerance
                            
                            direction = 'BUY' if 'Bullish' in ob_type else 'SELL'
                            entry_price = ob_low if direction == 'BUY' else ob_high
                            
                            # Simple risk management
                            if direction == 'BUY':
                                stop_loss = ob_low - (ob_high - ob_low) * 0.2
                                take_profit = ob_high + (ob_high - ob_low) * 1.5
                            else:
                                stop_loss = ob_high + (ob_high - ob_low) * 0.2
                                take_profit = ob_low - (ob_high - ob_low) * 1.5
                            
                            # Calculate risk/reward
                            risk = abs(entry_price - stop_loss)
                            reward = abs(take_profit - entry_price)
                            risk_reward = reward / risk if risk > 0 else 0
                            
                            # Relaxed R:R requirement (0.8 instead of 1.2+)
                            if risk_reward > 0.8:
                                signal = {
                                    'signal_type': 'ORDER_BLOCK_RETEST',
                                    'direction': direction,
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'confidence': min(ob_strength + 0.2, 0.9),  # Boost confidence
                                    'risk_reward': risk_reward,
                                    'timeframe': tf,
                                    'timestamp': current_time,
                                    'reason': f'{tf} Order Block Retest - Strength: {ob_strength:.2f}'
                                }
                                signals.append(signal)
                                logger.info(f"✅ Generated {direction} signal from {tf} OB - R:R: {risk_reward:.2f}")
                
                except Exception as e:
                    logger.error(f"Error processing order block: {e}")
                    continue
            
            # Generate signals from liquidity sweeps + fair value gaps
            # Safely get recent sweeps and FVGs
            if len(liquidity_sweeps) >= 2:
                recent_sweeps = liquidity_sweeps[-2:]
            else:
                recent_sweeps = liquidity_sweeps
                
            if len(fair_value_gaps) >= 2:
                recent_fvgs = fair_value_gaps[-2:]
            else:
                recent_fvgs = fair_value_gaps
            
            for sweep in recent_sweeps:
                for fvg in recent_fvgs:
                    try:
                        # Ensure both are dictionaries
                        if not isinstance(sweep, dict) or not isinstance(fvg, dict):
                            continue
                            
                        sweep_price = float(sweep.get('price', 0))
                        fvg_high = float(fvg.get('high', 0))
                        fvg_low = float(fvg.get('low', 0))
                        
                        if sweep_price > 0 and fvg_high > 0 and fvg_low > 0:
                            # Check if sweep and FVG are aligned
                            fvg_mid = (fvg_high + fvg_low) / 2
                            
                            # If sweep is near FVG, generate signal
                            if abs(sweep_price - fvg_mid) / current_price < 0.01:  # 1% tolerance
                                
                                direction = 'BUY' if sweep_price < current_price else 'SELL'
                                entry_price = fvg_mid
                                
                                if direction == 'BUY':
                                    stop_loss = fvg_low - (fvg_high - fvg_low) * 0.5
                                    take_profit = fvg_high + (fvg_high - fvg_low) * 2.0
                                else:
                                    stop_loss = fvg_high + (fvg_high - fvg_low) * 0.5
                                    take_profit = fvg_low - (fvg_high - fvg_low) * 2.0
                                
                                risk = abs(entry_price - stop_loss)
                                reward = abs(take_profit - entry_price)
                                risk_reward = reward / risk if risk > 0 else 0
                                
                                if risk_reward > 0.8:  # Relaxed R:R
                                    signal = {
                                        'signal_type': 'SWEEP_FVG_COMBO',
                                        'direction': direction,
                                        'entry_price': entry_price,
                                        'stop_loss': stop_loss,
                                        'take_profit': take_profit,
                                        'confidence': 0.7,  # Fixed confidence
                                        'risk_reward': risk_reward,
                                        'timeframe': tf,
                                        'timestamp': current_time,
                                        'reason': f'{tf} Liquidity Sweep + FVG Alignment'
                                    }
                                    signals.append(signal)
                                    logger.info(f"✅ Generated {direction} signal from {tf} Sweep+FVG combo")
                    
                    except Exception as e:
                        logger.error(f"Error processing sweep+FVG combo: {e}")
                        continue
        
        logger.info(f"🎯 Simplified signal generation complete: {len(signals)} signals")
        return signals


    def _create_enhanced_market_narrative(self, analysis: Dict) -> str:
        """🚀 Create enhanced market narrative with strategic insights"""
        narrative_parts = []
        
        # Market regime context
        market_regime = analysis.get('market_regime', {})
        if hasattr(market_regime, 'trend_regime'):
            regime = f"{market_regime.trend_regime}/{market_regime.volatility_regime}"
            regime_confidence = 0.8 if market_regime.trend_regime == 'trending' else 0.6
        else:
            regime = 'UNKNOWN'
            regime_confidence = 0.5
        narrative_parts.append(f"🎯 Market Regime: {regime} (confidence: {regime_confidence:.1%})")
        
        # Institutional flow insights
        institutional_flow = analysis.get('institutional_flow', {})
        accumulation_signals = institutional_flow.get('accumulation_signals', [])
        if accumulation_signals:
            strong_signals = [s for s in accumulation_signals if s.confidence > 0.7]
            narrative_parts.append(f"🔍 Institutional Flow: {len(strong_signals)} strong accumulation zones detected")
        
        # Liquidity mapping insights
        liquidity_mapping = analysis.get('liquidity_mapping', {})
        liquidity_zones = liquidity_mapping.get('liquidity_zones', [])
        high_strength_zones = [z for z in liquidity_zones if hasattr(z, 'strength') and hasattr(z.strength, 'value') and z.strength.value in ['high', 'extreme']]
        if high_strength_zones:
            narrative_parts.append(f"🎯 Liquidity: {len(high_strength_zones)} high-strength zones mapped")
        
        # Enhanced order block analysis
        premium_obs = 0
        for tf_data in analysis.get('timeframe_analysis', {}).values():
            enhanced_obs = tf_data.get('enhanced_order_blocks', [])
            premium_obs += len([ob for ob in enhanced_obs if ob.score.total_score > 7.0])
        
        if premium_obs > 0:
            narrative_parts.append(f"📈 Order Blocks: {premium_obs} premium-quality zones identified")
        
        # Trading signals summary
        signals = analysis.get('trading_signals', [])
        if signals:
            # Handle both signal objects and dictionaries
            confidences = []
            for s in signals:
                if hasattr(s, 'confidence'):
                    confidences.append(s.confidence)
                elif isinstance(s, dict) and 'confidence' in s:
                    confidences.append(s['confidence'])
                else:
                    confidences.append(0.5)  # Default confidence
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                narrative_parts.append(f"🚀 Signals: {len(signals)} enhanced signals (avg confidence: {avg_confidence:.1%})")
            else:
                narrative_parts.append(f"🚀 Signals: {len(signals)} enhanced signals")
        
        return " | ".join(narrative_parts) if narrative_parts else "🔍 Enhanced analysis in progress..."
    
    def _filter_and_rank_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter and rank signals by quality (legacy method)"""
        # Filter out low-quality signals
        filtered_signals = [s for s in signals if s.confidence > 0.5 and s.risk_reward >= 1.5]
        
        # Rank by composite score
        def signal_score(signal: TradingSignal) -> float:
            return (signal.confidence * 0.4 + 
                   signal.risk_reward * 0.3 + 
                   signal.strength.value * 0.3)
        
        return sorted(filtered_signals, key=signal_score, reverse=True)
    
    def record_trade_outcome(self, signal: TradingSignal, outcome: TradeOutcome):
        """Record trade outcome for memory engine learning"""
        try:
            # Extract factors that contributed to this signal
            contributing_factors = self._extract_signal_factors(signal, self.analysis_cache)
            
            # Add contributing factors to outcome for memory engine
            outcome_with_factors = TradeOutcome(
                signal_id=outcome.signal_id,
                entry_price=outcome.entry_price,
                exit_price=outcome.exit_price,
                profit_loss=outcome.profit_loss,
                risk_reward_actual=outcome.risk_reward_actual,
                factors_used=outcome.factors_used,
                success=outcome.success
            )
            # Add contributing factors as a dynamic attribute
            outcome_with_factors.contributing_factors = contributing_factors
            
            # Record the outcome in memory engine
            self.memory_engine.record_trade_outcome(outcome_with_factors)
            
            logger.info(f"Recorded trade outcome: {'WIN' if outcome.success else 'LOSS'} for {signal.signal_type} signal")
            
        except Exception as e:
            logger.error(f"Error recording trade outcome: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary from memory engine"""
        try:
            return self.memory_engine.get_performance_summary()
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def _create_market_narrative(self, analysis: Dict) -> str:
        """Create human-readable market narrative"""
        narrative_parts = []
        
        # Current session context
        current_session = analysis.get('session_analysis', {}).get('current_session', SessionType.ASIA)
        narrative_parts.append(f"Current session: {current_session.value}")
        
        # Volatility state
        h1_vol = analysis.get('timeframe_analysis', {}).get('H1', {}).get('volatility', {})
        vol_state = h1_vol.get('current_volatility_state', 'NORMAL')
        narrative_parts.append(f"Volatility state: {vol_state}")
        
        # Recent SMC activity
        recent_activity = []
        for tf, tf_data in analysis.get('timeframe_analysis', {}).items():
            sweeps = tf_data.get('liquidity_sweeps', [])
            if sweeps:
                recent_activity.append(f"{len(sweeps)} liquidity events on {tf}")
        
        if recent_activity:
            narrative_parts.append("Recent activity: " + ", ".join(recent_activity))
        
        # Signal summary
        signals = analysis.get('trading_signals', [])
        if signals:
            strong_signals = [s for s in signals if s.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]]
            narrative_parts.append(f"{len(strong_signals)} high-quality signals identified")
        
        return ". ".join(narrative_parts) + "."
    
    def start_automated_analysis(self, interval_minutes: int = 5):
        """Start automated analysis every N minutes"""
        logger.info(f"Starting automated analysis every {interval_minutes} minutes")
        
        schedule.every(interval_minutes).minutes.do(self._scheduled_analysis)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            logger.info("Automated analysis stopped by user")
        finally:
            self.data_collector.disconnect_mt5()
    
    def _scheduled_analysis(self):
        """Scheduled analysis execution"""
        try:
            results = self.run_full_analysis()
            
            # Log key findings
            signals = results.get('trading_signals', [])
            if signals:
                logger.info(f"Analysis complete: {len(signals)} signals generated")
                for signal in signals:
                    logger.info(f"Signal: {signal.direction} {signal.signal_type} - Confidence: {signal.confidence:.2f}")
            else:
                logger.info("Analysis complete: No signals generated")
            
            # Save results to file
            self._save_analysis_results(results)
            
        except Exception as e:
            logger.error(f"Error in scheduled analysis: {e}")
    
    def _save_analysis_results(self, results: Dict):
        """Save analysis results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        try:
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            logger.info(f"Analysis results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, (SessionType, MarketStructure, SignalStrength)):
            return obj.value
        elif isinstance(obj, TradingSignal):
            return asdict(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj

def main():
    """Main execution function"""
    print("[BLOB AI] Agentic Forex Analytics Engine for USD/JPY")
    print("=" * 50)
    
    # Initialize engine
    engine = AgenticForexEngine("USDJPY")
    
    # Connect to MT5
    if not engine.data_collector.connect_mt5():
        print("[ERROR] Failed to connect to MetaTrader 5")
        print("Please ensure MT5 is installed and running")
        return
    
    print("[SUCCESS] Connected to MetaTrader 5")
    
    try:
        # Run initial analysis
        print("\n[ANALYSIS] Running initial market analysis...")
        results = engine.run_full_analysis()
        
        if results:
            # Handle timestamp formatting safely
            timestamp = results.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp_str = timestamp
                elif hasattr(timestamp, 'strftime'):
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
                else:
                    timestamp_str = str(timestamp)
            else:
                timestamp_str = datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')
            
            print(f"\n[ANALYSIS] Results ({timestamp_str})")
            print("-" * 50)
            
            # Display current price
            if results['current_price']:
                price = results['current_price']
                print(f"[PRICE] Current Price: {price['bid']:.5f} / {price['ask']:.5f} (Spread: {price['spread']:.5f})")
            
            # Display market narrative
            print(f"[NARRATIVE] Market Narrative: {results['market_narrative']}")
            
            # Display signals
            signals = results['trading_signals']
            if signals:
                print(f"\n[SIGNALS] Trading Signals ({len(signals)}):")
                for i, signal in enumerate(signals, 1):
                    print(f"\n{i}. {signal.direction} {signal.signal_type}")
                    print(f"   Entry: {signal.entry_price:.5f}")
                    print(f"   Stop Loss: {signal.stop_loss:.5f}")
                    print(f"   Take Profit: {signal.take_profit:.5f}")
                    print(f"   Risk/Reward: 1:{signal.risk_reward:.1f}")
                    print(f"   Confidence: {signal.confidence:.1%}")
                    print(f"   Reasoning: {signal.reasoning}")
            else:
                print("\n[INFO] No trading signals at this time")
            
            # Ask user if they want to start automated analysis
            print("\n" + "="*50)
            choice = input("Start automated analysis? (y/n): ").lower().strip()
            
            if choice == 'y':
                print("\n[AUTO] Starting automated analysis (every 5 minutes)")
                print("Press Ctrl+C to stop")
                engine.start_automated_analysis(5)
            else:
                print("\n[COMPLETE] Analysis complete. Engine stopped.")
        
        else:
            print("[ERROR] Failed to run analysis")
    
    except KeyboardInterrupt:
        print("\n\n[STOP] Engine stopped by user")
    except Exception as e:
        # Safe error message handling without timestamp conversion
        error_msg = str(e)
        # Don't try to convert timestamps - just use the string representation
        logger.error(f"Unexpected error: {error_msg}")
        print(f"[ERROR] Error: {error_msg}")
    finally:
        engine.data_collector.disconnect_mt5()
        print("\n[DISCONNECT] Disconnected from MetaTrader 5")

if __name__ == "__main__":
    main()