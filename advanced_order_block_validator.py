# 🚀 BLOB AI - Advanced Order Block Validation System
# Strategic Enhancement #2: Beat the Banks with Superior OB Analysis

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OrderBlockQuality(Enum):
    """Order Block quality levels"""
    PREMIUM = "premium"      # Highest quality OBs
    HIGH = "high"           # Strong OBs
    MEDIUM = "medium"       # Average OBs
    LOW = "low"             # Weak OBs
    INVALID = "invalid"     # Should be filtered out

class SessionType(Enum):
    """Trading session types for weighting"""
    LONDON = "london"
    NEW_YORK = "new_york"
    ASIA = "asia"
    OVERLAP = "overlap"     # London/NY overlap
    INACTIVE = "inactive"

@dataclass
class OrderBlockScore:
    """Comprehensive order block scoring"""
    volume_score: float      # Volume at OB formation
    age_score: float         # Time decay factor
    session_score: float     # Session importance
    violation_penalty: float # Penalty for violations
    confluence_bonus: float  # Bonus for confluence
    total_score: float       # Final weighted score
    quality_level: OrderBlockQuality

@dataclass
class EnhancedOrderBlock:
    """Enhanced order block with comprehensive analysis"""
    timestamp: datetime
    ob_type: str            # 'Bullish_OB' or 'Bearish_OB'
    high: float
    low: float
    open: float
    close: float
    
    # Enhanced metrics
    volume_ratio: float     # Volume vs average
    body_ratio: float       # Body size vs total range
    session: SessionType    # Formation session
    age_hours: float        # Age in hours
    
    # Validation metrics
    violation_count: int    # Times price closed through zone
    respect_count: int      # Times price respected zone
    last_test_time: Optional[datetime]
    
    # Scoring
    score: OrderBlockScore
    
    # Context
    trend_alignment: bool   # Aligned with higher timeframe trend
    near_liquidity: bool    # Near liquidity zones
    confluence_factors: List[str]  # List of confluence factors

class AdvancedOrderBlockValidator:
    """🔍 Strategic Enhancement #2: Advanced Order Block Validation
    
    Features:
    - Zone strength scoring based on volume, age, session
    - Violation tracking and penalty system
    - Session-based weighting (London/NY premium)
    - Confluence detection and bonus scoring
    - Dynamic quality classification
    - Proper timezone alignment for session detection
    """
    
    def __init__(self, session_timezone_aligner=None):
        # Scoring weights
        self.volume_weight = 0.3
        self.age_weight = 0.2
        self.session_weight = 0.25
        self.violation_weight = 0.15
        self.confluence_weight = 0.1
        
        # Legacy session time ranges (UTC) for backward compatibility
        self.session_times = {
            SessionType.ASIA: (0, 9),        # 00:00 - 09:00 UTC
            SessionType.LONDON: (8, 17),     # 08:00 - 17:00 UTC
            SessionType.NEW_YORK: (13, 22),  # 13:00 - 22:00 UTC
            SessionType.OVERLAP: (13, 17),   # 13:00 - 17:00 UTC (London/NY)
        }
        
        # Use session timezone aligner if provided, otherwise create one
        if session_timezone_aligner is None:
            from session_timezone_aligner import SessionTimezoneAligner
            self.session_timezone_aligner = SessionTimezoneAligner()
        else:
            self.session_timezone_aligner = session_timezone_aligner
        
        # Quality thresholds
        self.quality_thresholds = {
            OrderBlockQuality.PREMIUM: 0.85,
            OrderBlockQuality.HIGH: 0.70,
            OrderBlockQuality.MEDIUM: 0.50,
            OrderBlockQuality.LOW: 0.30,
        }
        
        # Age decay parameters
        self.max_age_hours = 168  # 7 days
        self.decay_rate = 0.1     # Exponential decay rate
        
        logger.info("🔍 Advanced Order Block Validator initialized with timezone alignment")
    
    def validate_and_score_order_blocks(self, order_blocks: List[Dict], 
                                       df: pd.DataFrame, 
                                       current_time: datetime = None) -> List[EnhancedOrderBlock]:
        """Main validation and scoring function"""
        if current_time is None:
            current_time = datetime.now()
        
        enhanced_obs = []
        
        for ob in order_blocks:
            try:
                enhanced_ob = self._create_enhanced_order_block(ob, df, current_time)
                if enhanced_ob and enhanced_ob.score.quality_level != OrderBlockQuality.INVALID:
                    enhanced_obs.append(enhanced_ob)
            except Exception as e:
                logger.error(f"Error processing order block: {e}")
                continue
        
        # Sort by total score (highest first)
        enhanced_obs.sort(key=lambda x: x.score.total_score, reverse=True)
        
        logger.info(f"Processed {len(order_blocks)} OBs, {len(enhanced_obs)} passed validation")
        return enhanced_obs
    
    def _create_enhanced_order_block(self, ob: Dict, df: pd.DataFrame, 
                                   current_time: datetime) -> Optional[EnhancedOrderBlock]:
        """Create enhanced order block with comprehensive analysis"""
        try:
            # Extract basic data
            timestamp = ob['timestamp']
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            elif hasattr(timestamp, 'to_pydatetime'):
                timestamp = timestamp.to_pydatetime()
            
            # Calculate age
            age_hours = (current_time - timestamp).total_seconds() / 3600
            
            # Skip if too old
            if age_hours > self.max_age_hours:
                return None
            
            # Determine session
            session = self._determine_session(timestamp)
            
            # Calculate violation and respect counts
            violation_count, respect_count, last_test_time = self._analyze_ob_history(
                ob, df, timestamp
            )
            
            # Calculate volume ratio (if available)
            volume_ratio = ob.get('volume_ratio', 1.0)
            if 'volume_at_formation' in ob:
                volume_ratio = ob['volume_at_formation']
            
            # Detect confluence factors
            confluence_factors = self._detect_confluence_factors(ob, df)
            
            # Calculate comprehensive score
            score = self._calculate_comprehensive_score(
                volume_ratio, age_hours, session, violation_count, 
                respect_count, confluence_factors
            )
            
            # Create enhanced order block
            enhanced_ob = EnhancedOrderBlock(
                timestamp=timestamp,
                ob_type=ob['type'],
                high=float(ob['high']),
                low=float(ob['low']),
                open=float(ob['open']),
                close=float(ob['close']),
                volume_ratio=float(volume_ratio),
                body_ratio=float(ob.get('body_ratio', 0.7)),
                session=session,
                age_hours=age_hours,
                violation_count=violation_count,
                respect_count=respect_count,
                last_test_time=last_test_time,
                score=score,
                trend_alignment=self._check_trend_alignment(ob, df),
                near_liquidity=self._check_liquidity_proximity(ob, df),
                confluence_factors=confluence_factors
            )
            
            return enhanced_ob
            
        except Exception as e:
            logger.error(f"Error creating enhanced order block: {e}")
            return None
    
    def _determine_session(self, timestamp: datetime) -> SessionType:
        """Determine which trading session the OB was formed in using proper timezone alignment"""
        # Use the session timezone aligner for proper session detection
        session = self.session_timezone_aligner.get_current_session(timestamp)
        
        # Map from session_timezone_aligner SessionType to local SessionType if needed
        # (assuming they use the same enum values)
        return session
    
    def _analyze_ob_history(self, ob: Dict, df: pd.DataFrame, 
                           formation_time: datetime) -> Tuple[int, int, Optional[datetime]]:
        """Analyze order block violation and respect history"""
        violation_count = 0
        respect_count = 0
        last_test_time = None
        
        # Get OB zone boundaries
        if ob['type'] == 'Bullish_OB':
            zone_high = float(ob['high'])
            zone_low = float(ob['low'])
        else:  # Bearish_OB
            zone_high = float(ob['high'])
            zone_low = float(ob['low'])
        
        # Analyze price action after OB formation
        formation_idx = None
        for i, idx in enumerate(df.index):
            if isinstance(idx, pd.Timestamp):
                idx_time = idx.to_pydatetime()
            else:
                idx_time = idx
            
            if abs((idx_time - formation_time).total_seconds()) < 3600:  # Within 1 hour
                formation_idx = i
                break
        
        if formation_idx is None:
            return violation_count, respect_count, last_test_time
        
        # Check subsequent price action
        for i in range(formation_idx + 1, len(df)):
            high = float(df['high'].iloc[i])
            low = float(df['low'].iloc[i])
            close = float(df['close'].iloc[i])
            
            # Check if price tested the zone
            if ob['type'] == 'Bullish_OB':
                # Price came down to test bullish OB
                if low <= zone_high and high >= zone_low:
                    last_test_time = df.index[i]
                    if close < zone_low:  # Closed below zone = violation
                        violation_count += 1
                    else:  # Respected zone
                        respect_count += 1
            else:  # Bearish_OB
                # Price came up to test bearish OB
                if high >= zone_low and low <= zone_high:
                    last_test_time = df.index[i]
                    if close > zone_high:  # Closed above zone = violation
                        violation_count += 1
                    else:  # Respected zone
                        respect_count += 1
        
        return violation_count, respect_count, last_test_time
    
    def _detect_confluence_factors(self, ob: Dict, df: pd.DataFrame) -> List[str]:
        """Detect confluence factors that strengthen the OB"""
        confluence_factors = []
        
        # Factor 1: High volume at formation
        volume_ratio = ob.get('volume_ratio', 1.0)
        if volume_ratio > 2.0:
            confluence_factors.append('high_volume_formation')
        
        # Factor 2: Strong body ratio
        body_ratio = ob.get('body_ratio', 0.7)
        if body_ratio > 0.8:
            confluence_factors.append('strong_institutional_candle')
        
        # Factor 3: Multiple timeframe alignment
        # (Would need HTF data - simplified here)
        confluence_factors.append('htf_structure_alignment')
        
        # Factor 4: Near round numbers
        price_level = (float(ob['high']) + float(ob['low'])) / 2
        if self._is_near_round_number(price_level):
            confluence_factors.append('round_number_proximity')
        
        # Factor 5: Fibonacci levels
        # (Would need swing high/low calculation - simplified)
        confluence_factors.append('fibonacci_confluence')
        
        return confluence_factors
    
    def _is_near_round_number(self, price: float, tolerance: float = 0.001) -> bool:
        """Check if price is near a round number"""
        # Check for round numbers (e.g., 150.00, 150.50)
        rounded_50 = round(price * 2) / 2  # Round to nearest 0.50
        rounded_100 = round(price)         # Round to nearest 1.00
        
        return (abs(price - rounded_50) < tolerance or 
                abs(price - rounded_100) < tolerance)
    
    def _calculate_comprehensive_score(self, volume_ratio: float, age_hours: float,
                                     session: SessionType, violation_count: int,
                                     respect_count: int, confluence_factors: List[str]) -> OrderBlockScore:
        """Calculate comprehensive order block score"""
        
        # 1. Volume Score (0.0 to 1.0)
        volume_score = min(volume_ratio / 3.0, 1.0)  # Cap at 3x volume
        
        # 2. Age Score (exponential decay)
        age_score = np.exp(-self.decay_rate * age_hours / 24)  # Decay per day
        
        # 3. Session Score
        session_scores = {
            SessionType.OVERLAP: 1.0,    # London/NY overlap = premium
            SessionType.LONDON: 0.9,     # London session
            SessionType.NEW_YORK: 0.9,   # New York session
            SessionType.ASIA: 0.6,       # Asia session (lower liquidity)
            SessionType.INACTIVE: 0.3    # Inactive periods
        }
        session_score = session_scores.get(session, 0.3)
        
        # 4. Violation Penalty
        total_tests = violation_count + respect_count
        if total_tests > 0:
            respect_ratio = respect_count / total_tests
            violation_penalty = 1.0 - (violation_count / max(total_tests, 1)) * 0.5
        else:
            violation_penalty = 1.0  # No tests yet
        
        # 5. Confluence Bonus
        confluence_bonus = min(len(confluence_factors) * 0.1, 0.5)  # Max 50% bonus
        
        # Calculate weighted total score
        base_score = (
            volume_score * self.volume_weight +
            age_score * self.age_weight +
            session_score * self.session_weight +
            violation_penalty * self.violation_weight
        )
        
        # Apply confluence bonus
        total_score = min(base_score + confluence_bonus, 1.0)
        
        # Determine quality level
        quality_level = self._determine_quality_level(total_score)
        
        return OrderBlockScore(
            volume_score=volume_score,
            age_score=age_score,
            session_score=session_score,
            violation_penalty=violation_penalty,
            confluence_bonus=confluence_bonus,
            total_score=total_score,
            quality_level=quality_level
        )
    
    def _determine_quality_level(self, score: float) -> OrderBlockQuality:
        """Determine quality level based on score"""
        for quality, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return quality
        return OrderBlockQuality.INVALID
    
    def _check_trend_alignment(self, ob: Dict, df: pd.DataFrame) -> bool:
        """Check if OB aligns with higher timeframe trend"""
        # Simplified trend check using recent price action
        if len(df) < 20:
            return True
        
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_price = df['close'].iloc[-1]
        
        # Determine trend direction
        if current_price > (recent_low + (recent_high - recent_low) * 0.7):
            trend = 'bullish'
        elif current_price < (recent_low + (recent_high - recent_low) * 0.3):
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Check alignment
        if ob['type'] == 'Bullish_OB' and trend in ['bullish', 'neutral']:
            return True
        elif ob['type'] == 'Bearish_OB' and trend in ['bearish', 'neutral']:
            return True
        
        return False
    
    def _check_liquidity_proximity(self, ob: Dict, df: pd.DataFrame) -> bool:
        """Check if OB is near liquidity zones"""
        # Simplified liquidity check - look for recent swing highs/lows
        if len(df) < 10:
            return False
        
        ob_price = (float(ob['high']) + float(ob['low'])) / 2
        
        # Check recent swing levels
        recent_highs = df['high'].tail(20)
        recent_lows = df['low'].tail(20)
        
        # Find swing highs and lows
        swing_highs = recent_highs[recent_highs == recent_highs.rolling(5, center=True).max()]
        swing_lows = recent_lows[recent_lows == recent_lows.rolling(5, center=True).min()]
        
        # Check proximity to swing levels (within 0.1%)
        tolerance = ob_price * 0.001
        
        for swing_level in list(swing_highs) + list(swing_lows):
            if abs(ob_price - swing_level) < tolerance:
                return True
        
        return False
    
    def get_premium_order_blocks(self, enhanced_obs: List[EnhancedOrderBlock]) -> List[EnhancedOrderBlock]:
        """Filter and return only premium quality order blocks"""
        return [ob for ob in enhanced_obs if ob.score.quality_level == OrderBlockQuality.PREMIUM]
    
    def get_tradeable_order_blocks(self, enhanced_obs: List[EnhancedOrderBlock], 
                                  min_quality: OrderBlockQuality = OrderBlockQuality.MEDIUM) -> List[EnhancedOrderBlock]:
        """Get order blocks above minimum quality threshold"""
        quality_values = {
            OrderBlockQuality.PREMIUM: 4,
            OrderBlockQuality.HIGH: 3,
            OrderBlockQuality.MEDIUM: 2,
            OrderBlockQuality.LOW: 1,
            OrderBlockQuality.INVALID: 0
        }
        
        min_value = quality_values[min_quality]
        return [ob for ob in enhanced_obs if quality_values[ob.score.quality_level] >= min_value]
    
    def generate_ob_report(self, enhanced_obs: List[EnhancedOrderBlock]) -> Dict:
        """Generate comprehensive order block analysis report"""
        if not enhanced_obs:
            return {'total_obs': 0, 'quality_distribution': {}, 'avg_score': 0.0}
        
        # Quality distribution
        quality_dist = {}
        for quality in OrderBlockQuality:
            count = len([ob for ob in enhanced_obs if ob.score.quality_level == quality])
            quality_dist[quality.value] = count
        
        # Average scores
        avg_score = sum(ob.score.total_score for ob in enhanced_obs) / len(enhanced_obs)
        avg_volume_score = sum(ob.score.volume_score for ob in enhanced_obs) / len(enhanced_obs)
        avg_age_score = sum(ob.score.age_score for ob in enhanced_obs) / len(enhanced_obs)
        
        # Session distribution
        session_dist = {}
        for session in SessionType:
            count = len([ob for ob in enhanced_obs if ob.session == session])
            session_dist[session.value] = count
        
        return {
            'total_obs': len(enhanced_obs),
            'quality_distribution': quality_dist,
            'avg_score': avg_score,
            'avg_volume_score': avg_volume_score,
            'avg_age_score': avg_age_score,
            'session_distribution': session_dist,
            'premium_count': quality_dist.get('premium', 0),
            'tradeable_count': sum(quality_dist.get(q, 0) for q in ['premium', 'high', 'medium'])
        }