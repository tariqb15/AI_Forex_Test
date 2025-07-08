# 🚀 BLOB AI - Liquidity Mapping System
# Strategic Enhancement #3: Beat the Banks with Liquidity Intelligence

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

class LiquidityType(Enum):
    """Types of liquidity zones"""
    STOP_CLUSTER = "stop_cluster"          # Clustered stop losses
    ROUND_NUMBER = "round_number"          # Psychological levels
    SWING_HIGH = "swing_high"              # Recent swing highs
    SWING_LOW = "swing_low"                # Recent swing lows
    ORDER_BLOCK = "order_block"            # Unmitigated order blocks
    FAIR_VALUE_GAP = "fair_value_gap"      # Unfilled FVGs
    DAILY_LEVEL = "daily_level"            # Daily high/low levels
    WEEKLY_LEVEL = "weekly_level"          # Weekly high/low levels

class LiquidityStrength(Enum):
    """Liquidity zone strength levels"""
    EXTREME = "extreme"    # Very high liquidity
    HIGH = "high"         # High liquidity
    MEDIUM = "medium"     # Medium liquidity
    LOW = "low"           # Low liquidity

@dataclass
class LiquidityZone:
    """Represents a liquidity zone"""
    price_level: float
    liquidity_type: LiquidityType
    strength: LiquidityStrength
    volume_estimate: float      # Estimated volume at this level
    formation_time: datetime
    last_test_time: Optional[datetime]
    test_count: int            # How many times tested
    absorption_events: int     # Times liquidity was absorbed
    
    # Zone boundaries
    upper_bound: float
    lower_bound: float
    
    # Metadata
    confidence: float          # 0.0 to 1.0
    magnetic_strength: float   # How much price is attracted
    sweep_probability: float   # Probability of being swept

@dataclass
class LiquidityHeatmap:
    """Liquidity heatmap for visualization"""
    price_levels: List[float]
    liquidity_density: List[float]
    zone_types: List[LiquidityType]
    strength_levels: List[LiquidityStrength]
    
@dataclass
class AbsorptionEvent:
    """Represents liquidity absorption"""
    timestamp: datetime
    price_level: float
    volume_absorbed: float
    absorption_type: str       # 'buying', 'selling', 'mixed'
    smart_money_signature: bool # Shows institutional characteristics

class LiquidityMappingSystem:
    """🔍 Strategic Enhancement #3: Liquidity Mapping System
    
    Features:
    - Stop loss cluster detection
    - Round number liquidity mapping
    - Wick cluster analysis for stop inference
    - Liquidity absorption detection
    - Smart money defending level identification
    - Liquidity heatmap generation
    """
    
    def __init__(self):
        # Configuration
        self.wick_cluster_threshold = 3      # Min wicks for cluster
        self.round_number_tolerance = 0.0005 # 0.05% tolerance for round numbers
        self.swing_lookback = 20             # Bars to look back for swings
        self.absorption_volume_threshold = 2.0 # Volume threshold for absorption
        
        # Liquidity tracking
        self.liquidity_zones = []
        self.absorption_events = []
        self.heatmap_resolution = 50         # Number of price levels in heatmap
        
        # Round number levels (for USDJPY)
        self.round_numbers = self._generate_round_numbers(140, 160, [0.00, 0.25, 0.50, 0.75])
        
        logger.info("🔍 Liquidity Mapping System initialized")
    
    def analyze_liquidity_landscape(self, df: pd.DataFrame, 
                                  order_blocks: List[Dict] = None,
                                  fvgs: List[Dict] = None) -> Dict:
        """Main analysis function for liquidity mapping"""
        try:
            if len(df) < self.swing_lookback:
                return self._empty_liquidity_analysis()
            
            # Clear old data
            self.liquidity_zones.clear()
            self.absorption_events.clear()
            
            # Detect different types of liquidity
            stop_clusters = self._detect_stop_clusters(df)
            swing_levels = self._detect_swing_levels(df)
            round_number_zones = self._detect_round_number_liquidity(df)
            
            # Add order block and FVG liquidity
            if order_blocks:
                ob_liquidity = self._map_order_block_liquidity(order_blocks)
                self.liquidity_zones.extend(ob_liquidity)
            
            if fvgs:
                fvg_liquidity = self._map_fvg_liquidity(fvgs)
                self.liquidity_zones.extend(fvg_liquidity)
            
            # Combine all liquidity zones
            self.liquidity_zones.extend(stop_clusters)
            self.liquidity_zones.extend(swing_levels)
            self.liquidity_zones.extend(round_number_zones)
            
            # Detect absorption events
            absorption_events = self._detect_absorption_events(df)
            
            # Generate liquidity heatmap
            heatmap = self._generate_liquidity_heatmap(df)
            
            # Analyze liquidity flow
            flow_analysis = self._analyze_liquidity_flow(df)
            
            # Find high-probability sweep targets
            sweep_targets = self._identify_sweep_targets()
            
            return {
                'liquidity_zones': self.liquidity_zones,
                'absorption_events': absorption_events,
                'liquidity_heatmap': heatmap,
                'flow_analysis': flow_analysis,
                'sweep_targets': sweep_targets,
                'total_zones': len(self.liquidity_zones),
                'high_strength_zones': len([z for z in self.liquidity_zones if z.strength in [LiquidityStrength.HIGH, LiquidityStrength.EXTREME]])
            }
            
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            return self._empty_liquidity_analysis()
    
    def _detect_stop_clusters(self, df: pd.DataFrame) -> List[LiquidityZone]:
        """🎯 Detect stop loss clusters using wick analysis"""
        stop_clusters = []
        
        # Analyze wicks to infer stop loss locations
        upper_wicks = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wicks = df[['open', 'close']].min(axis=1) - df['low']
        
        # Find significant wicks (potential stop hunts)
        body_size = abs(df['close'] - df['open'])
        avg_body = body_size.rolling(window=20).mean()
        
        significant_upper_wicks = upper_wicks > avg_body * 1.5
        significant_lower_wicks = lower_wicks > avg_body * 1.5
        
        # Cluster analysis for upper wicks (sell stops)
        upper_wick_levels = df.loc[significant_upper_wicks, 'high'].values
        upper_clusters = self._cluster_price_levels(upper_wick_levels, df)
        
        for cluster_price, cluster_strength, cluster_count in upper_clusters:
            if cluster_count >= self.wick_cluster_threshold:
                zone = LiquidityZone(
                    price_level=cluster_price,
                    liquidity_type=LiquidityType.STOP_CLUSTER,
                    strength=self._determine_liquidity_strength(cluster_strength, cluster_count),
                    volume_estimate=cluster_count * 100,  # Estimated volume
                    formation_time=datetime.now(),
                    last_test_time=None,
                    test_count=0,
                    absorption_events=0,
                    upper_bound=cluster_price + cluster_price * 0.0002,
                    lower_bound=cluster_price - cluster_price * 0.0002,
                    confidence=min(cluster_strength, 1.0),
                    magnetic_strength=cluster_strength * 0.8,
                    sweep_probability=cluster_strength * 0.9  # High probability for stop clusters
                )
                stop_clusters.append(zone)
        
        # Cluster analysis for lower wicks (buy stops)
        lower_wick_levels = df.loc[significant_lower_wicks, 'low'].values
        lower_clusters = self._cluster_price_levels(lower_wick_levels, df)
        
        for cluster_price, cluster_strength, cluster_count in lower_clusters:
            if cluster_count >= self.wick_cluster_threshold:
                zone = LiquidityZone(
                    price_level=cluster_price,
                    liquidity_type=LiquidityType.STOP_CLUSTER,
                    strength=self._determine_liquidity_strength(cluster_strength, cluster_count),
                    volume_estimate=cluster_count * 100,
                    formation_time=datetime.now(),
                    last_test_time=None,
                    test_count=0,
                    absorption_events=0,
                    upper_bound=cluster_price + cluster_price * 0.0002,
                    lower_bound=cluster_price - cluster_price * 0.0002,
                    confidence=min(cluster_strength, 1.0),
                    magnetic_strength=cluster_strength * 0.8,
                    sweep_probability=cluster_strength * 0.9
                )
                stop_clusters.append(zone)
        
        return stop_clusters
    
    def _cluster_price_levels(self, price_levels: np.ndarray, df: pd.DataFrame) -> List[Tuple[float, float, int]]:
        """Cluster price levels and return (price, strength, count)"""
        if len(price_levels) == 0:
            return []
        
        clusters = []
        current_price = df['close'].iloc[-1]
        cluster_tolerance = current_price * 0.001  # 0.1% clustering tolerance
        
        # Sort price levels
        sorted_levels = np.sort(price_levels)
        
        i = 0
        while i < len(sorted_levels):
            cluster_prices = [sorted_levels[i]]
            j = i + 1
            
            # Find all prices within tolerance
            while j < len(sorted_levels) and sorted_levels[j] - sorted_levels[i] <= cluster_tolerance:
                cluster_prices.append(sorted_levels[j])
                j += 1
            
            if len(cluster_prices) >= 2:  # At least 2 levels for a cluster
                cluster_center = np.mean(cluster_prices)
                cluster_strength = len(cluster_prices) / len(price_levels)
                cluster_count = len(cluster_prices)
                
                clusters.append((cluster_center, cluster_strength, cluster_count))
            
            i = j
        
        return clusters
    
    def _detect_swing_levels(self, df: pd.DataFrame) -> List[LiquidityZone]:
        """🎯 Detect swing high/low liquidity zones"""
        swing_zones = []
        
        # Calculate swing highs and lows
        swing_window = 5
        
        # Swing highs
        for i in range(swing_window, len(df) - swing_window):
            current_high = df['high'].iloc[i]
            is_swing_high = True
            
            # Check if current high is higher than surrounding highs
            for j in range(i - swing_window, i + swing_window + 1):
                if j != i and df['high'].iloc[j] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                # Calculate strength based on how recent and significant
                age_factor = (len(df) - i) / len(df)  # More recent = higher strength
                significance = (current_high - df['low'].iloc[i-swing_window:i+swing_window+1].min()) / current_high
                strength = age_factor * significance
                
                zone = LiquidityZone(
                    price_level=current_high,
                    liquidity_type=LiquidityType.SWING_HIGH,
                    strength=self._determine_liquidity_strength(strength, 1),
                    volume_estimate=200,  # Estimated stop volume
                    formation_time=df.index[i],
                    last_test_time=None,
                    test_count=0,
                    absorption_events=0,
                    upper_bound=current_high + current_high * 0.0001,
                    lower_bound=current_high - current_high * 0.0001,
                    confidence=min(strength * 2, 1.0),
                    magnetic_strength=strength * 0.7,
                    sweep_probability=strength * 0.8
                )
                swing_zones.append(zone)
        
        # Swing lows (similar logic)
        for i in range(swing_window, len(df) - swing_window):
            current_low = df['low'].iloc[i]
            is_swing_low = True
            
            for j in range(i - swing_window, i + swing_window + 1):
                if j != i and df['low'].iloc[j] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                age_factor = (len(df) - i) / len(df)
                significance = (df['high'].iloc[i-swing_window:i+swing_window+1].max() - current_low) / current_low
                strength = age_factor * significance
                
                zone = LiquidityZone(
                    price_level=current_low,
                    liquidity_type=LiquidityType.SWING_LOW,
                    strength=self._determine_liquidity_strength(strength, 1),
                    volume_estimate=200,
                    formation_time=df.index[i],
                    last_test_time=None,
                    test_count=0,
                    absorption_events=0,
                    upper_bound=current_low + current_low * 0.0001,
                    lower_bound=current_low - current_low * 0.0001,
                    confidence=min(strength * 2, 1.0),
                    magnetic_strength=strength * 0.7,
                    sweep_probability=strength * 0.8
                )
                swing_zones.append(zone)
        
        return swing_zones
    
    def _detect_round_number_liquidity(self, df: pd.DataFrame) -> List[LiquidityZone]:
        """🎯 Detect round number liquidity zones"""
        round_zones = []
        current_price = df['close'].iloc[-1]
        
        # Find relevant round numbers near current price
        relevant_rounds = []
        for round_num in self.round_numbers:
            if abs(round_num - current_price) / current_price < 0.02:  # Within 2%
                relevant_rounds.append(round_num)
        
        for round_price in relevant_rounds:
            # Calculate how often price has reacted to this level
            reaction_count = 0
            for i in range(len(df)):
                if abs(df['high'].iloc[i] - round_price) < round_price * self.round_number_tolerance:
                    reaction_count += 1
                elif abs(df['low'].iloc[i] - round_price) < round_price * self.round_number_tolerance:
                    reaction_count += 1
            
            if reaction_count > 0:
                strength = min(reaction_count / 10.0, 1.0)  # Normalize to 0-1
                
                zone = LiquidityZone(
                    price_level=round_price,
                    liquidity_type=LiquidityType.ROUND_NUMBER,
                    strength=self._determine_liquidity_strength(strength, reaction_count),
                    volume_estimate=reaction_count * 150,
                    formation_time=datetime.now(),
                    last_test_time=None,
                    test_count=reaction_count,
                    absorption_events=0,
                    upper_bound=round_price + round_price * self.round_number_tolerance,
                    lower_bound=round_price - round_price * self.round_number_tolerance,
                    confidence=strength,
                    magnetic_strength=strength * 0.9,  # Round numbers are very magnetic
                    sweep_probability=strength * 0.6   # Lower sweep probability
                )
                round_zones.append(zone)
        
        return round_zones
    
    def _map_order_block_liquidity(self, order_blocks: List[Dict]) -> List[LiquidityZone]:
        """🎯 Map order block zones as liquidity"""
        ob_zones = []
        
        for ob in order_blocks:
            price_level = (float(ob['high']) + float(ob['low'])) / 2
            strength = ob.get('strength', 0.5)
            
            zone = LiquidityZone(
                price_level=price_level,
                liquidity_type=LiquidityType.ORDER_BLOCK,
                strength=self._determine_liquidity_strength(strength, 1),
                volume_estimate=strength * 300,
                formation_time=ob['timestamp'],
                last_test_time=None,
                test_count=0,
                absorption_events=0,
                upper_bound=float(ob['high']),
                lower_bound=float(ob['low']),
                confidence=strength,
                magnetic_strength=strength * 0.8,
                sweep_probability=strength * 0.4  # OBs less likely to be swept
            )
            ob_zones.append(zone)
        
        return ob_zones
    
    def _map_fvg_liquidity(self, fvgs: List[Dict]) -> List[LiquidityZone]:
        """🎯 Map FVG zones as liquidity magnets"""
        fvg_zones = []
        
        for fvg in fvgs:
            if hasattr(fvg, 'top') and hasattr(fvg, 'bottom'):
                price_level = (fvg.top + fvg.bottom) / 2
                importance = getattr(fvg, 'importance', 0.5)
            else:
                price_level = (float(fvg['top']) + float(fvg['bottom'])) / 2
                importance = fvg.get('importance', 0.5)
            
            zone = LiquidityZone(
                price_level=price_level,
                liquidity_type=LiquidityType.FAIR_VALUE_GAP,
                strength=self._determine_liquidity_strength(importance, 1),
                volume_estimate=importance * 250,
                formation_time=fvg.get('timestamp', datetime.now()),
                last_test_time=None,
                test_count=0,
                absorption_events=0,
                upper_bound=fvg.get('top', price_level + 0.01),
                lower_bound=fvg.get('bottom', price_level - 0.01),
                confidence=importance,
                magnetic_strength=importance * 0.9,  # FVGs are very magnetic
                sweep_probability=importance * 0.3   # Low sweep probability
            )
            fvg_zones.append(zone)
        
        return fvg_zones
    
    def _detect_absorption_events(self, df: pd.DataFrame) -> List[AbsorptionEvent]:
        """🎯 Detect liquidity absorption events"""
        absorption_events = []
        
        # Calculate volume metrics
        volume = df['tick_volume'] if 'tick_volume' in df.columns else pd.Series([1] * len(df))
        volume_ma = volume.rolling(window=20).mean()
        volume_ratio = volume / volume_ma
        
        # Look for high volume with small price movement (absorption)
        for i in range(20, len(df)):
            if pd.isna(volume_ratio.iloc[i]):
                continue
            
            # High volume condition
            if volume_ratio.iloc[i] > self.absorption_volume_threshold:
                # Small price movement condition
                price_range = df['high'].iloc[i] - df['low'].iloc[i]
                avg_range = (df['high'] - df['low']).rolling(window=20).mean().iloc[i]
                
                if price_range < avg_range * 0.7:  # Small range despite high volume
                    # Determine absorption type
                    body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
                    upper_wick = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
                    lower_wick = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
                    
                    if upper_wick > body_size * 1.5:
                        absorption_type = 'selling'
                    elif lower_wick > body_size * 1.5:
                        absorption_type = 'buying'
                    else:
                        absorption_type = 'mixed'
                    
                    # Check for smart money signature
                    smart_money_signature = (
                        volume_ratio.iloc[i] > 2.5 and  # Very high volume
                        price_range < avg_range * 0.5   # Very small range
                    )
                    
                    event = AbsorptionEvent(
                        timestamp=df.index[i],
                        price_level=df['close'].iloc[i],
                        volume_absorbed=volume.iloc[i],
                        absorption_type=absorption_type,
                        smart_money_signature=smart_money_signature
                    )
                    absorption_events.append(event)
        
        return absorption_events
    
    def _generate_liquidity_heatmap(self, df: pd.DataFrame) -> LiquidityHeatmap:
        """🎯 Generate liquidity density heatmap"""
        if not self.liquidity_zones:
            return LiquidityHeatmap([], [], [], [])
        
        # Define price range for heatmap
        current_price = df['close'].iloc[-1]
        price_range = current_price * 0.02  # 2% range around current price
        
        min_price = current_price - price_range
        max_price = current_price + price_range
        
        # Create price levels
        price_levels = np.linspace(min_price, max_price, self.heatmap_resolution)
        liquidity_density = np.zeros(self.heatmap_resolution)
        zone_types = []
        strength_levels = []
        
        # Calculate liquidity density at each level
        for i, price_level in enumerate(price_levels):
            total_liquidity = 0
            dominant_type = LiquidityType.STOP_CLUSTER
            dominant_strength = LiquidityStrength.LOW
            max_influence = 0
            
            for zone in self.liquidity_zones:
                # Calculate influence based on distance
                distance = abs(zone.price_level - price_level)
                max_distance = price_range * 0.1  # 10% of range
                
                if distance <= max_distance:
                    influence = (1 - distance / max_distance) * zone.volume_estimate
                    total_liquidity += influence
                    
                    if influence > max_influence:
                        max_influence = influence
                        dominant_type = zone.liquidity_type
                        dominant_strength = zone.strength
            
            liquidity_density[i] = total_liquidity
            zone_types.append(dominant_type)
            strength_levels.append(dominant_strength)
        
        return LiquidityHeatmap(
            price_levels=price_levels.tolist(),
            liquidity_density=liquidity_density.tolist(),
            zone_types=zone_types,
            strength_levels=strength_levels
        )
    
    def _analyze_liquidity_flow(self, df: pd.DataFrame) -> Dict:
        """🎯 Analyze liquidity flow patterns"""
        # Simplified liquidity flow analysis
        current_price = df['close'].iloc[-1]
        
        # Find liquidity above and below current price
        liquidity_above = sum(zone.volume_estimate for zone in self.liquidity_zones 
                             if zone.price_level > current_price)
        liquidity_below = sum(zone.volume_estimate for zone in self.liquidity_zones 
                             if zone.price_level < current_price)
        
        total_liquidity = liquidity_above + liquidity_below
        
        if total_liquidity > 0:
            liquidity_bias = (liquidity_above - liquidity_below) / total_liquidity
        else:
            liquidity_bias = 0.0
        
        # Determine flow direction
        if liquidity_bias > 0.2:
            flow_direction = 'upward_bias'
        elif liquidity_bias < -0.2:
            flow_direction = 'downward_bias'
        else:
            flow_direction = 'balanced'
        
        return {
            'liquidity_above': liquidity_above,
            'liquidity_below': liquidity_below,
            'total_liquidity': total_liquidity,
            'liquidity_bias': liquidity_bias,
            'flow_direction': flow_direction
        }
    
    def _identify_sweep_targets(self) -> List[Dict]:
        """🎯 Identify high-probability liquidity sweep targets"""
        sweep_targets = []
        
        # Sort zones by sweep probability
        sorted_zones = sorted(self.liquidity_zones, key=lambda x: x.sweep_probability, reverse=True)
        
        # Take top sweep candidates
        for zone in sorted_zones[:10]:  # Top 10 candidates
            if zone.sweep_probability > 0.6:  # High probability threshold
                sweep_targets.append({
                    'price_level': zone.price_level,
                    'liquidity_type': zone.liquidity_type.value,
                    'sweep_probability': zone.sweep_probability,
                    'estimated_volume': zone.volume_estimate,
                    'strength': zone.strength.value,
                    'magnetic_strength': zone.magnetic_strength
                })
        
        return sweep_targets
    
    def _determine_liquidity_strength(self, strength_value: float, count: int) -> LiquidityStrength:
        """Determine liquidity strength level"""
        # Combine strength value and count
        combined_strength = strength_value * (1 + count * 0.1)
        
        if combined_strength >= 0.8:
            return LiquidityStrength.EXTREME
        elif combined_strength >= 0.6:
            return LiquidityStrength.HIGH
        elif combined_strength >= 0.3:
            return LiquidityStrength.MEDIUM
        else:
            return LiquidityStrength.LOW
    
    def _generate_round_numbers(self, min_price: float, max_price: float, 
                               decimals: List[float]) -> List[float]:
        """Generate round number levels"""
        round_numbers = []
        
        for price in range(int(min_price), int(max_price) + 1):
            for decimal in decimals:
                round_numbers.append(price + decimal)
        
        return round_numbers
    
    def _empty_liquidity_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'liquidity_zones': [],
            'absorption_events': [],
            'liquidity_heatmap': LiquidityHeatmap([], [], [], []),
            'flow_analysis': {},
            'sweep_targets': [],
            'total_zones': 0,
            'high_strength_zones': 0
        }
    
    def get_nearest_liquidity(self, current_price: float, direction: str = 'both') -> List[LiquidityZone]:
        """Get nearest liquidity zones in specified direction"""
        if direction == 'above':
            zones = [z for z in self.liquidity_zones if z.price_level > current_price]
            zones.sort(key=lambda x: x.price_level)
        elif direction == 'below':
            zones = [z for z in self.liquidity_zones if z.price_level < current_price]
            zones.sort(key=lambda x: x.price_level, reverse=True)
        else:  # both
            zones = self.liquidity_zones.copy()
            zones.sort(key=lambda x: abs(x.price_level - current_price))
        
        return zones[:5]  # Return top 5 nearest