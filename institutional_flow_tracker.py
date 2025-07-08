# 🚀 BLOB AI - Institutional Flow Tracking System
# Strategic Enhancement #1: Beat the Banks with Volume Intelligence

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AccumulationSignal(Enum):
    """Types of institutional accumulation signals"""
    STEALTH_ACCUMULATION = "stealth_accumulation"  # High volume, low volatility
    BLOCK_TRADE_CLUSTER = "block_trade_cluster"    # Large volume clusters
    ABSORPTION_ZONE = "absorption_zone"            # Price rejection with volume
    SMART_MONEY_ENTRY = "smart_money_entry"        # Institutional entry pattern

@dataclass
class VolumeCluster:
    """Represents a volume cluster for institutional flow detection"""
    timestamp: datetime
    price_level: float
    volume_ratio: float  # Volume vs average
    volatility_ratio: float  # Volatility vs average
    cluster_strength: float  # 0.0 to 1.0
    signal_type: AccumulationSignal
    confidence: float

@dataclass
class InstitutionalFlow:
    """Represents detected institutional flow"""
    timestamp: datetime
    flow_type: str  # 'accumulation', 'distribution', 'neutral'
    strength: float  # 0.0 to 1.0
    price_zone: Tuple[float, float]  # (low, high)
    volume_profile: Dict
    cvd_delta: float  # Cumulative Volume Delta
    vwap_deviation: float
    stealth_score: float  # How hidden the flow is

class InstitutionalFlowTracker:
    """🔍 Strategic Enhancement #1: Institutional Flow Tracking
    
    Detects institutional accumulation/distribution patterns:
    - Volume anomalies at range bottoms (not spikes)
    - High volume, low volatility candles (stealth accumulation)
    - VWAP clusters and delta volume analysis
    - Cumulative Volume Delta (CVD) tracking
    """
    
    def __init__(self):
        self.volume_window = 20  # Rolling window for volume analysis
        self.volatility_window = 14  # ATR calculation window
        self.stealth_threshold = 0.7  # Threshold for stealth accumulation
        self.cluster_distance = 0.001  # Price distance for clustering (0.1%)
        
        # Volume profile tracking
        self.volume_profile = {}
        self.cvd_history = []
        self.vwap_history = []
        
        logger.info("🔍 Institutional Flow Tracker initialized")
    
    def analyze_institutional_flow(self, df: pd.DataFrame, symbol: str = "USDJPY") -> Dict:
        """Main analysis function for institutional flow detection"""
        try:
            if len(df) < self.volume_window:
                return self._empty_flow_analysis()
            
            # Calculate volume metrics
            volume_metrics = self._calculate_volume_metrics(df)
            
            # Detect stealth accumulation patterns
            stealth_signals = self._detect_stealth_accumulation(df, volume_metrics)
            
            # Analyze volume clusters
            volume_clusters = self._analyze_volume_clusters(df, volume_metrics)
            
            # Calculate CVD and VWAP analysis
            cvd_analysis = self._calculate_cvd_analysis(df)
            vwap_analysis = self._calculate_vwap_analysis(df)
            
            # Detect absorption zones
            absorption_zones = self._detect_absorption_zones(df, volume_metrics)
            
            # Generate institutional flow summary
            flow_summary = self._generate_flow_summary(
                stealth_signals, volume_clusters, cvd_analysis, 
                vwap_analysis, absorption_zones
            )
            
            return {
                'institutional_flow': flow_summary,
                'stealth_accumulation': stealth_signals,
                'volume_clusters': volume_clusters,
                'cvd_analysis': cvd_analysis,
                'vwap_analysis': vwap_analysis,
                'absorption_zones': absorption_zones,
                'flow_strength': flow_summary.strength if flow_summary else 0.0,
                'flow_type': flow_summary.flow_type if flow_summary else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"Error in institutional flow analysis: {e}")
            return self._empty_flow_analysis()
    
    def _calculate_volume_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive volume metrics"""
        # Basic volume statistics
        volume = df['tick_volume'] if 'tick_volume' in df.columns else df.get('volume', pd.Series([1] * len(df)))
        
        # Rolling averages
        volume_ma = volume.rolling(window=self.volume_window).mean()
        volume_std = volume.rolling(window=self.volume_window).std()
        
        # Volume ratio (current vs average)
        volume_ratio = volume / volume_ma
        
        # Calculate ATR for volatility normalization
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.volatility_window).mean()
        
        # Volatility ratio
        current_volatility = true_range
        volatility_ratio = current_volatility / atr
        
        # Volume-Price Trend (VPT) - similar to OBV
        price_change = df['close'].pct_change()
        vpt = (volume * price_change).cumsum()
        
        return {
            'volume': volume,
            'volume_ma': volume_ma,
            'volume_ratio': volume_ratio,
            'atr': atr,
            'volatility_ratio': volatility_ratio,
            'vpt': vpt,
            'true_range': true_range
        }
    
    def _detect_stealth_accumulation(self, df: pd.DataFrame, volume_metrics: Dict) -> List[VolumeCluster]:
        """🎯 Detect stealth accumulation: High volume + Low volatility"""
        stealth_signals = []
        
        volume_ratio = volume_metrics['volume_ratio']
        volatility_ratio = volume_metrics['volatility_ratio']
        
        # Stealth accumulation criteria:
        # 1. Volume > 1.5x average (institutional interest)
        # 2. Volatility < 0.8x average (controlled movement)
        # 3. Price near range bottom (accumulation zone)
        
        for i in range(self.volume_window, len(df)):
            if pd.isna(volume_ratio.iloc[i]) or pd.isna(volatility_ratio.iloc[i]):
                continue
                
            vol_ratio = volume_ratio.iloc[i]
            vol_ratio_val = volatility_ratio.iloc[i]
            
            # Check stealth accumulation conditions
            high_volume = vol_ratio > 1.5
            low_volatility = vol_ratio_val < 0.8
            
            if high_volume and low_volatility:
                # Calculate stealth score
                stealth_score = self._calculate_stealth_score(
                    vol_ratio, vol_ratio_val, df, i
                )
                
                if stealth_score > self.stealth_threshold:
                    cluster = VolumeCluster(
                        timestamp=df.index[i],
                        price_level=float(df['close'].iloc[i]),
                        volume_ratio=float(vol_ratio),
                        volatility_ratio=float(vol_ratio_val),
                        cluster_strength=stealth_score,
                        signal_type=AccumulationSignal.STEALTH_ACCUMULATION,
                        confidence=min(stealth_score, 1.0)
                    )
                    stealth_signals.append(cluster)
        
        return stealth_signals
    
    def _calculate_stealth_score(self, volume_ratio: float, volatility_ratio: float, 
                                df: pd.DataFrame, index: int) -> float:
        """Calculate stealth accumulation score (0.0 to 1.0)"""
        # Base score from volume/volatility ratio
        volume_score = min(volume_ratio / 3.0, 1.0)  # Cap at 3x volume
        volatility_score = max(0, 1.0 - volatility_ratio)  # Lower volatility = higher score
        
        # Range position score (prefer accumulation near lows)
        lookback = min(20, index)
        recent_high = df['high'].iloc[index-lookback:index+1].max()
        recent_low = df['low'].iloc[index-lookback:index+1].min()
        current_price = df['close'].iloc[index]
        
        if recent_high != recent_low:
            range_position = (current_price - recent_low) / (recent_high - recent_low)
            range_score = 1.0 - range_position  # Higher score for lower prices
        else:
            range_score = 0.5
        
        # Combine scores with weights
        stealth_score = (
            volume_score * 0.4 +      # 40% volume importance
            volatility_score * 0.4 +  # 40% volatility importance
            range_score * 0.2         # 20% range position importance
        )
        
        return min(stealth_score, 1.0)
    
    def _analyze_volume_clusters(self, df: pd.DataFrame, volume_metrics: Dict) -> List[VolumeCluster]:
        """🎯 Analyze volume clusters for block trade detection"""
        clusters = []
        volume_ratio = volume_metrics['volume_ratio']
        
        # Find significant volume spikes (potential block trades)
        for i in range(self.volume_window, len(df)):
            if pd.isna(volume_ratio.iloc[i]):
                continue
                
            vol_ratio = volume_ratio.iloc[i]
            
            # Block trade criteria: Volume > 2.5x average
            if vol_ratio > 2.5:
                # Check for clustering with nearby volume spikes
                cluster_strength = self._calculate_cluster_strength(df, i, volume_ratio)
                
                if cluster_strength > 0.6:
                    cluster = VolumeCluster(
                        timestamp=df.index[i],
                        price_level=float(df['close'].iloc[i]),
                        volume_ratio=float(vol_ratio),
                        volatility_ratio=float(volume_metrics['volatility_ratio'].iloc[i]),
                        cluster_strength=cluster_strength,
                        signal_type=AccumulationSignal.BLOCK_TRADE_CLUSTER,
                        confidence=min(cluster_strength, 1.0)
                    )
                    clusters.append(cluster)
        
        return clusters
    
    def _calculate_cluster_strength(self, df: pd.DataFrame, index: int, volume_ratio: pd.Series) -> float:
        """Calculate strength of volume cluster"""
        # Look for nearby volume spikes within price range
        lookback = 5
        lookahead = 5
        
        start_idx = max(0, index - lookback)
        end_idx = min(len(df), index + lookahead + 1)
        
        current_price = df['close'].iloc[index]
        price_tolerance = current_price * self.cluster_distance  # 0.1% price range
        
        cluster_volume = 0
        cluster_count = 0
        
        for i in range(start_idx, end_idx):
            if i == index:
                continue
                
            price_diff = abs(df['close'].iloc[i] - current_price)
            if price_diff <= price_tolerance and volume_ratio.iloc[i] > 1.5:
                cluster_volume += volume_ratio.iloc[i]
                cluster_count += 1
        
        # Strength based on clustered volume and count
        if cluster_count > 0:
            avg_cluster_volume = cluster_volume / cluster_count
            strength = min((avg_cluster_volume + cluster_count) / 10.0, 1.0)
        else:
            strength = volume_ratio.iloc[index] / 5.0  # Solo spike strength
        
        return min(strength, 1.0)
    
    def _calculate_cvd_analysis(self, df: pd.DataFrame) -> Dict:
        """🎯 Calculate Cumulative Volume Delta analysis"""
        # Simplified CVD calculation (buy volume - sell volume)
        # In real implementation, you'd need tick data for true CVD
        
        # Proxy: Use price movement direction with volume
        price_change = df['close'].diff()
        volume = df['tick_volume'] if 'tick_volume' in df.columns else pd.Series([1] * len(df))
        
        # Positive CVD when price goes up, negative when down
        cvd_delta = np.where(price_change > 0, volume, -volume)
        cvd_cumulative = pd.Series(cvd_delta).cumsum()
        
        # CVD trend analysis
        cvd_ma = cvd_cumulative.rolling(window=20).mean()
        cvd_trend = 'bullish' if cvd_cumulative.iloc[-1] > cvd_ma.iloc[-1] else 'bearish'
        
        # CVD divergence detection
        price_trend = 'bullish' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'bearish'
        cvd_divergence = cvd_trend != price_trend
        
        return {
            'cvd_current': float(cvd_cumulative.iloc[-1]),
            'cvd_trend': cvd_trend,
            'cvd_divergence': cvd_divergence,
            'cvd_strength': abs(cvd_cumulative.iloc[-1] - cvd_ma.iloc[-1]) / cvd_ma.std()
        }
    
    def _calculate_vwap_analysis(self, df: pd.DataFrame) -> Dict:
        """🎯 Calculate VWAP analysis for institutional flow"""
        # Calculate VWAP
        volume = df['tick_volume'] if 'tick_volume' in df.columns else pd.Series([1] * len(df))
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        # VWAP deviation
        current_price = df['close'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        vwap_deviation = (current_price - current_vwap) / current_vwap
        
        # VWAP trend
        vwap_slope = (vwap.iloc[-1] - vwap.iloc[-10]) / 10
        vwap_trend = 'bullish' if vwap_slope > 0 else 'bearish'
        
        return {
            'vwap_current': float(current_vwap),
            'vwap_deviation': float(vwap_deviation),
            'vwap_trend': vwap_trend,
            'price_above_vwap': current_price > current_vwap
        }
    
    def _detect_absorption_zones(self, df: pd.DataFrame, volume_metrics: Dict) -> List[Dict]:
        """🎯 Detect price absorption zones (smart money defending levels)"""
        absorption_zones = []
        
        # Look for high volume with small price movement (absorption)
        volume_ratio = volume_metrics['volume_ratio']
        
        for i in range(self.volume_window, len(df)):
            if pd.isna(volume_ratio.iloc[i]):
                continue
            
            # High volume condition
            if volume_ratio.iloc[i] > 2.0:
                # Check price movement (should be small for absorption)
                price_range = df['high'].iloc[i] - df['low'].iloc[i]
                avg_range = volume_metrics['atr'].iloc[i]
                
                if price_range < avg_range * 0.7:  # Small range despite high volume
                    # Check for wick rejection
                    body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
                    upper_wick = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
                    lower_wick = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
                    
                    # Absorption with rejection
                    if upper_wick > body_size * 1.5 or lower_wick > body_size * 1.5:
                        absorption_zones.append({
                            'timestamp': df.index[i],
                            'price_level': float(df['close'].iloc[i]),
                            'volume_ratio': float(volume_ratio.iloc[i]),
                            'absorption_type': 'upper_rejection' if upper_wick > lower_wick else 'lower_rejection',
                            'strength': min(volume_ratio.iloc[i] / 3.0, 1.0)
                        })
        
        return absorption_zones
    
    def _generate_flow_summary(self, stealth_signals: List[VolumeCluster], 
                              volume_clusters: List[VolumeCluster],
                              cvd_analysis: Dict, vwap_analysis: Dict,
                              absorption_zones: List[Dict]) -> Optional[InstitutionalFlow]:
        """Generate overall institutional flow summary"""
        if not any([stealth_signals, volume_clusters, absorption_zones]):
            return None
        
        # Determine flow type based on signals
        accumulation_signals = len(stealth_signals) + len([z for z in absorption_zones if z['absorption_type'] == 'lower_rejection'])
        distribution_signals = len([z for z in absorption_zones if z['absorption_type'] == 'upper_rejection'])
        
        if accumulation_signals > distribution_signals:
            flow_type = 'accumulation'
        elif distribution_signals > accumulation_signals:
            flow_type = 'distribution'
        else:
            flow_type = 'neutral'
        
        # Calculate overall strength
        total_signals = len(stealth_signals) + len(volume_clusters) + len(absorption_zones)
        avg_strength = sum([s.cluster_strength for s in stealth_signals + volume_clusters]) / max(total_signals, 1)
        
        # Calculate stealth score (how hidden the flow is)
        stealth_score = len(stealth_signals) / max(total_signals, 1)
        
        return InstitutionalFlow(
            timestamp=datetime.now(),
            flow_type=flow_type,
            strength=min(avg_strength, 1.0),
            price_zone=(0.0, 0.0),  # Would be calculated from actual signals
            volume_profile={},
            cvd_delta=cvd_analysis.get('cvd_current', 0.0),
            vwap_deviation=vwap_analysis.get('vwap_deviation', 0.0),
            stealth_score=stealth_score
        )
    
    def _empty_flow_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'institutional_flow': None,
            'stealth_accumulation': [],
            'volume_clusters': [],
            'cvd_analysis': {},
            'vwap_analysis': {},
            'absorption_zones': [],
            'flow_strength': 0.0,
            'flow_type': 'neutral'
        }