# 🕵️ BLOB AI - Market Manipulation Detection System
# Advanced Pattern Recognition for Institutional Behavior

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import math
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ManipulationType(Enum):
    """Types of market manipulation"""
    STOP_HUNT = "stop_hunt"                    # Hunting retail stops
    LIQUIDITY_SWEEP = "liquidity_sweep"        # Sweeping liquidity pools
    ICEBERG_ORDER = "iceberg_order"            # Hidden large orders
    SPOOFING = "spoofing"                      # Fake order placement
    MOMENTUM_IGNITION = "momentum_ignition"    # Creating false momentum
    WASH_TRADING = "wash_trading"              # Self-trading
    LAYERING = "layering"                      # Order book manipulation
    QUOTE_STUFFING = "quote_stuffing"          # High-frequency noise
    BEAR_RAID = "bear_raid"                    # Coordinated selling
    PUMP_AND_DUMP = "pump_and_dump"            # Artificial price inflation
    FRONT_RUNNING = "front_running"            # Trading ahead of orders
    DARK_POOL_ABUSE = "dark_pool_abuse"        # Dark pool manipulation

class InstitutionalPlayer(Enum):
    """Types of institutional players"""
    CENTRAL_BANK = "central_bank"
    COMMERCIAL_BANK = "commercial_bank"
    INVESTMENT_BANK = "investment_bank"
    HEDGE_FUND = "hedge_fund"
    PENSION_FUND = "pension_fund"
    SOVEREIGN_FUND = "sovereign_fund"
    PROP_TRADING_FIRM = "prop_trading_firm"
    MARKET_MAKER = "market_maker"
    HIGH_FREQUENCY_TRADER = "high_frequency_trader"
    RETAIL_AGGREGATOR = "retail_aggregator"

class ManipulationSeverity(Enum):
    """Severity levels of manipulation"""
    LOW = "low"          # Minor manipulation, limited impact
    MODERATE = "moderate" # Noticeable manipulation, moderate impact
    HIGH = "high"        # Strong manipulation, significant impact
    EXTREME = "extreme"   # Massive manipulation, market-moving

@dataclass
class ManipulationSignal:
    """Detected manipulation signal"""
    timestamp: datetime
    symbol: str
    manipulation_type: ManipulationType
    severity: ManipulationSeverity
    confidence: float  # 0-1
    price_level: float
    volume_anomaly: float
    institutional_player: Optional[InstitutionalPlayer]
    description: str
    exploitation_opportunity: Optional[str]
    risk_level: str
    duration_estimate: int  # minutes
    profit_potential: float  # pips

@dataclass
class LiquidityPool:
    """Liquidity pool identification"""
    price_level: float
    volume: float
    order_count: int
    pool_type: str  # 'buy_stops', 'sell_stops', 'support', 'resistance'
    strength: float  # 0-1
    vulnerability: float  # 0-1 (how likely to be swept)
    last_updated: datetime

@dataclass
class OrderFlowAnomaly:
    """Order flow anomaly detection"""
    timestamp: datetime
    anomaly_type: str
    magnitude: float
    buy_volume: float
    sell_volume: float
    price_impact: float
    institutional_signature: bool
    confidence: float

class ManipulationDetector:
    """🕵️ Advanced Market Manipulation Detection System
    
    Features:
    - Real-time manipulation pattern detection
    - Institutional footprint analysis
    - Liquidity pool mapping and sweep prediction
    - Order flow anomaly detection
    - Stop hunt identification
    - Spoofing and layering detection
    - Dark pool activity analysis
    - Exploitation opportunity identification
    """
    
    def __init__(self, lookback_periods: int = 1000):
        self.lookback_periods = lookback_periods
        
        # Data storage
        self.price_data = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.volume_data = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.order_flow_data = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.tick_data = defaultdict(lambda: deque(maxlen=10000))
        
        # Detection parameters
        self.volume_spike_threshold = 3.0  # 3x normal volume
        self.price_anomaly_threshold = 2.5  # 2.5 standard deviations
        self.order_flow_imbalance_threshold = 0.7  # 70% imbalance
        self.liquidity_sweep_threshold = 0.8  # 80% of liquidity consumed
        
        # Pattern tracking
        self.detected_manipulations = deque(maxlen=100)
        self.liquidity_pools = defaultdict(list)
        self.institutional_signatures = {}
        self.manipulation_history = defaultdict(list)
        
        # Statistical models
        self.volume_models = {}
        self.price_models = {}
        self.order_flow_models = {}
        
        logger.info("🕵️ Manipulation Detector initialized")
    
    def detect_manipulation(self, symbol: str, market_data: Dict, 
                          order_book: Dict = None, tick_data: List = None) -> List[ManipulationSignal]:
        """🎯 Main manipulation detection function"""
        signals = []
        
        try:
            # Update data
            self._update_market_data(symbol, market_data, order_book, tick_data)
            
            # Detect different types of manipulation
            signals.extend(self._detect_stop_hunts(symbol, market_data))
            signals.extend(self._detect_liquidity_sweeps(symbol, market_data, order_book))
            signals.extend(self._detect_iceberg_orders(symbol, order_book, tick_data))
            signals.extend(self._detect_spoofing(symbol, order_book))
            signals.extend(self._detect_momentum_ignition(symbol, market_data, tick_data))
            signals.extend(self._detect_layering(symbol, order_book))
            signals.extend(self._detect_quote_stuffing(symbol, tick_data))
            signals.extend(self._detect_wash_trading(symbol, tick_data))
            signals.extend(self._detect_front_running(symbol, market_data, tick_data))
            signals.extend(self._detect_dark_pool_activity(symbol, market_data))
            
            # Filter and rank signals
            signals = self._filter_and_rank_signals(signals)
            
            # Store detected manipulations
            for signal in signals:
                self.detected_manipulations.append(signal)
                self.manipulation_history[symbol].append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error detecting manipulation for {symbol}: {e}")
            return []
    
    def _detect_stop_hunts(self, symbol: str, market_data: Dict) -> List[ManipulationSignal]:
        """🎯 Detect stop hunting patterns"""
        signals = []
        
        if len(self.price_data[symbol]) < 50:
            return signals
        
        current_price = market_data.get('close', 0)
        prices = list(self.price_data[symbol])
        volumes = list(self.volume_data[symbol])
        
        # Identify potential stop levels
        support_levels = self._identify_support_resistance(prices, 'support')
        resistance_levels = self._identify_support_resistance(prices, 'resistance')
        
        # Check for stop hunt patterns
        for level in support_levels + resistance_levels:
            # Look for quick spike through level with high volume, then reversal
            if self._is_stop_hunt_pattern(prices, volumes, level, current_price):
                severity = self._calculate_stop_hunt_severity(prices, volumes, level)
                
                signal = ManipulationSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    manipulation_type=ManipulationType.STOP_HUNT,
                    severity=severity,
                    confidence=0.8,
                    price_level=level,
                    volume_anomaly=self._calculate_volume_anomaly(volumes),
                    institutional_player=InstitutionalPlayer.MARKET_MAKER,
                    description=f"Stop hunt detected at {level:.5f} level",
                    exploitation_opportunity="Trade reversal after stop sweep",
                    risk_level="moderate",
                    duration_estimate=15,
                    profit_potential=30
                )
                signals.append(signal)
        
        return signals
    
    def _detect_liquidity_sweeps(self, symbol: str, market_data: Dict, 
                               order_book: Dict = None) -> List[ManipulationSignal]:
        """💧 Detect liquidity sweep patterns"""
        signals = []
        
        if not order_book or len(self.volume_data[symbol]) < 20:
            return signals
        
        # Analyze order book for liquidity pools
        liquidity_pools = self._identify_liquidity_pools(order_book)
        
        # Check for rapid consumption of liquidity
        for pool in liquidity_pools:
            if pool.vulnerability > 0.7:  # High vulnerability
                current_volume = market_data.get('volume', 0)
                avg_volume = np.mean(list(self.volume_data[symbol])[-20:])
                
                if current_volume > avg_volume * 2:  # Volume spike
                    signal = ManipulationSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        manipulation_type=ManipulationType.LIQUIDITY_SWEEP,
                        severity=ManipulationSeverity.HIGH,
                        confidence=0.75,
                        price_level=pool.price_level,
                        volume_anomaly=current_volume / avg_volume,
                        institutional_player=InstitutionalPlayer.HEDGE_FUND,
                        description=f"Liquidity sweep at {pool.price_level:.5f}",
                        exploitation_opportunity="Follow institutional flow direction",
                        risk_level="high",
                        duration_estimate=30,
                        profit_potential=50
                    )
                    signals.append(signal)
        
        return signals
    
    def _detect_iceberg_orders(self, symbol: str, order_book: Dict = None, 
                             tick_data: List = None) -> List[ManipulationSignal]:
        """🧊 Detect iceberg order patterns"""
        signals = []
        
        if not order_book or not tick_data:
            return signals
        
        # Look for repeated small orders at same price level
        price_levels = defaultdict(list)
        
        for tick in tick_data[-100:]:  # Last 100 ticks
            price = tick.get('price', 0)
            size = tick.get('size', 0)
            price_levels[round(price, 5)].append(size)
        
        # Detect iceberg pattern
        for price, sizes in price_levels.items():
            if len(sizes) > 10:  # Many orders at same level
                size_consistency = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
                
                if size_consistency < 0.2:  # Very consistent sizes
                    signal = ManipulationSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        manipulation_type=ManipulationType.ICEBERG_ORDER,
                        severity=ManipulationSeverity.MODERATE,
                        confidence=0.7,
                        price_level=price,
                        volume_anomaly=len(sizes),
                        institutional_player=InstitutionalPlayer.INVESTMENT_BANK,
                        description=f"Iceberg order detected at {price:.5f}",
                        exploitation_opportunity="Anticipate large order completion",
                        risk_level="low",
                        duration_estimate=60,
                        profit_potential=20
                    )
                    signals.append(signal)
        
        return signals
    
    def _detect_spoofing(self, symbol: str, order_book: Dict = None) -> List[ManipulationSignal]:
        """👻 Detect spoofing patterns"""
        signals = []
        
        if not order_book:
            return signals
        
        # Track order book changes
        if not hasattr(self, 'previous_order_book'):
            self.previous_order_book = {}
        
        prev_book = self.previous_order_book.get(symbol, {})
        
        # Look for large orders that appear and disappear quickly
        if prev_book:
            # Check bids
            for price, size in order_book.get('bids', []):
                prev_size = dict(prev_book.get('bids', [])).get(price, 0)
                if size > prev_size * 5 and size > 1.0:  # Large new order
                    # This would be flagged for monitoring
                    pass
        
        self.previous_order_book[symbol] = order_book
        
        return signals
    
    def _detect_momentum_ignition(self, symbol: str, market_data: Dict, 
                                tick_data: List = None) -> List[ManipulationSignal]:
        """🚀 Detect momentum ignition patterns"""
        signals = []
        
        if not tick_data or len(tick_data) < 50:
            return signals
        
        # Analyze recent price movements
        recent_prices = [tick.get('price', 0) for tick in tick_data[-50:]]
        recent_volumes = [tick.get('size', 0) for tick in tick_data[-50:]]
        
        # Look for sudden price acceleration with volume spike
        price_changes = np.diff(recent_prices)
        volume_spike = np.max(recent_volumes[-10:]) / np.mean(recent_volumes[:-10]) if len(recent_volumes) > 10 else 1
        
        # Detect momentum ignition
        if volume_spike > 3 and np.std(price_changes[-10:]) > np.std(price_changes[:-10]) * 2:
            signal = ManipulationSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                manipulation_type=ManipulationType.MOMENTUM_IGNITION,
                severity=ManipulationSeverity.HIGH,
                confidence=0.8,
                price_level=recent_prices[-1],
                volume_anomaly=volume_spike,
                institutional_player=InstitutionalPlayer.PROP_TRADING_FIRM,
                description="Momentum ignition detected",
                exploitation_opportunity="Ride the artificial momentum",
                risk_level="high",
                duration_estimate=10,
                profit_potential=40
            )
            signals.append(signal)
        
        return signals
    
    def _detect_layering(self, symbol: str, order_book: Dict = None) -> List[ManipulationSignal]:
        """📚 Detect layering manipulation"""
        signals = []
        
        if not order_book:
            return signals
        
        # Analyze order book depth and distribution
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        # Look for unusual order clustering
        if len(bids) > 5 and len(asks) > 5:
            bid_sizes = [size for price, size in bids[:10]]
            ask_sizes = [size for price, size in asks[:10]]
            
            # Check for layering pattern (many similar-sized orders)
            bid_consistency = 1 - (np.std(bid_sizes) / np.mean(bid_sizes)) if np.mean(bid_sizes) > 0 else 0
            ask_consistency = 1 - (np.std(ask_sizes) / np.mean(ask_sizes)) if np.mean(ask_sizes) > 0 else 0
            
            if bid_consistency > 0.8 or ask_consistency > 0.8:
                signal = ManipulationSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    manipulation_type=ManipulationType.LAYERING,
                    severity=ManipulationSeverity.MODERATE,
                    confidence=0.6,
                    price_level=market_data.get('close', 0),
                    volume_anomaly=max(bid_consistency, ask_consistency),
                    institutional_player=InstitutionalPlayer.HIGH_FREQUENCY_TRADER,
                    description="Layering pattern detected in order book",
                    exploitation_opportunity="Anticipate true direction",
                    risk_level="moderate",
                    duration_estimate=5,
                    profit_potential=15
                )
                signals.append(signal)
        
        return signals
    
    def _detect_quote_stuffing(self, symbol: str, tick_data: List = None) -> List[ManipulationSignal]:
        """📊 Detect quote stuffing"""
        signals = []
        
        if not tick_data or len(tick_data) < 100:
            return signals
        
        # Analyze tick frequency
        recent_ticks = tick_data[-60:]  # Last 60 ticks
        timestamps = [tick.get('timestamp', datetime.now()) for tick in recent_ticks]
        
        if len(timestamps) > 1:
            # Calculate tick frequency
            time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                         for i in range(1, len(timestamps))]
            
            avg_interval = np.mean(time_diffs)
            min_interval = np.min(time_diffs)
            
            # Detect abnormally high frequency
            if min_interval < 0.1 and avg_interval < 0.5:  # Very fast ticks
                signal = ManipulationSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    manipulation_type=ManipulationType.QUOTE_STUFFING,
                    severity=ManipulationSeverity.LOW,
                    confidence=0.7,
                    price_level=market_data.get('close', 0),
                    volume_anomaly=1 / min_interval,
                    institutional_player=InstitutionalPlayer.HIGH_FREQUENCY_TRADER,
                    description="Quote stuffing detected",
                    exploitation_opportunity="Wait for normal market conditions",
                    risk_level="low",
                    duration_estimate=2,
                    profit_potential=5
                )
                signals.append(signal)
        
        return signals
    
    def _detect_wash_trading(self, symbol: str, tick_data: List = None) -> List[ManipulationSignal]:
        """🔄 Detect wash trading patterns"""
        signals = []
        
        if not tick_data or len(tick_data) < 50:
            return signals
        
        # Look for back-and-forth trading at same prices
        price_volume_pairs = [(tick.get('price', 0), tick.get('size', 0)) for tick in tick_data[-50:]]
        
        # Group by price levels
        price_groups = defaultdict(list)
        for price, volume in price_volume_pairs:
            price_groups[round(price, 5)].append(volume)
        
        # Detect wash trading pattern
        for price, volumes in price_groups.items():
            if len(volumes) > 10:  # Many trades at same price
                volume_pattern = np.array(volumes)
                # Look for alternating pattern
                if len(volume_pattern) > 4:
                    alternating_score = np.corrcoef(volume_pattern[:-1], volume_pattern[1:])[0, 1]
                    
                    if alternating_score < -0.5:  # Strong negative correlation (alternating)
                        signal = ManipulationSignal(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            manipulation_type=ManipulationType.WASH_TRADING,
                            severity=ManipulationSeverity.MODERATE,
                            confidence=0.6,
                            price_level=price,
                            volume_anomaly=len(volumes),
                            institutional_player=InstitutionalPlayer.PROP_TRADING_FIRM,
                            description=f"Wash trading detected at {price:.5f}",
                            exploitation_opportunity="Ignore artificial volume",
                            risk_level="low",
                            duration_estimate=20,
                            profit_potential=10
                        )
                        signals.append(signal)
        
        return signals
    
    def _detect_front_running(self, symbol: str, market_data: Dict, 
                            tick_data: List = None) -> List[ManipulationSignal]:
        """🏃 Detect front running patterns"""
        signals = []
        
        # This would require order flow data and institutional order detection
        # Simplified implementation
        
        if not tick_data or len(tick_data) < 30:
            return signals
        
        # Look for unusual price movements before large orders
        recent_volumes = [tick.get('size', 0) for tick in tick_data[-30:]]
        recent_prices = [tick.get('price', 0) for tick in tick_data[-30:]]
        
        # Detect large order followed by price movement
        max_volume_idx = np.argmax(recent_volumes)
        if max_volume_idx > 5:  # Not at the beginning
            pre_movement = np.std(recent_prices[:max_volume_idx])
            post_movement = np.std(recent_prices[max_volume_idx:])
            
            if pre_movement > post_movement * 2:  # Movement before large order
                signal = ManipulationSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    manipulation_type=ManipulationType.FRONT_RUNNING,
                    severity=ManipulationSeverity.HIGH,
                    confidence=0.5,
                    price_level=recent_prices[max_volume_idx],
                    volume_anomaly=recent_volumes[max_volume_idx] / np.mean(recent_volumes),
                    institutional_player=InstitutionalPlayer.HIGH_FREQUENCY_TRADER,
                    description="Potential front running detected",
                    exploitation_opportunity="Follow the institutional order",
                    risk_level="moderate",
                    duration_estimate=5,
                    profit_potential=25
                )
                signals.append(signal)
        
        return signals
    
    def _detect_dark_pool_activity(self, symbol: str, market_data: Dict) -> List[ManipulationSignal]:
        """🌑 Detect dark pool activity"""
        signals = []
        
        # Look for price movements without corresponding visible volume
        if len(self.price_data[symbol]) < 20 or len(self.volume_data[symbol]) < 20:
            return signals
        
        recent_prices = list(self.price_data[symbol])[-20:]
        recent_volumes = list(self.volume_data[symbol])[-20:]
        
        # Calculate price movement vs volume correlation
        price_changes = np.abs(np.diff(recent_prices))
        volume_changes = np.diff(recent_volumes)
        
        if len(price_changes) > 5 and len(volume_changes) > 5:
            correlation = np.corrcoef(price_changes, volume_changes[1:])[0, 1]
            
            # Low correlation suggests dark pool activity
            if correlation < 0.3 and np.mean(price_changes[-5:]) > np.mean(price_changes[:-5]):
                signal = ManipulationSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    manipulation_type=ManipulationType.DARK_POOL_ABUSE,
                    severity=ManipulationSeverity.MODERATE,
                    confidence=0.6,
                    price_level=market_data.get('close', 0),
                    volume_anomaly=1 - correlation,
                    institutional_player=InstitutionalPlayer.INVESTMENT_BANK,
                    description="Dark pool activity detected",
                    exploitation_opportunity="Follow hidden institutional flow",
                    risk_level="moderate",
                    duration_estimate=45,
                    profit_potential=35
                )
                signals.append(signal)
        
        return signals
    
    def _update_market_data(self, symbol: str, market_data: Dict, 
                          order_book: Dict = None, tick_data: List = None):
        """Update internal market data"""
        # Update price data
        if 'close' in market_data:
            self.price_data[symbol].append(market_data['close'])
        
        # Update volume data
        if 'volume' in market_data:
            self.volume_data[symbol].append(market_data['volume'])
        
        # Update tick data
        if tick_data:
            for tick in tick_data:
                self.tick_data[symbol].append(tick)
    
    def _identify_support_resistance(self, prices: List[float], level_type: str) -> List[float]:
        """Identify support and resistance levels"""
        if len(prices) < 20:
            return []
        
        prices_array = np.array(prices)
        levels = []
        
        # Find local extremes
        if level_type == 'support':
            # Find local minima
            for i in range(5, len(prices) - 5):
                if all(prices[i] <= prices[i-j] for j in range(1, 6)) and \
                   all(prices[i] <= prices[i+j] for j in range(1, 6)):
                    levels.append(prices[i])
        else:
            # Find local maxima
            for i in range(5, len(prices) - 5):
                if all(prices[i] >= prices[i-j] for j in range(1, 6)) and \
                   all(prices[i] >= prices[i+j] for j in range(1, 6)):
                    levels.append(prices[i])
        
        return levels
    
    def _is_stop_hunt_pattern(self, prices: List[float], volumes: List[float], 
                            level: float, current_price: float) -> bool:
        """Check if pattern matches stop hunt"""
        if len(prices) < 10 or len(volumes) < 10:
            return False
        
        # Look for spike through level with high volume, then reversal
        recent_prices = prices[-10:]
        recent_volumes = volumes[-10:]
        
        # Check if price spiked through level
        level_breached = any(abs(price - level) < 0.0001 for price in recent_prices)
        
        # Check for volume spike
        avg_volume = np.mean(volumes[:-10]) if len(volumes) > 10 else np.mean(volumes)
        max_recent_volume = max(recent_volumes)
        volume_spike = max_recent_volume > avg_volume * 2
        
        # Check for reversal
        price_reversed = abs(current_price - level) > 0.0002
        
        return level_breached and volume_spike and price_reversed
    
    def _calculate_stop_hunt_severity(self, prices: List[float], volumes: List[float], 
                                    level: float) -> ManipulationSeverity:
        """Calculate severity of stop hunt"""
        volume_spike_ratio = max(volumes[-5:]) / np.mean(volumes[:-5]) if len(volumes) > 5 else 1
        
        if volume_spike_ratio > 5:
            return ManipulationSeverity.EXTREME
        elif volume_spike_ratio > 3:
            return ManipulationSeverity.HIGH
        elif volume_spike_ratio > 2:
            return ManipulationSeverity.MODERATE
        else:
            return ManipulationSeverity.LOW
    
    def _calculate_volume_anomaly(self, volumes: List[float]) -> float:
        """Calculate volume anomaly score"""
        if len(volumes) < 10:
            return 1.0
        
        recent_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-1])
        
        return recent_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _identify_liquidity_pools(self, order_book: Dict) -> List[LiquidityPool]:
        """Identify liquidity pools in order book"""
        pools = []
        
        if not order_book:
            return pools
        
        # Analyze bid side
        bids = order_book.get('bids', [])
        for price, size in bids[:10]:  # Top 10 levels
            if size > 1.0:  # Significant size
                pool = LiquidityPool(
                    price_level=price,
                    volume=size,
                    order_count=1,  # Simplified
                    pool_type='buy_stops',
                    strength=min(size / 10.0, 1.0),
                    vulnerability=0.5,  # Default
                    last_updated=datetime.now()
                )
                pools.append(pool)
        
        # Analyze ask side
        asks = order_book.get('asks', [])
        for price, size in asks[:10]:  # Top 10 levels
            if size > 1.0:  # Significant size
                pool = LiquidityPool(
                    price_level=price,
                    volume=size,
                    order_count=1,  # Simplified
                    pool_type='sell_stops',
                    strength=min(size / 10.0, 1.0),
                    vulnerability=0.5,  # Default
                    last_updated=datetime.now()
                )
                pools.append(pool)
        
        return pools
    
    def _filter_and_rank_signals(self, signals: List[ManipulationSignal]) -> List[ManipulationSignal]:
        """Filter and rank manipulation signals by importance"""
        if not signals:
            return signals
        
        # Sort by confidence and severity
        severity_weights = {
            ManipulationSeverity.EXTREME: 4,
            ManipulationSeverity.HIGH: 3,
            ManipulationSeverity.MODERATE: 2,
            ManipulationSeverity.LOW: 1
        }
        
        def signal_score(signal):
            return signal.confidence * severity_weights.get(signal.severity, 1)
        
        sorted_signals = sorted(signals, key=signal_score, reverse=True)
        
        # Return top 5 signals
        return sorted_signals[:5]
    
    def get_exploitation_strategy(self, signal: ManipulationSignal) -> Dict:
        """🎯 Get exploitation strategy for detected manipulation"""
        strategies = {
            ManipulationType.STOP_HUNT: {
                'entry_strategy': 'Enter after stop sweep completion',
                'direction': 'Opposite to sweep direction',
                'stop_loss': '20 pips beyond sweep level',
                'take_profit': '2:1 risk reward',
                'position_size': 'Conservative (0.5% risk)',
                'timing': 'Immediate after reversal confirmation'
            },
            ManipulationType.LIQUIDITY_SWEEP: {
                'entry_strategy': 'Follow institutional flow',
                'direction': 'Same as sweep direction',
                'stop_loss': '30 pips beyond entry',
                'take_profit': '3:1 risk reward',
                'position_size': 'Moderate (1% risk)',
                'timing': 'After sweep completion'
            },
            ManipulationType.MOMENTUM_IGNITION: {
                'entry_strategy': 'Ride artificial momentum',
                'direction': 'Same as ignition direction',
                'stop_loss': '15 pips beyond entry',
                'take_profit': '1.5:1 risk reward',
                'position_size': 'Aggressive (1.5% risk)',
                'timing': 'Quick entry and exit'
            }
        }
        
        base_strategy = strategies.get(signal.manipulation_type, {
            'entry_strategy': 'Wait for confirmation',
            'direction': 'Trend following',
            'stop_loss': '25 pips',
            'take_profit': '2:1 risk reward',
            'position_size': 'Conservative (0.5% risk)',
            'timing': 'Next session'
        })
        
        # Add signal-specific details
        strategy = base_strategy.copy()
        strategy.update({
            'signal_confidence': signal.confidence,
            'estimated_duration': signal.duration_estimate,
            'profit_potential': signal.profit_potential,
            'risk_level': signal.risk_level,
            'institutional_player': signal.institutional_player.value if signal.institutional_player else 'unknown'
        })
        
        return strategy
    
    def get_manipulation_summary(self, symbol: str = None) -> Dict:
        """📊 Get summary of detected manipulations"""
        if symbol:
            signals = [s for s in self.detected_manipulations if s.symbol == symbol]
        else:
            signals = list(self.detected_manipulations)
        
        if not signals:
            return {'total_signals': 0, 'summary': 'No manipulations detected'}
        
        # Count by type
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        player_counts = defaultdict(int)
        
        for signal in signals:
            type_counts[signal.manipulation_type.value] += 1
            severity_counts[signal.severity.value] += 1
            if signal.institutional_player:
                player_counts[signal.institutional_player.value] += 1
        
        return {
            'total_signals': len(signals),
            'by_type': dict(type_counts),
            'by_severity': dict(severity_counts),
            'by_player': dict(player_counts),
            'avg_confidence': np.mean([s.confidence for s in signals]),
            'total_profit_potential': sum(s.profit_potential for s in signals)
        }

def create_manipulation_detector(lookback_periods: int = 1000) -> ManipulationDetector:
    """Factory function to create manipulation detector"""
    return ManipulationDetector(lookback_periods)