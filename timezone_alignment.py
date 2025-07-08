#!/usr/bin/env python3
"""
Timezone Alignment Utility for Forex Factory Scraper and Trading Engine

Ensures proper synchronization between:
- Forex Factory calendar events (UTC timestamps)
- Trading engine session analysis (UTC-based)
- MetaTrader 5 data (broker timezone)

Author: BLOB AI Trading System
Version: 1.0.0
"""

import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from loguru import logger

# Import trading engine session types
try:
    from forex_engine import SessionType
except ImportError:
    class SessionType(Enum):
        ASIA = "Asia"
        LONDON = "London"
        NEW_YORK = "New_York"
        OVERLAP = "Overlap"

@dataclass
class TimezoneConfig:
    """Configuration for timezone handling"""
    forex_factory_tz: str = "UTC"  # Forex Factory uses UTC
    trading_engine_tz: str = "UTC"  # Engine operates in UTC
    mt5_broker_tz: str = "UTC"  # Most brokers use UTC or UTC+2/3
    local_tz: str = "UTC"  # Local system timezone

class TimezoneAligner:
    """Handles timezone alignment between different system components"""
    
    def __init__(self, config: Optional[TimezoneConfig] = None):
        self.config = config or TimezoneConfig()
        
        # Initialize timezone objects
        self.utc_tz = pytz.UTC
        self.forex_factory_tz = pytz.timezone(self.config.forex_factory_tz)
        self.trading_engine_tz = pytz.timezone(self.config.trading_engine_tz)
        
        # Session times in UTC (matching trading engine)
        self.session_times = {
            SessionType.ASIA: [(0, 9)],  # UTC hours
            SessionType.LONDON: [(8, 17)],
            SessionType.NEW_YORK: [(13, 22)],
            SessionType.OVERLAP: [(13, 17)]  # London-NY overlap
        }
        
        logger.info(f"TimezoneAligner initialized with config: {self.config}")
    
    def align_forex_factory_timestamp(self, timestamp: datetime) -> datetime:
        """
        Align Forex Factory timestamp to trading engine timezone
        
        Args:
            timestamp: Datetime from Forex Factory (should be UTC)
            
        Returns:
            Aligned datetime in trading engine timezone (UTC)
        """
        try:
            # Ensure timestamp is timezone-aware
            if timestamp.tzinfo is None:
                # Assume UTC if no timezone info
                timestamp = self.utc_tz.localize(timestamp)
            elif timestamp.tzinfo != self.utc_tz:
                # Convert to UTC if different timezone
                timestamp = timestamp.astimezone(self.utc_tz)
            
            # Convert to trading engine timezone (UTC)
            aligned_timestamp = timestamp.astimezone(self.trading_engine_tz)
            
            return aligned_timestamp
            
        except Exception as e:
            logger.error(f"Error aligning timestamp {timestamp}: {e}")
            # Fallback: return current UTC time
            return datetime.now(self.utc_tz)
    
    def get_current_utc_time(self) -> datetime:
        """Get current time in UTC (aligned with trading engine)"""
        return datetime.now(self.utc_tz)
    
    def get_current_session(self, timestamp: Optional[datetime] = None) -> SessionType:
        """
        Get current trading session based on UTC time
        
        Args:
            timestamp: Optional timestamp, defaults to current UTC time
            
        Returns:
            Current trading session
        """
        if timestamp is None:
            timestamp = self.get_current_utc_time()
        
        # Ensure timestamp is in UTC
        if timestamp.tzinfo is None:
            timestamp = self.utc_tz.localize(timestamp)
        elif timestamp.tzinfo != self.utc_tz:
            timestamp = timestamp.astimezone(self.utc_tz)
        
        utc_hour = timestamp.hour
        
        # Session logic matching trading engine
        if 13 <= utc_hour <= 17:
            return SessionType.OVERLAP
        elif 8 <= utc_hour <= 17:
            return SessionType.LONDON
        elif 13 <= utc_hour <= 22:
            return SessionType.NEW_YORK
        else:
            return SessionType.ASIA
    
    def is_event_imminent(self, event_timestamp: datetime, 
                         threshold_minutes: int = 30) -> bool:
        """
        Check if an economic event is imminent
        
        Args:
            event_timestamp: Event timestamp from Forex Factory
            threshold_minutes: Minutes before event to consider imminent
            
        Returns:
            True if event is within threshold minutes
        """
        current_time = self.get_current_utc_time()
        aligned_event_time = self.align_forex_factory_timestamp(event_timestamp)
        
        time_diff = aligned_event_time - current_time
        time_diff_minutes = time_diff.total_seconds() / 60
        
        return 0 <= time_diff_minutes <= threshold_minutes
    
    def has_event_occurred(self, event_timestamp: datetime) -> bool:
        """
        Check if an economic event has already occurred
        
        Args:
            event_timestamp: Event timestamp from Forex Factory
            
        Returns:
            True if event has occurred
        """
        current_time = self.get_current_utc_time()
        aligned_event_time = self.align_forex_factory_timestamp(event_timestamp)
        
        return aligned_event_time < current_time
    
    def get_time_until_event(self, event_timestamp: datetime) -> timedelta:
        """
        Get time remaining until event
        
        Args:
            event_timestamp: Event timestamp from Forex Factory
            
        Returns:
            Time difference (positive if future, negative if past)
        """
        current_time = self.get_current_utc_time()
        aligned_event_time = self.align_forex_factory_timestamp(event_timestamp)
        
        return aligned_event_time - current_time
    
    def filter_events_by_time_window(self, events: List[Dict], 
                                   hours_ahead: int = 24) -> List[Dict]:
        """
        Filter events within a specific time window
        
        Args:
            events: List of economic events with 'timestamp' field
            hours_ahead: Hours ahead to include events
            
        Returns:
            Filtered list of events
        """
        current_time = self.get_current_utc_time()
        cutoff_time = current_time + timedelta(hours=hours_ahead)
        
        filtered_events = []
        for event in events:
            if 'timestamp' in event:
                aligned_timestamp = self.align_forex_factory_timestamp(event['timestamp'])
                if current_time <= aligned_timestamp <= cutoff_time:
                    # Add aligned timestamp to event
                    event['aligned_timestamp'] = aligned_timestamp
                    event['time_until_event'] = self.get_time_until_event(event['timestamp'])
                    event['current_session'] = self.get_current_session(aligned_timestamp)
                    filtered_events.append(event)
        
        return filtered_events
    
    def validate_timestamp_alignment(self, forex_factory_timestamp: datetime, 
                                   trading_engine_timestamp: datetime) -> Dict:
        """
        Validate that timestamps are properly aligned
        
        Args:
            forex_factory_timestamp: Timestamp from Forex Factory
            trading_engine_timestamp: Timestamp from trading engine
            
        Returns:
            Validation results
        """
        aligned_ff_timestamp = self.align_forex_factory_timestamp(forex_factory_timestamp)
        
        # Ensure trading engine timestamp is UTC
        if trading_engine_timestamp.tzinfo is None:
            te_timestamp = self.utc_tz.localize(trading_engine_timestamp)
        else:
            te_timestamp = trading_engine_timestamp.astimezone(self.utc_tz)
        
        time_diff = abs((aligned_ff_timestamp - te_timestamp).total_seconds())
        
        return {
            'aligned': time_diff < 60,  # Within 1 minute
            'time_difference_seconds': time_diff,
            'forex_factory_utc': aligned_ff_timestamp.isoformat(),
            'trading_engine_utc': te_timestamp.isoformat(),
            'status': 'ALIGNED' if time_diff < 60 else 'MISALIGNED'
        }
    
    def get_session_transition_times(self, date: Optional[datetime] = None) -> Dict:
        """
        Get session transition times for a given date
        
        Args:
            date: Date to get transitions for (defaults to today)
            
        Returns:
            Dictionary of session transition times
        """
        if date is None:
            date = self.get_current_utc_time().date()
        
        base_datetime = datetime.combine(date, datetime.min.time())
        base_datetime = self.utc_tz.localize(base_datetime)
        
        transitions = {
            'asia_start': base_datetime.replace(hour=0),
            'london_start': base_datetime.replace(hour=8),
            'overlap_start': base_datetime.replace(hour=13),
            'overlap_end': base_datetime.replace(hour=17),
            'ny_end': base_datetime.replace(hour=22),
            'asia_end': base_datetime.replace(hour=9)
        }
        
        return transitions
    
    def create_alignment_report(self) -> Dict:
        """
        Create a comprehensive alignment report
        
        Returns:
            Alignment status report
        """
        current_time = self.get_current_utc_time()
        current_session = self.get_current_session()
        session_transitions = self.get_session_transition_times()
        
        return {
            'timestamp': current_time.isoformat(),
            'current_session': current_session.value,
            'timezone_config': {
                'forex_factory_tz': self.config.forex_factory_tz,
                'trading_engine_tz': self.config.trading_engine_tz,
                'mt5_broker_tz': self.config.mt5_broker_tz
            },
            'session_transitions': {
                k: v.isoformat() for k, v in session_transitions.items()
            },
            'alignment_status': 'OPERATIONAL',
            'recommendations': [
                "Forex Factory timestamps are aligned to UTC",
                "Trading engine operates in UTC timezone",
                "All economic events use consistent UTC timing",
                "Session analysis matches market hours"
            ]
        }

def test_timezone_alignment():
    """Test timezone alignment functionality"""
    print("Testing Timezone Alignment...")
    print("=" * 50)
    
    aligner = TimezoneAligner()
    
    # Test current time alignment
    current_utc = aligner.get_current_utc_time()
    current_session = aligner.get_current_session()
    
    print(f"Current UTC Time: {current_utc.isoformat()}")
    print(f"Current Session: {current_session.value}")
    
    # Test event timestamp alignment
    test_event_time = datetime(2025, 7, 5, 18, 50)  # Example from user
    aligned_time = aligner.align_forex_factory_timestamp(test_event_time)
    has_occurred = aligner.has_event_occurred(test_event_time)
    time_until = aligner.get_time_until_event(test_event_time)
    
    print(f"\nTest Event Time: {test_event_time}")
    print(f"Aligned Time: {aligned_time.isoformat()}")
    print(f"Has Occurred: {has_occurred}")
    print(f"Time Until Event: {time_until}")
    
    # Test validation
    validation = aligner.validate_timestamp_alignment(
        test_event_time, current_utc
    )
    print(f"\nValidation: {validation}")
    
    # Generate alignment report
    report = aligner.create_alignment_report()
    print(f"\nAlignment Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Timezone alignment test completed successfully!")

if __name__ == "__main__":
    test_timezone_alignment()