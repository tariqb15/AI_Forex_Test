#!/usr/bin/env python3
"""
Session Timezone Aligner for BLOB AI Forex Engine

Proper timezone alignment between broker time and session times to ensure
session-based signals are generated at the correct times.

Author: BLOB AI Trading System
Version: 1.0.0
"""

import pytz
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum
from loguru import logger

class SessionType(Enum):
    """Trading session types"""
    ASIA = "Asia"
    LONDON = "London"
    NEW_YORK = "New_York"
    OVERLAP = "Overlap"
    INACTIVE = "Inactive"

class SessionTimezoneAligner:
    """
    Handles proper timezone alignment between broker time and session times
    
    Most forex brokers use UTC or UTC+2/3 depending on DST.
    This class converts broker time to UTC for proper session analysis.
    """
    
    def __init__(self, broker_timezone_offset: int = 0):
        """
        Initialize the session timezone aligner
        
        Args:
            broker_timezone_offset: Hours offset from UTC (e.g., 0 for UTC, 2 for UTC+2)
        """
        self.broker_timezone_offset = broker_timezone_offset
        
        # Session times in UTC (standard forex market hours)
        self.utc_session_times = {
            SessionType.ASIA: [(0, 9)],      # 00:00-09:00 UTC
            SessionType.LONDON: [(8, 17)],   # 08:00-17:00 UTC  
            SessionType.NEW_YORK: [(13, 22)], # 13:00-22:00 UTC
            SessionType.OVERLAP: [(13, 17)]   # 13:00-17:00 UTC (London-NY overlap)
        }
        
        logger.info(f"SessionTimezoneAligner initialized with broker offset: UTC{broker_timezone_offset:+d}")
    
    def broker_time_to_utc(self, broker_time: datetime) -> datetime:
        """
        Convert broker time to UTC
        
        Args:
            broker_time: Naive datetime in broker timezone
            
        Returns:
            Datetime in UTC
        """
        try:
            # Subtract broker offset to get UTC
            utc_time = broker_time - timedelta(hours=self.broker_timezone_offset)
            return utc_time
        except Exception as e:
            logger.error(f"Error converting broker time to UTC: {e}")
            return broker_time
    
    def utc_to_broker_time(self, utc_time: datetime) -> datetime:
        """
        Convert UTC time to broker time
        
        Args:
            utc_time: Datetime in UTC
            
        Returns:
            Naive datetime in broker timezone
        """
        try:
            # Add broker offset to get broker time
            broker_time = utc_time + timedelta(hours=self.broker_timezone_offset)
            return broker_time
        except Exception as e:
            logger.error(f"Error converting UTC to broker time: {e}")
            return utc_time
    
    def get_current_session(self, broker_time: Optional[datetime] = None) -> SessionType:
        """
        Get current trading session based on broker time
        
        Args:
            broker_time: Optional broker time, defaults to current time
            
        Returns:
            Current trading session
        """
        if broker_time is None:
            broker_time = datetime.now()
        
        # Convert broker time to UTC for session analysis
        utc_time = self.broker_time_to_utc(broker_time)
        utc_hour = utc_time.hour
        
        # Check sessions in priority order (overlap first)
        if self._is_in_session_range(utc_hour, SessionType.OVERLAP):
            return SessionType.OVERLAP
        elif self._is_in_session_range(utc_hour, SessionType.LONDON):
            return SessionType.LONDON
        elif self._is_in_session_range(utc_hour, SessionType.NEW_YORK):
            return SessionType.NEW_YORK
        elif self._is_in_session_range(utc_hour, SessionType.ASIA):
            return SessionType.ASIA
        else:
            return SessionType.INACTIVE
    
    def _is_in_session_range(self, utc_hour: int, session: SessionType) -> bool:
        """
        Check if UTC hour is within session range
        
        Args:
            utc_hour: Hour in UTC (0-23)
            session: Session type to check
            
        Returns:
            True if hour is within session range
        """
        session_ranges = self.utc_session_times.get(session, [])
        
        for start_hour, end_hour in session_ranges:
            if start_hour <= utc_hour < end_hour:
                return True
        
        return False
    
    def is_session_active(self, session: SessionType, broker_time: Optional[datetime] = None) -> bool:
        """
        Check if a specific session is currently active
        
        Args:
            session: Session type to check
            broker_time: Optional broker time, defaults to current time
            
        Returns:
            True if session is active
        """
        current_session = self.get_current_session(broker_time)
        
        # Overlap session is also considered London and NY active
        if current_session == SessionType.OVERLAP:
            return session in [SessionType.LONDON, SessionType.NEW_YORK, SessionType.OVERLAP]
        
        return current_session == session
    
    def get_session_start_end_broker_time(self, session: SessionType, date: Optional[datetime] = None) -> Tuple[datetime, datetime]:
        """
        Get session start and end times in broker timezone
        
        Args:
            session: Session type
            date: Date for session times (defaults to today)
            
        Returns:
            Tuple of (start_time, end_time) in broker timezone
        """
        if date is None:
            date = datetime.now().date()
        
        session_ranges = self.utc_session_times.get(session, [(0, 0)])
        start_hour, end_hour = session_ranges[0]  # Take first range
        
        # Create UTC times
        utc_start = datetime.combine(date, datetime.min.time().replace(hour=start_hour))
        utc_end = datetime.combine(date, datetime.min.time().replace(hour=end_hour))
        
        # Convert to broker time
        broker_start = self.utc_to_broker_time(utc_start)
        broker_end = self.utc_to_broker_time(utc_end)
        
        return broker_start, broker_end
    
    def get_session_info(self, broker_time: Optional[datetime] = None) -> Dict:
        """
        Get comprehensive session information
        
        Args:
            broker_time: Optional broker time, defaults to current time
            
        Returns:
            Dictionary with session information
        """
        if broker_time is None:
            broker_time = datetime.now()
        
        utc_time = self.broker_time_to_utc(broker_time)
        current_session = self.get_current_session(broker_time)
        
        # Get next session transition
        next_session, time_to_next = self._get_next_session_transition(utc_time)
        
        return {
            'broker_time': broker_time.strftime('%Y-%m-%d %H:%M:%S'),
            'utc_time': utc_time.strftime('%Y-%m-%d %H:%M:%S'),
            'current_session': current_session.value,
            'next_session': next_session.value if next_session else 'Unknown',
            'time_to_next_session_minutes': int(time_to_next.total_seconds() / 60) if time_to_next else None,
            'broker_timezone_offset': f"UTC{self.broker_timezone_offset:+d}",
            'session_active': {
                'asia': self.is_session_active(SessionType.ASIA, broker_time),
                'london': self.is_session_active(SessionType.LONDON, broker_time),
                'new_york': self.is_session_active(SessionType.NEW_YORK, broker_time),
                'overlap': self.is_session_active(SessionType.OVERLAP, broker_time)
            }
        }
    
    def _get_next_session_transition(self, utc_time: datetime) -> Tuple[Optional[SessionType], Optional[timedelta]]:
        """
        Get the next session transition time
        
        Args:
            utc_time: Current UTC time
            
        Returns:
            Tuple of (next_session, time_until_transition)
        """
        current_hour = utc_time.hour
        current_minute = utc_time.minute
        
        # Define session transition hours in UTC
        transitions = [
            (0, SessionType.ASIA),    # 00:00 UTC - Asia starts
            (8, SessionType.LONDON),  # 08:00 UTC - London starts
            (13, SessionType.OVERLAP), # 13:00 UTC - Overlap starts
            (17, SessionType.NEW_YORK), # 17:00 UTC - NY continues (overlap ends)
            (22, SessionType.ASIA),   # 22:00 UTC - NY ends, Asia prep
        ]
        
        # Find next transition
        for hour, session in transitions:
            if current_hour < hour or (current_hour == hour and current_minute < 0):
                next_transition = utc_time.replace(hour=hour, minute=0, second=0, microsecond=0)
                time_diff = next_transition - utc_time
                return session, time_diff
        
        # If no transition found today, get first transition tomorrow
        tomorrow = utc_time + timedelta(days=1)
        next_transition = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        time_diff = next_transition - utc_time
        return SessionType.ASIA, time_diff
    
    def validate_session_alignment(self) -> Dict:
        """
        Validate that session alignment is working correctly
        
        Returns:
            Validation results
        """
        current_time = datetime.now()
        session_info = self.get_session_info(current_time)
        
        # Test conversion both ways
        utc_time = self.broker_time_to_utc(current_time)
        converted_back = self.utc_to_broker_time(utc_time)
        
        time_diff = abs((current_time - converted_back).total_seconds())
        
        return {
            'status': 'ALIGNED' if time_diff < 1 else 'MISALIGNED',
            'time_difference_seconds': time_diff,
            'session_info': session_info,
            'conversion_test': {
                'original_broker_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'converted_utc': utc_time.strftime('%Y-%m-%d %H:%M:%S'),
                'converted_back': converted_back.strftime('%Y-%m-%d %H:%M:%S')
            }
        }

def test_session_alignment():
    """Test session timezone alignment"""
    print("Testing Session Timezone Alignment...")
    print("=" * 50)
    
    # Test with different broker timezone offsets
    for offset in [0, 2, 3, -5]:  # UTC, UTC+2, UTC+3, UTC-5
        print(f"\nTesting with broker timezone UTC{offset:+d}:")
        aligner = SessionTimezoneAligner(broker_timezone_offset=offset)
        
        # Test current session
        current_time = datetime.now()
        session_info = aligner.get_session_info(current_time)
        
        print(f"  Broker Time: {session_info['broker_time']}")
        print(f"  UTC Time: {session_info['utc_time']}")
        print(f"  Current Session: {session_info['current_session']}")
        print(f"  Next Session: {session_info['next_session']} (in {session_info['time_to_next_session_minutes']} min)")
        
        # Validate alignment
        validation = aligner.validate_session_alignment()
        print(f"  Alignment Status: {validation['status']}")
        
        if validation['status'] == 'MISALIGNED':
            print(f"  ⚠️  Time difference: {validation['time_difference_seconds']} seconds")
        else:
            print(f"  ✅ Alignment verified")
    
    print("\n✅ Session timezone alignment test completed!")

if __name__ == "__main__":
    test_session_alignment()