#!/usr/bin/env python3
"""
Forex Factory Calendar Scraper

This script demonstrates how to scrape economic calendar data from Forex Factory.
It uses Selenium WebDriver to handle the dynamic content and extract event information.

Features:
- Scrapes current economic events
- Extracts event details (time, currency, impact, forecast, actual, previous)
- Filters events by impact level or currency
- Handles dynamic content loading

Requirements:
- selenium
- webdriver-manager
- pandas (optional, for data manipulation)
"""

import time
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass
from enum import Enum
import pytz

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not installed. Install with: pip install selenium webdriver-manager")

# Import timezone alignment utility
try:
    from timezone_alignment import TimezoneAligner, TimezoneConfig
except ImportError:
    print("Warning: timezone_alignment module not found, using basic timezone handling")
    TimezoneAligner = None
    TimezoneConfig = None

class ForexFactoryScraperError(Exception):
    """Custom exception for scraper errors"""
    pass

class EventImpact(Enum):
    """Economic event impact levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    UNKNOWN = "Unknown"

class TradingDecision(Enum):
    """Trading decisions based on economic events"""
    AVOID_TRADING = "avoid_trading"
    REDUCE_POSITION = "reduce_position"
    NORMAL_TRADING = "normal_trading"
    INCREASE_VOLATILITY_EXPECTED = "increase_volatility"

@dataclass
class EconomicEvent:
    """Structured economic event data"""
    date: str
    time: str
    currency: str
    impact: EventImpact
    event_name: str
    actual: str
    forecast: str
    previous: str
    parsed_datetime: Optional[datetime] = None
    minutes_until_event: Optional[int] = None
    trading_recommendation: Optional[TradingDecision] = None
    volatility_score: Optional[float] = None

class ForexFactoryScraper:
    """
    Scraper for Forex Factory economic calendar data
    """
    
    def __init__(self, headless: bool = True, timeout: int = 30, timezone_config: Optional['TimezoneConfig'] = None):
        """
        Initialize the scraper
        
        Args:
            headless: Run browser in headless mode
            timeout: Maximum wait time for page elements
            timezone_config: Timezone configuration for alignment
        """
        if not SELENIUM_AVAILABLE:
            raise ForexFactoryScraperError("Selenium is required but not installed")
            
        self.headless = headless
        self.timeout = timeout
        self.driver = None
        self.base_url = "https://www.forexfactory.com/calendar"
        
        # Initialize timezone alignment
        if TimezoneAligner and TimezoneConfig:
            # Note: Disabled timezone aligner to work with broker time
            # self.timezone_aligner = TimezoneAligner(timezone_config or TimezoneConfig())
            self.timezone_aligner = None
            print("Timezone alignment disabled - using broker time")
        else:
            self.timezone_aligner = None
            print("Warning: Timezone alignment disabled - using basic UTC handling")
        
    def __enter__(self):
        """Context manager entry"""
        self._setup_driver()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        
    def _setup_driver(self):
        """Setup Chrome WebDriver with appropriate options"""
        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            
        except Exception as e:
            raise ForexFactoryScraperError(f"Failed to setup WebDriver: {e}")
            
    def scrape_calendar(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """
        Scrape economic calendar events
        
        Args:
            days_ahead: Number of days ahead to scrape (default: 7)
            
        Returns:
            List of event dictionaries
        """
        if not self.driver:
            raise ForexFactoryScraperError("Driver not initialized")
            
        try:
            # Navigate to calendar page
            print(f"Navigating to {self.base_url}...")
            self.driver.get(self.base_url)
            
            # Wait for calendar table to load
            wait = WebDriverWait(self.driver, self.timeout)
            calendar_table = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "calendar__table"))
            )
            
            # Wait a bit more for dynamic content
            time.sleep(3)
            
            # Extract events using JavaScript
            events_script = """
            const events = [];
            const table = document.querySelector('.calendar__table');
            if (table) {
                const rows = table.querySelectorAll('tr.calendar__row');
                rows.forEach(row => {
                    if (row.classList.contains('calendar__row--day-breaker') || 
                        row.classList.contains('subhead')) {
                        return;
                    }
                    
                    const timeEl = row.querySelector('.calendar__time');
                    const currencyEl = row.querySelector('.calendar__currency');
                    const impactEl = row.querySelector('.calendar__impact');
                    const eventEl = row.querySelector('.calendar__event');
                    const actualEl = row.querySelector('.calendar__actual');
                    const forecastEl = row.querySelector('.calendar__forecast');
                    const previousEl = row.querySelector('.calendar__previous');
                    const dateEl = row.querySelector('.calendar__date');
                    
                    if (eventEl && eventEl.textContent.trim()) {
                        let impact = 'Low';
                        if (impactEl) {
                            const span = impactEl.querySelector('span');
                            if (span) {
                                const cls = span.className;
                                if (cls.includes('high')) impact = 'High';
                                else if (cls.includes('medium')) impact = 'Medium';
                            }
                        }
                        
                        events.push({
                            date: dateEl ? dateEl.textContent.trim() : '',
                            time: timeEl ? timeEl.textContent.trim() : '',
                            currency: currencyEl ? currencyEl.textContent.trim() : '',
                            impact: impact,
                            event: eventEl.textContent.trim(),
                            actual: actualEl ? actualEl.textContent.trim() : '',
                            forecast: forecastEl ? forecastEl.textContent.trim() : '',
                            previous: previousEl ? previousEl.textContent.trim() : ''
                        });
                    }
                });
            }
            return events;
            """
            
            raw_events = self.driver.execute_script(events_script)
            print(f"Successfully scraped {len(raw_events)} events")
            
            # Convert to structured events with enhanced data
            events = self._process_events(raw_events)
            print(f"Processed {len(events)} events with enhanced data")
            return events
            
        except Exception as e:
            raise ForexFactoryScraperError(f"Failed to scrape calendar: {e}")
            
    def _process_events(self, raw_events: List[Dict]) -> List[EconomicEvent]:
        """Process raw events into structured EconomicEvent objects"""
        processed_events = []
        current_date = None
        
        for event_data in raw_events:
            try:
                # Parse impact level
                impact = EventImpact.LOW
                if event_data.get('impact'):
                    impact_str = event_data['impact'].lower()
                    if 'high' in impact_str:
                        impact = EventImpact.HIGH
                    elif 'medium' in impact_str:
                        impact = EventImpact.MEDIUM
                    elif 'low' in impact_str:
                        impact = EventImpact.LOW
                    else:
                        impact = EventImpact.UNKNOWN
                
                # Update current date if provided
                if event_data.get('date') and event_data['date'].strip():
                    current_date = event_data['date'].strip()
                
                # Create structured event
                event = EconomicEvent(
                    date=current_date or '',
                    time=event_data.get('time', '').strip(),
                    currency=event_data.get('currency', '').strip(),
                    impact=impact,
                    event_name=event_data.get('event', '').strip(),
                    actual=event_data.get('actual', '').strip(),
                    forecast=event_data.get('forecast', '').strip(),
                    previous=event_data.get('previous', '').strip()
                )
                
                # Parse datetime and add sophisticated reasoning
                self._enhance_event_data(event)
                processed_events.append(event)
                
            except Exception as e:
                print(f"Warning: Failed to process event {event_data}: {e}")
                continue
        
        return processed_events
    
    def _enhance_event_data(self, event: EconomicEvent):
        """Add sophisticated reasoning and timing analysis to event"""
        # Parse datetime
        event.parsed_datetime = self._parse_event_datetime(event.date, event.time)
        
        # Align timestamp with trading engine if timezone aligner is available
        if self.timezone_aligner and event.parsed_datetime:
            event.parsed_datetime = self.timezone_aligner.align_forex_factory_timestamp(event.parsed_datetime)
            
        # Calculate minutes until event
        if event.parsed_datetime:
            if self.timezone_aligner:
                now = self.timezone_aligner.get_current_utc_time()
                time_diff = event.parsed_datetime - now
                event.minutes_until_event = int(time_diff.total_seconds() / 60)
            else:
                # Fallback to basic UTC handling
                if event.parsed_datetime.tzinfo is None:
                    event.parsed_datetime = pytz.UTC.localize(event.parsed_datetime)
                now = datetime.now(pytz.UTC)
                time_diff = event.parsed_datetime - now
                event.minutes_until_event = int(time_diff.total_seconds() / 60)
        
        # Calculate volatility score
        event.volatility_score = self._calculate_volatility_score(event)
        
        # Generate trading recommendation
        event.trading_recommendation = self._generate_trading_recommendation(event)
    
    def _parse_event_datetime(self, date_str: str, time_str: str) -> Optional[datetime]:
        """Parse event date and time into datetime object"""
        try:
            if not date_str or not time_str:
                return None
            
            # Handle various date formats
            current_year = datetime.now().year
            
            # Parse date (e.g., "Mon Jun 30", "Tue Jul 1")
            date_patterns = [
                r'\w+\s+(\w+)\s+(\d+)',  # "Mon Jun 30"
                r'(\w+)\s+(\d+)',        # "Jun 30"
            ]
            
            month_map = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            
            parsed_date = None
            for pattern in date_patterns:
                match = re.search(pattern, date_str.lower())
                if match:
                    if len(match.groups()) == 2:
                        month_str, day_str = match.groups()
                        month = month_map.get(month_str[:3])
                        if month:
                            parsed_date = datetime(current_year, month, int(day_str))
                            break
            
            if not parsed_date:
                return None
            
            # Parse time (e.g., "6:50pm", "All Day", "8:30am")
            if time_str.lower() in ['all day', 'tentative']:
                # Use noon for all-day events
                return parsed_date.replace(hour=12, minute=0, tzinfo=timezone.utc)
            
            time_match = re.search(r'(\d+):(\d+)(am|pm)', time_str.lower())
            if time_match:
                hour, minute, period = time_match.groups()
                hour = int(hour)
                minute = int(minute)
                
                if period == 'pm' and hour != 12:
                    hour += 12
                elif period == 'am' and hour == 12:
                    hour = 0
                
                return parsed_date.replace(hour=hour, minute=minute, tzinfo=timezone.utc)
            
            return None
            
        except Exception as e:
            print(f"Warning: Failed to parse datetime '{date_str} {time_str}': {e}")
            return None
    
    def _calculate_volatility_score(self, event: EconomicEvent) -> float:
        """Calculate volatility score based on event characteristics"""
        score = 0.0
        
        # Base score from impact level
        impact_scores = {
            EventImpact.HIGH: 1.0,
            EventImpact.MEDIUM: 0.6,
            EventImpact.LOW: 0.2,
            EventImpact.UNKNOWN: 0.1
        }
        score += impact_scores.get(event.impact, 0.1)
        
        # Currency importance multiplier
        major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        if event.currency in major_currencies:
            score *= 1.5
        
        # Event type importance
        high_impact_keywords = [
            'interest rate', 'nfp', 'employment', 'gdp', 'inflation', 'cpi',
            'fomc', 'ecb', 'boe', 'rba', 'boc', 'rbnz', 'snb', 'boj'
        ]
        
        event_lower = event.event_name.lower()
        for keyword in high_impact_keywords:
            if keyword in event_lower:
                score *= 1.3
                break
        
        # Forecast vs actual deviation (if available)
        if event.actual and event.forecast:
            try:
                actual_val = float(re.sub(r'[^\d.-]', '', event.actual))
                forecast_val = float(re.sub(r'[^\d.-]', '', event.forecast))
                if forecast_val != 0:
                    deviation = abs((actual_val - forecast_val) / forecast_val)
                    score *= (1 + deviation)
            except (ValueError, ZeroDivisionError):
                pass
        
        return min(score, 5.0)  # Cap at 5.0
    
    def _generate_trading_recommendation(self, event: EconomicEvent) -> TradingDecision:
        """Generate sophisticated trading recommendation based on event analysis"""
        # High impact events within 30 minutes
        if (event.impact == EventImpact.HIGH and 
            event.minutes_until_event is not None and 
            -30 <= event.minutes_until_event <= 30):
            return TradingDecision.AVOID_TRADING
        
        # Medium/High impact events within 15 minutes
        if (event.impact in [EventImpact.HIGH, EventImpact.MEDIUM] and 
            event.minutes_until_event is not None and 
            -15 <= event.minutes_until_event <= 15):
            return TradingDecision.REDUCE_POSITION
        
        # High volatility score events
        if event.volatility_score and event.volatility_score > 2.0:
            if event.minutes_until_event is not None and event.minutes_until_event > 60:
                return TradingDecision.INCREASE_VOLATILITY_EXPECTED
            else:
                return TradingDecision.REDUCE_POSITION
        
        return TradingDecision.NORMAL_TRADING
    
    def filter_events(self, events: List[EconomicEvent], 
                     impact_levels: Optional[List[EventImpact]] = None,
                     currencies: Optional[List[str]] = None,
                     time_window_minutes: Optional[int] = None) -> List[EconomicEvent]:
        """
        Filter events by impact level and/or currency
        
        Args:
            events: List of event dictionaries
            impact_levels: List of impact levels to include ('High', 'Medium', 'Low')
            currencies: List of currencies to include ('USD', 'EUR', etc.)
            
        Returns:
            Filtered list of events
        """
        filtered = events
        
        if impact_levels:
            filtered = [e for e in filtered if e.impact in impact_levels]
            
        if currencies:
            filtered = [e for e in filtered if e.currency in currencies]
        
        if time_window_minutes is not None:
            filtered = [e for e in filtered if 
                       e.minutes_until_event is not None and 
                       abs(e.minutes_until_event) <= time_window_minutes]
            
        return filtered
        
    def get_high_impact_events(self, events: List[EconomicEvent]) -> List[EconomicEvent]:
        """
        Get only high impact events
        
        Args:
            events: List of EconomicEvent objects
            
        Returns:
            List of high impact events
        """
        return self.filter_events(events, impact_levels=[EventImpact.HIGH])
    
    def get_imminent_events(self, events: List[EconomicEvent], 
                           minutes_ahead: int = 60) -> List[EconomicEvent]:
        """
        Get events happening within specified time window
        
        Args:
            events: List of EconomicEvent objects
            minutes_ahead: Time window in minutes
            
        Returns:
            List of imminent events
        """
        if self.timezone_aligner:
            # Use timezone aligner for precise timing
            imminent_events = []
            for event in events:
                if event.parsed_datetime and self.timezone_aligner.is_event_imminent(event.parsed_datetime, minutes_ahead):
                    imminent_events.append(event)
            return imminent_events
        else:
            # Fallback to basic filtering
            return self.filter_events(events, time_window_minutes=minutes_ahead)
    
    def get_trading_recommendations(self, events: List[EconomicEvent]) -> Dict[str, List[EconomicEvent]]:
        """
        Group events by trading recommendations
        
        Args:
            events: List of EconomicEvent objects
            
        Returns:
            Dictionary grouped by trading decisions
        """
        recommendations = {
            TradingDecision.AVOID_TRADING.value: [],
            TradingDecision.REDUCE_POSITION.value: [],
            TradingDecision.NORMAL_TRADING.value: [],
            TradingDecision.INCREASE_VOLATILITY_EXPECTED.value: []
        }
        
        for event in events:
            if event.trading_recommendation:
                recommendations[event.trading_recommendation.value].append(event)
        
        return recommendations
    
    def analyze_market_impact(self, events: List[EconomicEvent]) -> Dict[str, any]:
        """
        Provide sophisticated market impact analysis
        
        Args:
            events: List of EconomicEvent objects
            
        Returns:
            Market analysis dictionary
        """
        analysis = {
            'total_events': len(events),
            'high_impact_count': len([e for e in events if e.impact == EventImpact.HIGH]),
            'imminent_high_impact': len([e for e in events if 
                                       e.impact == EventImpact.HIGH and 
                                       e.minutes_until_event is not None and 
                                       0 <= e.minutes_until_event <= 60]),
            'currency_exposure': {},
            'volatility_forecast': 'low',
            'trading_window_recommendation': 'normal',
            'key_events_today': []
        }
        
        # Currency exposure analysis
        for event in events:
            if event.currency:
                if event.currency not in analysis['currency_exposure']:
                    analysis['currency_exposure'][event.currency] = {
                        'event_count': 0,
                        'high_impact_count': 0,
                        'avg_volatility_score': 0.0
                    }
                
                analysis['currency_exposure'][event.currency]['event_count'] += 1
                if event.impact == EventImpact.HIGH:
                    analysis['currency_exposure'][event.currency]['high_impact_count'] += 1
                if event.volatility_score:
                    analysis['currency_exposure'][event.currency]['avg_volatility_score'] += event.volatility_score
        
        # Calculate average volatility scores
        for currency_data in analysis['currency_exposure'].values():
            if currency_data['event_count'] > 0:
                currency_data['avg_volatility_score'] /= currency_data['event_count']
        
        # Overall volatility forecast
        high_volatility_events = [e for e in events if e.volatility_score and e.volatility_score > 2.0]
        if len(high_volatility_events) > 3:
            analysis['volatility_forecast'] = 'high'
        elif len(high_volatility_events) > 1:
            analysis['volatility_forecast'] = 'medium'
        
        # Trading window recommendation
        avoid_trading_events = [e for e in events if 
                              e.trading_recommendation == TradingDecision.AVOID_TRADING]
        if avoid_trading_events:
            analysis['trading_window_recommendation'] = 'avoid_high_impact_windows'
        elif high_volatility_events:
            analysis['trading_window_recommendation'] = 'reduced_position_sizing'
        
        # Key events identification
        key_events = sorted([e for e in events if e.volatility_score and e.volatility_score > 1.5],
                           key=lambda x: x.volatility_score, reverse=True)[:5]
        analysis['key_events_today'] = [{
            'time': event.time,
            'currency': event.currency,
            'event': event.event_name,
            'impact': event.impact.value,
            'volatility_score': event.volatility_score,
            'recommendation': event.trading_recommendation.value if event.trading_recommendation else None
        } for event in key_events]
        
        return analysis
        
    def save_to_json(self, events: List[EconomicEvent], filename: str = "forex_events.json"):
        """
        Save events to JSON file
        
        Args:
            events: List of EconomicEvent objects
            filename: Output filename
        """
        try:
            # Convert EconomicEvent objects to dictionaries
            events_data = []
            for event in events:
                event_dict = {
                    'date': event.date,
                    'time': event.time,
                    'currency': event.currency,
                    'impact': event.impact.value,
                    'event_name': event.event_name,
                    'actual': event.actual,
                    'forecast': event.forecast,
                    'previous': event.previous,
                    'parsed_datetime': event.parsed_datetime.isoformat() if event.parsed_datetime else None,
                    'minutes_until_event': event.minutes_until_event,
                    'trading_recommendation': event.trading_recommendation.value if event.trading_recommendation else None,
                    'volatility_score': event.volatility_score
                }
                events_data.append(event_dict)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(events_data, f, indent=2, ensure_ascii=False)
            print(f"Events saved to {filename}")
        except Exception as e:
            raise ForexFactoryScraperError(f"Failed to save events: {e}")
    
    def close(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

def main():
    """Example usage of the enhanced Forex Factory scraper with agentic AI capabilities"""
    if not SELENIUM_AVAILABLE:
        print("Please install required packages:")
        print("pip install selenium webdriver-manager")
        return
        
    try:
        # Use context manager for automatic cleanup
        with ForexFactoryScraper(headless=True) as scraper:
            print("Starting Enhanced Forex Factory calendar scraping with AI reasoning...")
            print("=" * 70)
            
            # Scrape all events
            events = scraper.scrape_calendar()
            
            if events:
                print(f"\nTotal events found: {len(events)}")
                
                # Perform sophisticated market analysis
                market_analysis = scraper.analyze_market_impact(events)
                print(f"\n=== MARKET IMPACT ANALYSIS ===")
                print(f"Total Events: {market_analysis['total_events']}")
                print(f"High Impact Events: {market_analysis['high_impact_count']}")
                print(f"Imminent High Impact (next 60 min): {market_analysis['imminent_high_impact']}")
                print(f"Volatility Forecast: {market_analysis['volatility_forecast'].upper()}")
                print(f"Trading Recommendation: {market_analysis['trading_window_recommendation']}")
                
                # Currency exposure analysis
                print(f"\n=== CURRENCY EXPOSURE ===")
                for currency, data in market_analysis['currency_exposure'].items():
                    print(f"{currency}: {data['event_count']} events, {data['high_impact_count']} high impact, "
                          f"avg volatility: {data['avg_volatility_score']:.2f}")
                
                # Trading recommendations
                recommendations = scraper.get_trading_recommendations(events)
                print(f"\n=== TRADING RECOMMENDATIONS ===")
                for decision, event_list in recommendations.items():
                    if event_list:
                        print(f"{decision.replace('_', ' ').title()}: {len(event_list)} events")
                
                # Key events today
                print(f"\n=== KEY EVENTS TODAY ===")
                for i, key_event in enumerate(market_analysis['key_events_today'], 1):
                    print(f"{i}. {key_event['time']} {key_event['currency']} - {key_event['event']}")
                    print(f"   Impact: {key_event['impact']}, Volatility: {key_event['volatility_score']:.2f}")
                    print(f"   Recommendation: {key_event['recommendation']}")
                
                # Imminent events analysis
                imminent_events = scraper.get_imminent_events(events, minutes_ahead=120)
                print(f"\n=== IMMINENT EVENTS (Next 2 Hours) ===")
                for event in imminent_events[:10]:  # Show first 10
                    time_status = "LIVE" if abs(event.minutes_until_event) <= 15 else f"in {event.minutes_until_event} min"
                    print(f"{event.time} {event.currency} - {event.event_name} ({time_status})")
                    print(f"   Impact: {event.impact.value}, Volatility Score: {event.volatility_score:.2f}")
                    print(f"   Trading Action: {event.trading_recommendation.value}")
                
                # Save enhanced data
                scraper.save_to_json(events, "enhanced_forex_events.json")
                
                # Save analysis results
                with open("market_analysis.json", 'w') as f:
                    json.dump(market_analysis, f, indent=2, default=str)
                
                print(f"\n=== FILES SAVED ===")
                print("✓ enhanced_forex_events.json - Complete event data with AI analysis")
                print("✓ market_analysis.json - Sophisticated market impact analysis")
                
                # Display sample enhanced events
                print(f"\n=== SAMPLE ENHANCED EVENTS ===")
                for i, event in enumerate(events[:3]):
                    print(f"\nEvent {i+1}: {event.event_name}")
                    print(f"  📅 DateTime: {event.parsed_datetime}")
                    print(f"  ⏰ Minutes Until: {event.minutes_until_event}")
                    print(f"  💱 Currency: {event.currency}")
                    print(f"  📊 Impact: {event.impact.value}")
                    print(f"  🔥 Volatility Score: {event.volatility_score:.2f}")
                    print(f"  🎯 Trading Recommendation: {event.trading_recommendation.value}")
                    print(f"  📈 Actual: {event.actual} | Forecast: {event.forecast} | Previous: {event.previous}")
                
            else:
                print("No events found")
                
    except ForexFactoryScraperError as e:
        print(f"Scraper error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()