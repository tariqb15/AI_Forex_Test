#!/usr/bin/env python3
"""
BLOB AI - Enhanced Automated Trading Orchestrator
Integrates signal generation with trade execution for fully automated trading
Enhanced with real-time economic event awareness and timezone synchronization
"""

import time
import schedule
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pytz
from loguru import logger
import MetaTrader5 as mt5
import json
from pathlib import Path

# Import our modules
from forex_engine import AgenticForexEngine, TradingSignal
from multi_pair_engine import MultiPairAnalysisEngine
from trade_executor import TradeExecutor, Position
from config import ForexEngineConfig
from forex_factory_scraper import ForexFactoryScraper, EconomicEvent
from timezone_alignment import TimezoneAligner, TimezoneConfig
from edge_case_handler import EdgeCaseHandler

class AutomatedTrader:
    """Enhanced automated trading orchestrator with economic event awareness and multi-pair support"""
    
    def __init__(self, config_path: Optional[str] = None, currency_pairs: Optional[List[str]] = None):
        self.config = ForexEngineConfig(config_path) if config_path else ForexEngineConfig()
        self.currency_pairs = currency_pairs or ["USDJPY"]  # Default to single pair for backward compatibility
        self.multi_pair_engine = None
        self.forex_engine = None  # Keep for backward compatibility
        self.trade_executor = None
        self.edge_case_handler = None
        self.forex_scraper = None
        self.timezone_aligner = None
        self.is_running = False
        self.last_signal_time = None
        self.last_event_check = None
        self.trading_session_active = False
        
        # Economic event tracking
        self.current_events = []
        self.high_impact_events = []
        self.event_risk_level = "LOW"  # LOW, MEDIUM, HIGH
        
        # Performance tracking
        self.daily_stats = {
            "signals_generated": 0,
            "trades_executed": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "start_balance": 0.0,
            "events_monitored": 0,
            "event_based_decisions": 0
        }
        
    def initialize(self) -> bool:
        """Initialize the automated trading system"""
        logger.info("Initializing BLOB AI Automated Trading System...")
        
        # Initialize MT5
        if not mt5.initialize():
            logger.error("Failed to initialize MetaTrader 5")
            return False
        
        # Test MT5 connection
        account_info = mt5.account_info()
        if not account_info:
            logger.error("Failed to get account information")
            return False
        
        logger.info(f"Connected to MT5 Account: {account_info.login}")
        logger.info(f"Account Balance: ${account_info.balance:.2f}")
        logger.info(f"Account Equity: ${account_info.equity:.2f}")
        
        # Initialize engines
        try:
            # Initialize multi-pair engine if multiple pairs, otherwise single pair for compatibility
            if len(self.currency_pairs) > 1:
                logger.info(f"Initializing multi-pair engine for {len(self.currency_pairs)} pairs: {', '.join(self.currency_pairs)}")
                self.multi_pair_engine = MultiPairAnalysisEngine(self.currency_pairs)
                # Also initialize single engine for the first pair for compatibility
                self.forex_engine = AgenticForexEngine(self.currency_pairs[0])
            else:
                logger.info(f"Initializing single-pair engine for {self.currency_pairs[0]}")
                self.forex_engine = AgenticForexEngine(self.currency_pairs[0])
                self.multi_pair_engine = None
            
            self.trade_executor = TradeExecutor(self.config)
            self.edge_case_handler = EdgeCaseHandler()
            
            # Initialize economic event monitoring
            self.forex_scraper = ForexFactoryScraper()
            # Note: Disabled timezone aligner to work with broker time
            # self.timezone_aligner = TimezoneAligner(TimezoneConfig())
            self.timezone_aligner = None
            
            # Record starting balance
            self.daily_stats["start_balance"] = account_info.balance
            
            # Initialize date tracking for daily statistics
            self._last_stats_date = datetime.now().date()
            
            # Initial event check
            self._update_economic_events()
            
            logger.info("All systems initialized successfully")
            if self.multi_pair_engine:
                logger.info(f"Multi-pair engine with {len(self.currency_pairs)} pairs and edge case validation enabled")
            else:
                logger.info(f"{self.currency_pairs[0]} engine with edge case validation enabled")
            logger.info(f"Economic event monitoring: {len(self.current_events)} events tracked")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize engines: {e}")
            return False
    
    def is_trading_session_active(self) -> bool:
        """Check if current time is within trading hours"""
        # Use broker time instead of UTC
        if self.forex_engine:
            now = self.forex_engine._get_broker_time().replace(tzinfo=None)
        else:
            now = datetime.now()  # Fallback to local time
        
        # Define trading sessions (broker time - adjust as needed for your broker)
        sessions = {
            "london": (7, 16),    # 7:00 - 16:00 broker time
            "new_york": (12, 21), # 12:00 - 21:00 broker time
            "asian": (21, 6)      # 21:00 - 06:00 broker time (next day)
        }
        
        current_hour = now.hour
        
        # Check if within any major trading session
        for session, (start, end) in sessions.items():
            if session == "asian":
                # Asian session spans midnight
                if current_hour >= start or current_hour <= end:
                    return True
            else:
                if start <= current_hour <= end:
                    return True
        
        return False
    
    def _update_economic_events(self):
        """Update economic events and assess market risk"""
        try:
            # Setup driver if not already initialized
            if not self.forex_scraper.driver:
                self.forex_scraper._setup_driver()
            
            # First scrape all events from calendar
            all_events = self.forex_scraper.scrape_calendar(days_ahead=7)
            
            # Get current events (next 24 hours)
            self.current_events = self.forex_scraper.get_imminent_events(all_events, minutes_ahead=1440)
            
            # Filter high impact events
            self.high_impact_events = [
                event for event in self.current_events 
                if event.impact in ['HIGH', 'MEDIUM']
            ]
            
            # Assess overall event risk level
            self._assess_event_risk_level()
            
            self.daily_stats["events_monitored"] = len(self.current_events)
            self.last_event_check = datetime.now()  # Use local time for event checking
            
            if self.high_impact_events:
                logger.info(f"📅 {len(self.high_impact_events)} high/medium impact events in next 24h")
                for event in self.high_impact_events[:3]:  # Log first 3
                    # Calculate time to event using broker time
                    broker_time = self.forex_engine._get_broker_time().replace(tzinfo=None) if self.forex_engine else datetime.now()
                    time_to_event = (event.datetime - broker_time).total_seconds() / 3600
                    logger.info(f"  • {event.currency} {event.event} in {time_to_event:.1f}h (Impact: {event.impact})")
            
        except Exception as e:
            logger.error(f"Error updating economic events: {e}")
    
    def _assess_event_risk_level(self):
        """Assess current market risk based on upcoming events"""
        # Use broker time for event proximity checking
        now = self.forex_engine._get_broker_time().replace(tzinfo=None) if self.forex_engine else datetime.now()
        
        # Check for events in next 2 hours
        imminent_high_impact = [
            event for event in self.high_impact_events
            if (event.datetime - now).total_seconds() <= 7200  # 2 hours
        ]
        
        # Check for events in next 30 minutes
        very_imminent = [
            event for event in self.high_impact_events
            if (event.datetime - now).total_seconds() <= 1800  # 30 minutes
        ]
        
        if very_imminent:
            self.event_risk_level = "HIGH"
        elif imminent_high_impact:
            self.event_risk_level = "MEDIUM"
        elif self.high_impact_events:
            self.event_risk_level = "LOW"
        else:
            self.event_risk_level = "LOW"
    
    def _should_avoid_trading_due_to_events(self) -> bool:
        """Check if trading should be avoided due to upcoming events"""
        # Use broker time for event proximity checking
        now = self.forex_engine._get_broker_time().replace(tzinfo=None) if self.forex_engine else datetime.now()
        
        # Avoid trading 15 minutes before and after high impact events
        for event in self.high_impact_events:
            time_to_event = (event.datetime - now).total_seconds()
            
            # 15 minutes before to 15 minutes after
            if -900 <= time_to_event <= 900:  # 15 minutes = 900 seconds
                logger.warning(f"🚫 Avoiding trading due to {event.currency} {event.event} event")
                return True
        
        return False
    
    def should_generate_signals(self) -> bool:
        """Determine if new signals should be generated with event awareness"""
        # Check if trading session is active
        if not self.is_trading_session_active():
            return False
        
        # Note: Events are updated every 30 minutes via scheduled task
        # No need to update here to avoid conflicts
        
        # Check if we should avoid trading due to events
        if self._should_avoid_trading_due_to_events():
            self.daily_stats["event_based_decisions"] += 1
            return False
        
        # Limit signal generation frequency (every 15 minutes)
        if self.last_signal_time:
            time_since_last = datetime.now() - self.last_signal_time
            if time_since_last < timedelta(minutes=15):
                return False
        
        return True
    
    def generate_and_execute_signals(self):
        """Enhanced trading logic with economic event awareness and multi-pair support"""
        try:
            logger.info("=== Starting Enhanced Signal Generation and Execution Cycle ===")
            logger.info(f"📊 Market Risk Level: {self.event_risk_level}")
            
            # Check if we should generate signals
            if not self.should_generate_signals():
                logger.info("Skipping signal generation - outside trading hours, too frequent, or event risk")
                return
            
            # Generate signals with event context
            if self.multi_pair_engine:
                logger.info(f"Generating event-aware trading signals for {len(self.currency_pairs)} pairs...")
                # Run multi-pair analysis
                analysis_results = self.multi_pair_engine.run_parallel_analysis()
                
                # Extract all signals from multi-pair results
                raw_signals = []
                for pair_result in analysis_results.values():
                    if pair_result.get('status') == 'success' and pair_result.get('signals'):
                        raw_signals.extend(pair_result['signals'])
            else:
                logger.info("Generating event-aware trading signals...")
                raw_signals = self.forex_engine.generate_signals()
            
            # Apply edge case validation to signals
            signals = self._validate_signals_with_edge_cases(raw_signals) if raw_signals else []
            
            if not signals:
                logger.info("No signals generated")
                return
            
            # Filter signals based on event risk
            filtered_signals = self._filter_signals_by_events(signals)
            
            self.daily_stats["signals_generated"] += len(signals)
            self.last_signal_time = datetime.now()
            
            logger.info(f"Generated {len(signals)} signals, {len(filtered_signals)} after event filtering:")
            for i, signal in enumerate(filtered_signals, 1):
                logger.info(f"  {i}. {signal.direction} {signal.signal_type} - "
                          f"Entry: {signal.entry_price}, SL: {signal.stop_loss}, "
                          f"TP: {signal.take_profit}, Confidence: {signal.confidence:.2f}, "
                          f"R/R: {signal.risk_reward:.2f}")
            
            # Execute the best signal with event-adjusted risk
            if filtered_signals:
                executed_position = self.trade_executor.execute_best_signal(filtered_signals)
                
                if executed_position:
                    self.daily_stats["trades_executed"] += 1
                    logger.info(f"✅ Trade executed successfully: Ticket {executed_position.ticket}")
                    
                    # Log trade details with event context
                    self._log_trade_execution_with_events(executed_position)
                else:
                    logger.info("No trade executed (filtered by risk management or duplicates)")
            else:
                logger.info("No signals passed event filtering")
            
        except Exception as e:
            logger.error(f"Error in signal generation and execution: {e}")
    
    def _filter_signals_by_events(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals based on economic events affecting their currencies"""
        if not self.high_impact_events:
            return signals
        
        filtered_signals = []
        # Use broker time for event proximity checking
        now = self.forex_engine._get_broker_time().replace(tzinfo=None) if self.forex_engine else datetime.now()
        
        for signal in signals:
            # Extract currency from symbol (e.g., EURUSD -> EUR, USD)
            base_currency = signal.symbol[:3]
            quote_currency = signal.symbol[3:6]
            
            # Check if any high impact events affect this currency pair
            affected_by_events = False
            for event in self.high_impact_events:
                time_to_event = (event.datetime - now).total_seconds() / 3600  # hours
                
                # Skip signals for currencies with events in next 4 hours
                if (event.currency in [base_currency, quote_currency] and 
                    0 <= time_to_event <= 4):
                    logger.info(f"🚫 Filtering {signal.symbol} signal due to {event.currency} event in {time_to_event:.1f}h")
                    affected_by_events = True
                    break
            
            if not affected_by_events:
                # Adjust confidence based on overall market risk
                if self.event_risk_level == "HIGH":
                    signal.confidence *= 0.7  # Reduce confidence by 30%
                elif self.event_risk_level == "MEDIUM":
                    signal.confidence *= 0.85  # Reduce confidence by 15%
                
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def _validate_signals_with_edge_cases(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Validate signals against edge case scenarios"""
        validated_signals = []
        
        try:
            for signal in signals:
                # Get current market analysis for context
                analysis_results = self.forex_engine.run_full_analysis()
                
                # Check if analysis_results is a dictionary (not a string error message)
                if not isinstance(analysis_results, dict):
                    logger.warning(f"Analysis results is not a dictionary: {type(analysis_results)}, skipping edge case validation")
                    validated_signals.append(signal)
                    continue
                
                # Prepare context for edge case validation
                context = {
                    'signal': signal,
                    'current_session': self.forex_engine._get_current_session(),
                    'symbol': self.forex_engine.symbol
                }
                
                # Prepare signal data and market data for validation
                signal_data = {
                    'signal_type': signal.signal_type,
                    'direction': signal.direction,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'confidence': signal.confidence,
                    'timestamp': signal.timestamp
                }
                
                # Extract market data from analysis
                market_data = {
                    'structure_breaks': analysis_results.get('smc_analysis', {}).get('structure_breaks', []),
                    'order_blocks': analysis_results.get('timeframe_analysis', {}).get('M15', {}).get('order_blocks', []),
                    'fair_value_gaps': analysis_results.get('timeframe_analysis', {}).get('M15', {}).get('fair_value_gaps', []),
                    'liquidity_sweeps': analysis_results.get('timeframe_analysis', {}).get('M15', {}).get('liquidity_sweeps', []),
                    'atr': analysis_results.get('volatility_analysis', {}).get('atr_14', 0.01),
                    'trend_context': analysis_results.get('trend_analysis', {}),
                    'H4': analysis_results.get('timeframe_analysis', {}).get('H4', {})
                }
                
                # Validate signal with edge case handler
                validation_result = self.edge_case_handler.validate_signal(signal_data, market_data, context)
                
                if validation_result.is_valid:
                    # Apply confidence adjustment
                    signal.confidence = max(0.0, min(1.0, signal.confidence + validation_result.confidence_adjustment))
                    validated_signals.append(signal)
                    logger.info(f"✅ Signal validated: {signal.signal_type} - {validation_result.reasoning}")
                else:
                    logger.warning(f"🚫 Signal rejected: {signal.signal_type} - {validation_result.reasoning}")
                    
        except Exception as e:
            logger.error(f"Error in edge case validation: {e}")
            # Return original signals if validation fails
            return signals
            
        logger.info(f"Edge case validation: {len(signals)} → {len(validated_signals)} signals")
        return validated_signals
    
    def _log_trade_execution_with_events(self, position: Position):
        """Enhanced trade logging with economic event context"""
        # Get relevant events for this currency pair
        base_currency = position.symbol[:3]
        quote_currency = position.symbol[3:6]
        
        relevant_events = [
            event for event in self.current_events
            if event.currency in [base_currency, quote_currency]
        ]
        
        trade_log = {
            "timestamp": datetime.now().isoformat(),
            "ticket": position.ticket,
            "symbol": position.symbol,
            "type": position.order_type.value,
            "volume": position.volume,
            "entry_price": position.entry_price,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "magic_number": position.magic_number,
            "event_context": {
                "risk_level": self.event_risk_level,
                "relevant_events": [
                    {
                        "currency": event.currency,
                        "event": event.event,
                        "impact": event.impact,
                        "time_to_event_hours": (event.datetime - (self.forex_engine._get_broker_time().replace(tzinfo=None) if self.forex_engine else datetime.now())).total_seconds() / 3600
                    }
                    for event in relevant_events[:3]  # Limit to 3 most relevant
                ]
            }
        }
        
        # Save to enhanced trade log file
        log_file = Path("enhanced_trade_log.json")
        
        if log_file.exists():
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(trade_log)
        
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
        
        # Also call original logging method
        self._log_trade_execution(position)
    
    def manage_existing_positions(self):
        """Manage all open positions with advanced strategies"""
        try:
            logger.info("Managing existing positions...")
            
            # Get current positions
            positions = self.trade_executor.position_manager.get_open_positions()
            
            if not positions:
                logger.info("No open positions to manage")
                return
            
            logger.info(f"Managing {len(positions)} open positions")
            
            # Apply advanced exit strategies
            self.trade_executor.manage_open_positions()
            
            # Log position status
            for pos in positions:
                logger.info(f"Position {pos.ticket}: {pos.order_type.value} {pos.volume} lots, "
                          f"Entry: {pos.entry_price}, Current: {pos.current_price}, "
                          f"P&L: ${pos.profit:.2f}")
            
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
    
    def _reset_daily_statistics(self):
        """Reset daily statistics for a new trading day"""
        account_info = mt5.account_info()
        current_balance = account_info.balance if account_info else 0
        
        self.daily_stats = {
            "signals_generated": 0,
            "trades_executed": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "start_balance": current_balance,
            "events_monitored": 0,
            "event_based_decisions": 0
        }
        logger.info(f"📊 Daily statistics reset for new trading day. Starting balance: ${current_balance:.2f}")
    
    def update_daily_statistics(self):
        """Update daily trading statistics"""
        print(f"\n=== DEBUG: update_daily_statistics called ===")
        print(f"Magic number: {self.config.trading.magic_number}")
        print(f"Current daily_stats: {self.daily_stats}")
        try:
            logger.info(f"📊 Starting daily statistics update. Current stats before update: W:{self.daily_stats['winning_trades']}, L:{self.daily_stats['losing_trades']}, P&L:${self.daily_stats['total_profit']:.2f}")
            
            # Check if we need to reset for a new day
            if not hasattr(self, '_last_stats_date'):
                self._last_stats_date = datetime.now().date()
            
            current_date = datetime.now().date()
            if current_date != self._last_stats_date:
                logger.info("🔄 New trading day detected, resetting daily statistics")
                self._reset_daily_statistics()
                self._last_stats_date = current_date
            
            # Get today's closed trades
            today = datetime.now().date()
            logger.info(f"📊 Querying MT5 deals for today: {today}")
            deals = mt5.history_deals_get(
                datetime.combine(today, datetime.min.time()), 
                datetime.now()
            )
            
            logger.info(f"📊 MT5 deals query result: {deals is not None}, Type: {type(deals)}")
            
            # Reset counters to avoid accumulation
            today_winning = 0
            today_losing = 0
            today_profit = 0.0
            
            if deals:
                logger.info(f"📊 Found {len(deals)} total deals for today")
                for deal in deals:
                    if deal.magic == self.config.trading.magic_number:
                        logger.info(f"📊 Deal found: Magic={deal.magic}, Profit=${deal.profit:.2f}, Time={datetime.fromtimestamp(deal.time)}")
                        if deal.profit > 0:
                            today_winning += 1
                        elif deal.profit < 0:
                            today_losing += 1
                        
                        today_profit += deal.profit
                    else:
                        logger.info(f"📊 Skipping deal with different magic number: {deal.magic} (expected: {self.config.trading.magic_number})")
            else:
                logger.info("📊 No deals found for today")
            
            logger.info(f"📊 Calculated today's stats: W:{today_winning}, L:{today_losing}, P&L:${today_profit:.2f}")
            
            # Update with actual counts (not accumulative)
            self.daily_stats["winning_trades"] = today_winning
            self.daily_stats["losing_trades"] = today_losing
            self.daily_stats["total_profit"] = today_profit
            
            logger.info(f"📊 Updated daily stats: W:{self.daily_stats['winning_trades']}, L:{self.daily_stats['losing_trades']}, P&L:${self.daily_stats['total_profit']:.2f}")
            
            # Log daily performance
            self._log_daily_performance()
            
        except Exception as e:
            logger.error(f"Error updating daily statistics: {e}")
    
    def _log_trade_execution(self, position: Position):
        """Log detailed trade execution information"""
        trade_log = {
            "timestamp": datetime.now().isoformat(),
            "ticket": position.ticket,
            "symbol": position.symbol,
            "type": position.order_type.value,
            "volume": position.volume,
            "entry_price": position.entry_price,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "magic_number": position.magic_number
        }
        
        # Save to trade log file
        log_file = Path("trade_log.json")
        
        if log_file.exists():
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(trade_log)
        
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
    
    def _log_daily_performance(self):
        """Log daily performance summary"""
        print(f"\n=== DEBUG: _log_daily_performance called ===")
        print(f"daily_stats at start of _log_daily_performance: {self.daily_stats}")
        account_info = mt5.account_info()
        current_balance = account_info.balance if account_info else 0
        
        win_rate = 0
        total_trades = self.daily_stats["winning_trades"] + self.daily_stats["losing_trades"]
        if total_trades > 0:
            win_rate = (self.daily_stats["winning_trades"] / total_trades) * 100
        
        performance_summary = f"""
        📊 ENHANCED DAILY PERFORMANCE SUMMARY 📊
        ==========================================
        Signals Generated: {self.daily_stats['signals_generated']}
        Trades Executed: {self.daily_stats['trades_executed']}
        Winning Trades: {self.daily_stats['winning_trades']}
        Losing Trades: {self.daily_stats['losing_trades']}
        Win Rate: {win_rate:.1f}%
        Total P&L: ${self.daily_stats['total_profit']:.2f}
        Start Balance: ${self.daily_stats['start_balance']:.2f}
        Current Balance: ${current_balance:.2f}
        Daily Return: {((current_balance - self.daily_stats['start_balance']) / self.daily_stats['start_balance'] * 100):.2f}%
        
        📅 ECONOMIC EVENT AWARENESS:
        Events Monitored: {self.daily_stats['events_monitored']}
        Event-Based Decisions: {self.daily_stats['event_based_decisions']}
        Current Risk Level: {self.event_risk_level}
        High Impact Events (24h): {len(self.high_impact_events)}
        ==========================================
        """
        
        logger.info(performance_summary)
    
    def run_trading_cycle(self):
        """Execute one complete enhanced trading cycle with event awareness"""
        logger.info("🔄 Starting enhanced trading cycle...")
        
        # Note: Economic events are updated every 30 minutes via scheduled task
        # Only update here if no recent update (fallback safety)
        if (not self.last_event_check or 
            (datetime.now() - self.last_event_check).total_seconds() > 1800):  # 30 minutes
            logger.info("📅 Fallback event update (no recent scheduled update)")
            self._update_economic_events()
        
        # 1. Generate and execute event-aware signals
        self.generate_and_execute_signals()
        
        # 2. Manage existing positions with event context
        self.manage_existing_positions()
        
        # 3. Update statistics
        self.update_daily_statistics()
        
        # 4. Get portfolio status with event context
        portfolio_status = self.trade_executor.get_portfolio_status()
        logger.info(f"Portfolio: {portfolio_status['open_positions']} positions, "
                   f"Total P&L: ${portfolio_status['total_profit']:.2f}, "
                   f"Event Risk: {self.event_risk_level}")
        
        logger.info("✅ Enhanced trading cycle completed")
    
    def start_automated_trading(self):
        """Start the automated trading system"""
        if not self.initialize():
            logger.error("Failed to initialize automated trading system")
            return
        
        self.is_running = True
        logger.info("🚀 BLOB AI Enhanced Automated Trading System STARTED")
        logger.info("📅 Economic event monitoring: ACTIVE")
        logger.info("🌐 Timezone synchronization: UTC aligned")
        
        # Run initial trading cycle immediately
        logger.info("🎯 Running initial trading cycle...")
        self.run_trading_cycle()
        
        # Schedule enhanced trading cycles
        schedule.every(5).minutes.do(self.run_trading_cycle)
        schedule.every(1).hours.do(self.update_daily_statistics)
        schedule.every(30).minutes.do(self._update_economic_events)  # Regular event updates
        
        logger.info("⏰ Scheduled cycles: Trading every 5 minutes, Statistics every hour, Events every 30 minutes")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            self.stop_automated_trading()
    
    def stop_automated_trading(self):
        """Stop the automated trading system"""
        self.is_running = False
        
        # Final performance summary
        self._log_daily_performance()
        
        # Shutdown MT5
        mt5.shutdown()
        
        logger.info("🛑 BLOB AI Enhanced Automated Trading System STOPPED")
        logger.info("📊 Final event statistics logged")
    
    def run_single_cycle(self):
        """Run a single trading cycle for testing"""
        if not self.initialize():
            logger.error("Failed to initialize automated trading system")
            return
        
        try:
            self.run_trading_cycle()
        finally:
            mt5.shutdown()

# CLI Interface
if __name__ == "__main__":
    import argparse
    
    # Configure logger for CLI usage
    logger.add("automated_trader_cli.log", rotation="1 day", retention="7 days")
    
    parser = argparse.ArgumentParser(description="BLOB AI Automated Trading System")
    parser.add_argument("--mode", choices=["auto", "single", "status"], default="single",
                       help="Trading mode: auto (continuous), single (one cycle), status (portfolio only)")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--pairs", type=str, nargs="+", 
                       help="Currency pairs to trade (e.g., --pairs USDJPY EURJPY GBPJPY)")
    parser.add_argument("--multi-pair", action="store_true",
                       help="Enable multi-pair trading with default 18 pairs")
    
    args = parser.parse_args()
    
    # Determine currency pairs to use
    if args.multi_pair:
        # Use the same 18 pairs as in start_multi_pair_trading.py
        currency_pairs = [
            "EURJPY", "GBPJPY", "USDJPY", "CHFJPY", "AUDJPY", "NZDJPY",
            "EURAUD", "EURNZD", "GBPNZD", "NZDUSD", "GBPUSD", "GBPCHF",
            "GBPAUD", "EURCAD", "AUDUSD", "USDCHF", "EURCHF", "AUDCAD"
        ]
        print(f"Multi-pair mode enabled with {len(currency_pairs)} pairs")
        print(f"Pairs: {', '.join(currency_pairs)}")
    elif args.pairs:
        currency_pairs = args.pairs
        print(f"Custom pairs specified: {', '.join(currency_pairs)}")
    else:
        currency_pairs = ["USDJPY"]  # Default single pair
        print("Single-pair mode (USDJPY)")
    
    trader = AutomatedTrader(args.config, currency_pairs)
    
    if args.mode == "auto":
        trader.start_automated_trading()
    elif args.mode == "single":
        trader.run_single_cycle()
    elif args.mode == "status":
        if trader.initialize():
            try:
                status = trader.trade_executor.get_portfolio_status()
                print(json.dumps(status, indent=2))
            finally:
                mt5.shutdown()