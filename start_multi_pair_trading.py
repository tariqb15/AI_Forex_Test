#!/usr/bin/env python3
"""
BLOB AI - Multi-Pair Automated Trading System
Analyzes and trades multiple currency pairs simultaneously
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import our modules
from multi_pair_engine import MultiPairAnalysisEngine
from trade_executor import TradeExecutor, Position
from config import ForexEngineConfig
from forex_factory_scraper import ForexFactoryScraper, EconomicEvent
from edge_case_handler import EdgeCaseHandler
from signal_conflict_resolver import SignalConflictResolver
from order_management_system import OrderManagementSystem

class MultiPairAutomatedTrader:
    """Multi-pair automated trading orchestrator"""
    
    def __init__(self, currency_pairs: List[str], config_path: Optional[str] = None):
        self.currency_pairs = currency_pairs
        self.config = ForexEngineConfig(config_path) if config_path else ForexEngineConfig()
        self.multi_pair_engine = None
        self.trade_executor = None
        self.edge_case_handler = None
        self.forex_scraper = None
        self.signal_conflict_resolver = None
        self.order_management = None
        self.is_running = False
        self.last_signal_time = None
        self.last_event_check = None
        self.start_time = None
        self.signal_stats = {
            'total_generated': 0,
            'executed': 0,
            'filtered': 0
        }
        
        # Economic event tracking
        self.current_events = []
        self.high_impact_events = []
        self.event_risk_level = "LOW"  # LOW, MEDIUM, HIGH
        
        # Performance tracking per pair
        self.pair_stats = {}
        for pair in currency_pairs:
            self.pair_stats[pair] = {
                "signals_generated": 0,
                "trades_executed": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_profit": 0.0
            }
        
        # Overall daily stats
        self.daily_stats = {
            "total_signals": 0,
            "total_trades": 0,
            "total_profit": 0.0,
            "start_balance": 0.0,
            "events_monitored": 0,
            "pairs_analyzed": len(currency_pairs)
        }
        
    def initialize(self) -> bool:
        """Initialize the multi-pair trading system"""
        logger.info("🚀 Initializing BLOB AI Multi-Pair Trading System...")
        logger.info(f"📊 Currency Pairs: {', '.join(self.currency_pairs)}")
        
        # Initialize MT5
        if not mt5.initialize():
            logger.error("❌ Failed to initialize MetaTrader 5")
            return False
        
        # Test MT5 connection
        account_info = mt5.account_info()
        if not account_info:
            logger.error("❌ Failed to get account information")
            return False
        
        logger.info(f"✅ Connected to MT5 Account: {account_info.login}")
        logger.info(f"💰 Account Balance: ${account_info.balance:.2f}")
        logger.info(f"📈 Account Equity: ${account_info.equity:.2f}")
        
        # Initialize engines
        try:
            self.multi_pair_engine = MultiPairAnalysisEngine(self.currency_pairs)
            self.trade_executor = TradeExecutor(self.config)
            self.edge_case_handler = EdgeCaseHandler()
            self.signal_conflict_resolver = SignalConflictResolver(self.config)
            self.order_management = OrderManagementSystem(self.config)
            
            # Initialize economic event monitoring
            self.forex_scraper = ForexFactoryScraper()
            
            # Record starting balance
            self.daily_stats["start_balance"] = account_info.balance
            
            # Initialize date tracking for daily statistics
            self._last_stats_date = datetime.now().date()
            
            # Initial event check
            self._update_economic_events()
            
            logger.info("✅ All systems initialized successfully")
            logger.info(f"🔍 Multi-pair analysis engine with {len(self.currency_pairs)} pairs")
            logger.info(f"📅 Economic event monitoring: {len(self.current_events)} events tracked")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize engines: {e}")
            return False
    
    def is_trading_session_active(self) -> bool:
        """Check if current time is within trading hours"""
        now = datetime.now()
        
        # Define trading sessions (UTC time)
        sessions = {
            "london": (7, 16),    # 7:00 - 16:00 UTC
            "new_york": (12, 21), # 12:00 - 21:00 UTC
            "asian": (21, 6)      # 21:00 - 06:00 UTC (next day)
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
            self.last_event_check = datetime.now()
            
            if self.high_impact_events:
                logger.info(f"📅 {len(self.high_impact_events)} high/medium impact events in next 24h")
                for event in self.high_impact_events[:3]:  # Log first 3
                    time_to_event = (event.datetime - datetime.now()).total_seconds() / 3600
                    logger.info(f"  • {event.currency} {event.event} in {time_to_event:.1f}h (Impact: {event.impact})")
            
        except Exception as e:
            logger.error(f"❌ Error updating economic events: {e}")
    
    def _assess_event_risk_level(self):
        """Assess current market risk based on upcoming events"""
        now = datetime.now()
        
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
    
    def should_generate_signals(self) -> bool:
        """Check if signals should be generated"""
        # Check trading session
        if not self.is_trading_session_active():
            return False
        
        # Check event risk
        if self.event_risk_level == "HIGH":
            logger.info("⚠️ Skipping signal generation - HIGH event risk")
            return False
        
        # Check frequency (don't generate signals too frequently)
        if self.last_signal_time:
            time_since_last = datetime.now() - self.last_signal_time
            if time_since_last < timedelta(minutes=5):  # Minimum 5 minutes between signal generations
                return False
        
        return True
    
    def analyze_all_pairs(self) -> Dict:
        """Analyze all currency pairs simultaneously"""
        logger.info(f"🔍 Analyzing {len(self.currency_pairs)} currency pairs...")
        
        # Use ThreadPoolExecutor for parallel analysis
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(self.currency_pairs), 8)) as executor:
            # Submit analysis tasks
            future_to_pair = {
                executor.submit(self.multi_pair_engine.analyze_single_pair, pair): pair 
                for pair in self.currency_pairs
            }
            
            # Collect results
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout per pair
                    results[pair] = result
                    
                    if 'error' not in result:
                        signals = result.get('trading_signals', [])
                        self.pair_stats[pair]["signals_generated"] += len(signals)
                        logger.info(f"✅ {pair}: {len(signals)} signals generated")
                    else:
                        logger.warning(f"⚠️ {pair}: {result['error']}")
                        
                except Exception as e:
                    logger.error(f"❌ {pair}: Analysis failed - {e}")
                    results[pair] = {'error': str(e), 'currency_pair': pair}
        
        return results
    
    def generate_and_execute_signals(self):
        """Enhanced multi-pair trading logic with economic event awareness"""
        try:
            logger.info("=== Starting Multi-Pair Signal Generation and Execution Cycle ===")
            logger.info(f"📊 Market Risk Level: {self.event_risk_level}")
            
            # Check if we should generate signals
            if not self.should_generate_signals():
                logger.info("⏸️ Skipping signal generation - outside trading hours or event risk")
                return
            
            # Analyze all pairs
            analysis_results = self.analyze_all_pairs()
            
            # Collect all signals from all pairs
            all_signals = []
            for pair, result in analysis_results.items():
                # Ensure result is a dictionary
                if not isinstance(result, dict):
                    logger.warning(f"⚠️ {pair}: Invalid result type {type(result)}: {result}")
                    continue
                    
                if 'error' not in result:
                    signals = result.get('trading_signals', [])
                    for signal in signals:
                        signal.currency_pair = pair  # Add pair identifier
                        all_signals.append(signal)
            
            logger.info(f"📈 Total signals collected: {len(all_signals)}")
            self.daily_stats["total_signals"] += len(all_signals)
            self.signal_stats['total_generated'] += len(all_signals)
            
            if not all_signals:
                logger.info("📭 No signals generated across all pairs")
                return
            
            # Filter signals by event risk
            event_filtered_signals = self._filter_signals_by_events(all_signals)
            
            if not event_filtered_signals:
                logger.info("🚫 All signals filtered out due to event risk")
                self.signal_stats['filtered'] += len(all_signals)
                return
            
            # Resolve signal conflicts and improve signal quality
            logger.info(f"🔧 Resolving signal conflicts for {len(event_filtered_signals)} signals...")
            conflict_resolution = self.signal_conflict_resolver.resolve_signal_conflicts(event_filtered_signals)
            
            logger.info(f"📊 Conflict Resolution Results:")
            logger.info(f"   - Conflicts Found: {conflict_resolution.conflicts_found}")
            logger.info(f"   - Conflicts Resolved: {conflict_resolution.conflicts_resolved}")
            logger.info(f"   - Consolidation Applied: {conflict_resolution.consolidation_applied}")
            logger.info(f"   - Final Signals: {len(conflict_resolution.resolved_signals)}")
            logger.info(f"   - Reasoning: {conflict_resolution.reasoning}")
            
            if not conflict_resolution.resolved_signals:
                logger.info("🚫 No valid signals after conflict resolution")
                self.signal_stats['filtered'] += len(event_filtered_signals)
                return
            
            # Select best signals from conflict-resolved signals
            best_signals = self._select_best_signals(conflict_resolution.resolved_signals)
            
            # Execute selected signals with duplicate trade prevention
            executed_count = 0
            for signal in best_signals:
                try:
                    # Check if trade should be allowed (duplicate prevention)
                    should_allow, reason = self.order_management.should_allow_trade(signal.currency_pair, signal.direction)
                    
                    if not should_allow:
                        logger.warning(f"🚫 Trade blocked for {signal.currency_pair} {signal.direction}: {reason}")
                        continue
                    
                    # Update config for the specific pair
                    self.config.trading.symbol = signal.currency_pair
                    
                    # Execute the signal
                    position = self.trade_executor.execute_signal(signal)
                    
                    # Handle different result types
                    if position is None:
                        logger.warning(f"⚠️ Failed to execute trade: {signal.currency_pair} {signal.direction} - No result returned")
                    elif hasattr(position, 'retcode'):
                        # This is an MT5 result object
                        if position.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"✅ Trade executed: {signal.currency_pair} {signal.direction} at {signal.entry_price}")
                            self.pair_stats[signal.currency_pair]["trades_executed"] += 1
                            self.daily_stats["total_trades"] += 1
                            executed_count += 1
                        else:
                            logger.warning(f"⚠️ Failed to execute trade: {signal.currency_pair} {signal.direction} - MT5 Error: {position.retcode}")
                    elif hasattr(position, 'ticket'):
                        # This is a Position object (successful execution)
                        logger.info(f"✅ Trade executed: {signal.currency_pair} {signal.direction} at {signal.entry_price} - Ticket: {position.ticket}")
                        self.pair_stats[signal.currency_pair]["trades_executed"] += 1
                        self.daily_stats["total_trades"] += 1
                        executed_count += 1
                    else:
                        logger.warning(f"⚠️ Unexpected result type for {signal.currency_pair} {signal.direction}: {type(position)}")
                        
                except Exception as e:
                    logger.error(f"❌ Error executing signal for {signal.currency_pair}: {e}")
            
            # Update signal statistics
            self.signal_stats['executed'] += executed_count
            filtered_count = len(event_filtered_signals) - len(best_signals)
            self.signal_stats['filtered'] += filtered_count
            
            # Update timestamp
            self.last_signal_time = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Error in signal generation cycle: {e}")
    
    def _filter_signals_by_events(self, signals: List) -> List:
        """Filter signals based on economic events"""
        if self.event_risk_level == "LOW":
            return signals
        
        filtered = []
        for signal in signals:
            # Get currency from pair (e.g., USD from USDJPY)
            currencies = [signal.currency_pair[:3], signal.currency_pair[3:]]
            
            # Check if any upcoming events affect this currency pair
            affected = False
            for event in self.high_impact_events:
                if event.currency in currencies:
                    time_to_event = (event.datetime - datetime.now()).total_seconds() / 3600
                    if time_to_event <= 2:  # Within 2 hours
                        affected = True
                        break
            
            if not affected:
                filtered.append(signal)
            else:
                logger.info(f"🚫 Filtered {signal.currency_pair} signal due to upcoming {event.currency} event")
                self.daily_stats["event_based_decisions"] += 1
        
        return filtered
    
    def _select_best_signals(self, signals: List, max_signals: int = 3) -> List:
        """Select the best signals for execution"""
        if not signals:
            return []
        
        # Sort by confidence (assuming signals have confidence attribute)
        try:
            sorted_signals = sorted(signals, key=lambda x: getattr(x, 'confidence', 0), reverse=True)
        except:
            sorted_signals = signals
        
        # Limit to max_signals and ensure we don't exceed position limits
        current_positions = len(self.trade_executor.position_manager.get_open_positions())
        available_slots = max(0, self.config.trading.max_open_positions - current_positions)
        
        selected = sorted_signals[:min(max_signals, available_slots)]
        
        if len(selected) < len(sorted_signals):
            logger.info(f"📊 Selected {len(selected)} best signals out of {len(sorted_signals)}")
        
        return selected
    
    def manage_existing_positions(self):
        """Manage existing positions across all pairs"""
        try:
            positions = self.trade_executor.position_manager.get_open_positions()
            
            if not positions:
                return
            
            logger.info(f"📊 Managing {len(positions)} open positions")
            
            # Apply exit strategies
            self.trade_executor.exit_manager.update_trailing_stops()
            self.trade_executor.exit_manager.apply_breakeven_stops()
            self.trade_executor.exit_manager.apply_time_based_exits()
            
            # Update position statistics
            for position in positions:
                pair = position.symbol
                if pair in self.pair_stats:
                    if position.profit > 0:
                        # This is a simplified check - in reality you'd track when positions close
                        pass
            
        except Exception as e:
            logger.error(f"❌ Error managing positions: {e}")
    
    def print_daily_summary(self):
        """Print comprehensive daily trading summary"""
        try:
            # Print order management system status
            if self.order_management:
                self.order_management.print_status_report()
            
            account_info = mt5.account_info()
            current_balance = account_info.balance if account_info else 0
            
            daily_pnl = current_balance - self.daily_stats["start_balance"]
            
            logger.info("\n" + "="*60)
            logger.info("📊 MULTI-PAIR DAILY PERFORMANCE SUMMARY")
            logger.info("="*60)
            logger.info(f"💰 Starting Balance: ${self.daily_stats['start_balance']:.2f}")
            logger.info(f"💰 Current Balance: ${current_balance:.2f}")
            logger.info(f"📈 Daily P&L: ${daily_pnl:.2f}")
            logger.info(f"📊 Total Signals: {self.daily_stats['total_signals']}")
            logger.info(f"🎯 Total Trades: {self.daily_stats['total_trades']}")
            logger.info(f"📅 Events Monitored: {self.daily_stats['events_monitored']}")
            logger.info(f"🌍 Pairs Analyzed: {self.daily_stats['pairs_analyzed']}")
            
            # Signal generation stats
            if hasattr(self, 'signal_stats'):
                logger.info(f"📡 Signals Generated Today: {self.signal_stats.get('total_generated', 0)}")
                logger.info(f"🎯 Signals Executed: {self.signal_stats.get('executed', 0)}")
                logger.info(f"🚫 Signals Filtered: {self.signal_stats.get('filtered', 0)}")
            
            # Economic events impact
            if self.forex_scraper and hasattr(self.forex_scraper, 'get_events_for_date'):
                events_today = len([e for e in self.forex_scraper.get_events_for_date(datetime.now().date()) 
                                  if e.impact in ['Medium', 'High']])
                logger.info(f"📅 High/Medium Impact Events Today: {events_today}")
            
            # System uptime
            if hasattr(self, 'start_time') and self.start_time:
                uptime = datetime.now() - self.start_time
                hours, remainder = divmod(uptime.total_seconds(), 3600)
                minutes, _ = divmod(remainder, 60)
                logger.info(f"⏱️ System Uptime: {int(hours)}h {int(minutes)}m")
            
            # Per-pair breakdown
            logger.info("\n📊 PER-PAIR BREAKDOWN:")
            for pair, stats in self.pair_stats.items():
                logger.info(f"  {pair}: {stats['signals_generated']} signals, {stats['trades_executed']} trades")
            
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"❌ Error printing daily summary: {e}")
    
    def run_trading_cycle(self):
        """Run a complete trading cycle"""
        try:
            # Check if it's a new day and reset stats
            current_date = datetime.now().date()
            if current_date != self._last_stats_date:
                self.print_daily_summary()
                self._reset_daily_stats()
                self._last_stats_date = current_date
            
            # Update economic events periodically
            if not self.last_event_check or \
               (datetime.now() - self.last_event_check) > timedelta(hours=1):
                self._update_economic_events()
            
            # Generate and execute signals
            self.generate_and_execute_signals()
            
            # Update order management system
            self.order_management.update_positions()
            
            # Apply risk management
            risk_updates = self.order_management.apply_risk_management()
            if risk_updates:
                logger.info(f"🛡️ Applying {len(risk_updates)} risk management updates...")
                self.order_management.execute_position_updates(risk_updates)
            
            # Manage existing positions (legacy)
            self.manage_existing_positions()
            
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")
    
    def _reset_daily_stats(self):
        """Reset daily statistics"""
        account_info = mt5.account_info()
        self.daily_stats = {
            "total_signals": 0,
            "total_trades": 0,
            "total_profit": 0.0,
            "start_balance": account_info.balance if account_info else 0,
            "events_monitored": 0,
            "pairs_analyzed": len(self.currency_pairs)
        }
        
        for pair in self.pair_stats:
            self.pair_stats[pair] = {
                "signals_generated": 0,
                "trades_executed": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_profit": 0.0
            }
    
    def start_automated_trading(self):
        """Start the automated trading system"""
        if not self.initialize():
            logger.error("❌ Failed to initialize trading system")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        logger.info("🚀 Starting Multi-Pair Automated Trading...")
        
        # Run initial trading cycle immediately
        logger.info("🎯 Running initial trading cycle...")
        self.run_trading_cycle()
        
        # Schedule trading cycles every 5 minutes
        schedule.every(5).minutes.do(self.run_trading_cycle)
        
        # Schedule daily summary at midnight
        schedule.every().day.at("00:00").do(self.print_daily_summary)
        
        logger.info("⏰ Scheduled cycles: Trading every 5 minutes, Daily summary at midnight")
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("🛑 Stopping automated trading...")
            self.stop_trading()
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            self.stop_trading()
    
    def stop_trading(self):
        """Stop the automated trading system"""
        self.is_running = False
        
        # Print final summary
        self.print_daily_summary()
        
        # Close forex scraper
        if self.forex_scraper and self.forex_scraper.driver:
            self.forex_scraper.driver.quit()
        
        # Disconnect MT5
        mt5.shutdown()
        
        logger.info("✅ Multi-Pair Automated Trading System stopped")

def main():
    """Main function to start multi-pair trading"""
    # Define currency pairs to trade
    currency_pairs = [
        'EURJPY', 'GBPJPY', 'USDJPY', 'CHFJPY', 'EURAUD', 'AUDJPY', 'EURNZD', 
        'GBPNZD', 'NZDUSD', 'GBPUSD', 'NZDJPY', 'GBPCHF', 'GBPAUD', 'EURCAD', 
        'AUDUSD', 'USDCHF', 'EURCHF', 'AUDCAD'
    ]
    
    logger.info(f"🌍 Initializing Multi-Pair Trading with {len(currency_pairs)} pairs")
    
    # Create and start the multi-pair trader
    trader = MultiPairAutomatedTrader(currency_pairs)
    trader.start_automated_trading()

if __name__ == "__main__":
    main()