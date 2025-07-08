#!/usr/bin/env python3
"""
BLOB AI - Automated Trade Execution Engine
Implements position tracking, duplicate prevention, automatic execution, and advanced exit strategies
"""

import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pytz
import time
from loguru import logger

# Import existing classes
from forex_engine import TradingSignal, SignalStrength, MarketStructure
from config import ForexEngineConfig

class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"

@dataclass
class Position:
    """Represents an open trading position"""
    ticket: int
    symbol: str
    order_type: OrderType
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    current_price: float
    profit: float
    open_time: datetime
    signal_id: str
    status: PositionStatus
    magic_number: int

@dataclass
class TradeRequest:
    """Trade execution request"""
    signal: TradingSignal
    volume: float
    stop_loss: float
    take_profit: float
    magic_number: int

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: ForexEngineConfig):
        self.config = config
        self.daily_loss_limit = config.trading.max_daily_loss
        self.max_positions = config.trading.max_open_positions
        self.risk_per_trade = config.trading.risk_per_trade
        
    def get_account_balance(self) -> float:
        """Get current account balance"""
        account_info = mt5.account_info()
        return account_info.balance if account_info else 0.0
    
    def get_account_equity(self) -> float:
        """Get current account equity"""
        account_info = mt5.account_info()
        return account_info.equity if account_info else 0.0
    
    def calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        today = datetime.now().date()
        deals = mt5.history_deals_get(datetime.combine(today, datetime.min.time()), datetime.now())
        
        if not deals:
            return 0.0
            
        daily_pnl = sum(deal.profit for deal in deals if deal.type in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL])
        return daily_pnl
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        daily_pnl = self.calculate_daily_pnl()
        balance = self.get_account_balance()
        
        if daily_pnl < 0:
            loss_percentage = abs(daily_pnl) / balance
            return loss_percentage >= self.daily_loss_limit
        return False
    
    def calculate_position_size(self, signal: TradingSignal, account_balance: float) -> float:
        """Calculate optimal position size based on risk management"""
        # Calculate risk amount in account currency
        risk_amount = account_balance * self.risk_per_trade
        
        # Calculate stop loss distance in pips
        if signal.direction == "BUY":
            sl_distance_price = signal.entry_price - signal.stop_loss
        else:
            sl_distance_price = signal.stop_loss - signal.entry_price
        
        # Convert to pips (for USDJPY: 1 pip = 0.01)
        sl_distance_pips = sl_distance_price / 0.01
        
        # Calculate position size
        # For USDJPY: 1 lot = 100,000 units, 1 pip = $10 per lot
        pip_value = 10.0  # $10 per pip for 1 lot USDJPY
        position_size = risk_amount / (sl_distance_pips * pip_value)
        
        # Apply minimum and maximum limits
        min_lot = 0.01
        max_lot = 10.0
        
        return max(min_lot, min(position_size, max_lot))
    
    def validate_trade_risk(self, signal: TradingSignal, position_size: float) -> bool:
        """Validate if trade meets risk criteria"""
        # Check daily loss limit
        if self.check_daily_loss_limit():
            logger.warning("Daily loss limit exceeded - blocking new trades")
            return False
        
        # Check risk/reward ratio
        if signal.risk_reward < 1.0:
            logger.warning(f"Poor risk/reward ratio: {signal.risk_reward} - blocking trade")
            return False
        
        # Check confidence level
        if signal.confidence < self.config.trading.min_confidence:
            logger.warning(f"Low confidence signal: {signal.confidence} < {self.config.trading.min_confidence} - blocking trade")
            return False
        
        return True

class PositionManager:
    """Manages open positions and prevents duplicates"""
    
    def __init__(self, config: ForexEngineConfig):
        self.config = config
        self.symbol = config.trading.symbol
        self.magic_number = config.trading.magic_number
        # Demo mode: store simulated positions
        self._demo_positions = []
        
    def get_open_positions(self) -> List[Position]:
        """Get all open positions for the symbol"""
        from config import TradingMode
        
        # Demo mode: return simulated positions
        if self.config.trading_mode == TradingMode.DEMO:
            return self._demo_positions.copy()
        
        # Live mode: get real MT5 positions
        positions = []
        mt5_positions = mt5.positions_get(symbol=self.symbol)
        
        if not mt5_positions:
            return positions
        
        for pos in mt5_positions:
            if pos.magic == self.magic_number:
                position = Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    order_type=OrderType.BUY if pos.type == mt5.POSITION_TYPE_BUY else OrderType.SELL,
                    volume=pos.volume,
                    entry_price=pos.price_open,
                    stop_loss=pos.sl,
                    take_profit=pos.tp,
                    current_price=pos.price_current,
                    profit=pos.profit,
                    open_time=datetime.fromtimestamp(pos.time, tz=pytz.UTC),
                    signal_id=f"{pos.magic}_{pos.ticket}",
                    status=PositionStatus.OPEN,
                    magic_number=pos.magic
                )
                positions.append(position)
        
        return positions
    
    def check_duplicate_trades(self, signal: TradingSignal) -> bool:
        """Check if similar trade already exists"""
        open_positions = self.get_open_positions()
        
        for position in open_positions:
            # Check for same symbol and direction trades
            if (position.symbol == signal.symbol and 
                ((signal.direction == "BUY" and position.order_type == OrderType.BUY) or 
                 (signal.direction == "SELL" and position.order_type == OrderType.SELL))):
                
                # Calculate pip difference based on currency pair
                pip_size = self._get_pip_size(signal.symbol)
                price_difference = abs(signal.entry_price - position.entry_price)
                pip_difference = price_difference / pip_size
                
                if pip_difference < 20:  # 20 pips threshold
                    logger.info(f"Duplicate trade detected: {signal.symbol} {signal.direction} near {position.entry_price}")
                    return True
        
        return False
    
    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for a currency pair"""
        if 'JPY' in symbol:
            return 0.01  # JPY pairs: 1 pip = 0.01
        elif any(x in symbol for x in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return 0.1   # Precious metals: 1 pip = 0.1
        else:
            return 0.0001  # Major pairs: 1 pip = 0.0001
    
    def get_position_count(self) -> int:
        """Get current number of open positions"""
        return len(self.get_open_positions())
    
    def add_demo_position(self, position: Position):
        """Add a simulated position in demo mode"""
        from config import TradingMode
        if self.config.trading_mode == TradingMode.DEMO:
            self._demo_positions.append(position)
            logger.info(f"DEMO MODE: Added simulated position {position.ticket}")
    
    def close_position(self, ticket: int, reason: str = "Manual close") -> bool:
        """Close a specific position"""
        from config import TradingMode
        
        # Demo mode: remove from simulated positions
        if self.config.trading_mode == TradingMode.DEMO:
            for i, pos in enumerate(self._demo_positions):
                if pos.ticket == ticket:
                    self._demo_positions.pop(i)
                    logger.info(f"DEMO MODE: Closed simulated position {ticket}: {reason}")
                    return True
            logger.error(f"DEMO MODE: Position {ticket} not found")
            return False
        
        # Live mode: close real MT5 position
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error(f"Position {ticket} not found")
            return False
        
        pos = position[0]
        
        # Prepare close request
        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(pos.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(pos.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "magic": self.magic_number,
            "comment": f"Close: {reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"LIVE MODE: Position {ticket} closed successfully: {reason}")
            return True
        else:
            logger.error(f"LIVE MODE: Failed to close position {ticket}: {result.comment}")
            return False

class ExitManager:
    """Advanced exit strategy management"""
    
    def __init__(self, config: ForexEngineConfig):
        self.config = config
        self.position_manager = PositionManager(config)
        
    def update_trailing_stops(self):
        """Update trailing stops for profitable positions"""
        positions = self.position_manager.get_open_positions()
        
        for position in positions:
            if position.profit > 0:  # Only trail profitable positions
                self._apply_trailing_stop(position)
    
    def _apply_trailing_stop(self, position: Position):
        """Apply trailing stop to a position"""
        current_price = position.current_price
        trailing_distance = 0.15  # 15 pips trailing distance
        
        if position.order_type == OrderType.BUY:
            new_sl = current_price - trailing_distance
            if new_sl > position.stop_loss:
                self._modify_position(position.ticket, new_sl, position.take_profit)
        else:
            new_sl = current_price + trailing_distance
            if new_sl < position.stop_loss:
                self._modify_position(position.ticket, new_sl, position.take_profit)
    
    def apply_breakeven_stops(self):
        """Move stops to breakeven when in profit"""
        positions = self.position_manager.get_open_positions()
        
        for position in positions:
            # Move to breakeven when 20 pips in profit
            profit_threshold = 0.20  # 20 pips
            
            if position.order_type == OrderType.BUY:
                if position.current_price >= position.entry_price + profit_threshold:
                    if position.stop_loss < position.entry_price:
                        self._modify_position(position.ticket, position.entry_price, position.take_profit)
            else:
                if position.current_price <= position.entry_price - profit_threshold:
                    if position.stop_loss > position.entry_price:
                        self._modify_position(position.ticket, position.entry_price, position.take_profit)
    
    def apply_time_based_exits(self):
        """Close positions based on time criteria"""
        positions = self.position_manager.get_open_positions()
        # Use UTC time to match position.open_time timezone
        current_time = datetime.now(tz=pytz.UTC)
        
        for position in positions:
            # Close positions older than 24 hours if not profitable
            time_diff = current_time - position.open_time
            if time_diff > timedelta(hours=24) and position.profit <= 0:
                self.position_manager.close_position(position.ticket, "Time-based exit")
    
    def _modify_position(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        """Modify position stop loss and take profit"""
        # Get position to find its symbol
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error(f"Position {ticket} not found")
            return False
        
        pos = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": new_tp,
            "magic": self.config.trading.magic_number,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {ticket} modified: SL={new_sl}, TP={new_tp}")
            return True
        else:
            logger.error(f"Failed to modify position {ticket}: {result.comment}")
            return False

class TradeExecutor:
    """Main trade execution engine"""
    
    def __init__(self, config: ForexEngineConfig):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.position_manager = PositionManager(config)
        self.exit_manager = ExitManager(config)
        self.symbol = config.trading.symbol
        self.magic_number = config.trading.magic_number
        
    def execute_best_signal(self, signals: List[TradingSignal]) -> Optional[Position]:
        """Automatically execute the best signal from the list"""
        if not signals:
            logger.info("No signals to execute")
            return None
        
        # Filter and rank signals
        best_signal = self._select_best_signal(signals)
        
        if not best_signal:
            logger.info("No suitable signal found for execution")
            return None
        
        return self.execute_signal(best_signal)
    
    def _select_best_signal(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Select the best signal for execution"""
        # Check position limits
        if self.position_manager.get_position_count() >= self.config.trading.max_open_positions:
            logger.warning("Maximum position limit reached")
            return None
        
        # Filter out duplicate signals
        valid_signals = []
        for signal in signals:
            if not self.position_manager.check_duplicate_trades(signal):
                valid_signals.append(signal)
        
        if not valid_signals:
            logger.info("All signals filtered out due to duplicates")
            return None
        
        # Rank by composite score (confidence × risk_reward)
        ranked_signals = sorted(valid_signals, 
                              key=lambda s: s.confidence * s.risk_reward, 
                              reverse=True)
        
        return ranked_signals[0]
    
    def execute_signal(self, signal: TradingSignal) -> Optional[Position]:
        """Execute a trading signal"""
        logger.info(f"Executing signal: {signal.direction} {signal.signal_type} at {signal.entry_price}")
        
        try:
            # Calculate position size
            account_balance = self.risk_manager.get_account_balance()
            if account_balance is None or account_balance <= 0:
                logger.error(f"Invalid account balance: {account_balance}")
                return None
                
            position_size = self.risk_manager.calculate_position_size(signal, account_balance)
            if position_size is None or position_size <= 0:
                logger.error(f"Invalid position size calculated: {position_size}")
                return None
            
            logger.info(f"Position size calculated: {position_size} lots for {signal.symbol}")
            
            # Validate trade risk
            if not self.risk_manager.validate_trade_risk(signal, position_size):
                logger.warning("Trade rejected by risk management")
                return None
            
            # Execute the trade
            result = self._place_market_order(signal, position_size)
            if result is None:
                logger.error(f"Failed to place market order for {signal.symbol} {signal.direction}")
            return result
            
        except Exception as e:
            logger.error(f"Error in execute_signal for {signal.symbol}: {e}")
            return None
    
    def _validate_and_adjust_stops(self, symbol: str, direction: str, entry_price: float, stop_loss: float, take_profit: float) -> tuple:
        """Validate and adjust stop loss and take profit levels according to broker requirements"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return stop_loss, take_profit
            
            # Get minimum stop level in points
            stops_level = symbol_info.trade_stops_level
            point = symbol_info.point
            
            # Convert stops level to price
            min_distance = stops_level * point
            
            # Validate stop loss and take profit levels
            if direction == "BUY":
                # For BUY orders: SL must be below entry, TP must be above entry
                if stop_loss >= entry_price:
                    # Invalid SL - set it below entry price with minimum distance
                    adjusted_sl = entry_price - max(min_distance, 0.0001)  # At least 1 pip below
                    logger.warning(f"Invalid BUY SL {stop_loss} >= entry {entry_price}. Adjusted to {adjusted_sl}")
                    stop_loss = adjusted_sl
                elif stop_loss > entry_price - min_distance:
                    # SL too close to entry
                    adjusted_sl = entry_price - min_distance
                    logger.warning(f"Adjusted SL from {stop_loss} to {adjusted_sl} (min distance: {min_distance})")
                    stop_loss = adjusted_sl
                    
                if take_profit <= entry_price:
                    # Invalid TP - set it above entry price with minimum distance
                    adjusted_tp = entry_price + max(min_distance, 0.0001)  # At least 1 pip above
                    logger.warning(f"Invalid BUY TP {take_profit} <= entry {entry_price}. Adjusted to {adjusted_tp}")
                    take_profit = adjusted_tp
                elif take_profit < entry_price + min_distance:
                    # TP too close to entry
                    adjusted_tp = entry_price + min_distance
                    logger.warning(f"Adjusted TP from {take_profit} to {adjusted_tp} (min distance: {min_distance})")
                    take_profit = adjusted_tp
            else:  # SELL
                # For SELL orders: SL must be above entry, TP must be below entry
                if stop_loss <= entry_price:
                    # Invalid SL - set it above entry price with minimum distance
                    adjusted_sl = entry_price + max(min_distance, 0.0001)  # At least 1 pip above
                    logger.warning(f"Invalid SELL SL {stop_loss} <= entry {entry_price}. Adjusted to {adjusted_sl}")
                    stop_loss = adjusted_sl
                elif stop_loss < entry_price + min_distance:
                    # SL too close to entry
                    adjusted_sl = entry_price + min_distance
                    logger.warning(f"Adjusted SL from {stop_loss} to {adjusted_sl} (min distance: {min_distance})")
                    stop_loss = adjusted_sl
                    
                if take_profit >= entry_price:
                    # Invalid TP - set it below entry price with minimum distance
                    adjusted_tp = entry_price - max(min_distance, 0.0001)  # At least 1 pip below
                    logger.warning(f"Invalid SELL TP {take_profit} >= entry {entry_price}. Adjusted to {adjusted_tp}")
                    take_profit = adjusted_tp
                elif take_profit > entry_price - min_distance:
                    # TP too close to entry
                    adjusted_tp = entry_price - min_distance
                    logger.warning(f"Adjusted TP from {take_profit} to {adjusted_tp} (min distance: {min_distance})")
                    take_profit = adjusted_tp
            
            # Round to symbol's digits
            digits = symbol_info.digits
            stop_loss = round(stop_loss, digits)
            take_profit = round(take_profit, digits)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error validating stops: {e}")
            return stop_loss, take_profit
    
    def _place_market_order(self, signal: TradingSignal, volume: float) -> Optional[Position]:
        """Place a market order"""
        # Use signal's symbol instead of self.symbol for multi-pair trading
        symbol = signal.symbol
        
        # Get current prices
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"Failed to get tick data for {symbol}")
            return None
        
        # Determine order type and price
        if signal.direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        
        # Validate and adjust stop levels
        validated_sl, validated_tp = self._validate_and_adjust_stops(
            symbol, signal.direction, price, signal.stop_loss, signal.take_profit
        )
        
        # Check if in demo mode
        from config import TradingMode
        if self.config.trading_mode == TradingMode.DEMO:
            # Demo mode: simulate trade without actually placing it
            import random
            simulated_ticket = random.randint(100000, 999999)
            logger.info(f"DEMO MODE: Simulated order execution - Ticket {simulated_ticket}")
            logger.info(f"DEMO MODE: {signal.direction} {volume} lots of {symbol} at {price}")
            logger.info(f"DEMO MODE: SL={validated_sl}, TP={validated_tp}")
            
            # Create simulated position object
            demo_position = Position(
                ticket=simulated_ticket,
                symbol=symbol,
                order_type=OrderType.BUY if signal.direction == "BUY" else OrderType.SELL,
                volume=volume,
                entry_price=price,
                stop_loss=validated_sl,
                take_profit=validated_tp,
                current_price=price,
                profit=0.0,
                open_time=datetime.now(),
                signal_id=f"{self.magic_number}_{simulated_ticket}",
                status=PositionStatus.OPEN,
                magic_number=self.magic_number
            )
            
            # Add to position manager's demo positions
            self.position_manager.add_demo_position(demo_position)
            
            return demo_position
        
        # Live mode: place actual order
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": validated_sl,
            "tp": validated_tp,
            "magic": self.magic_number,
            "comment": f"BLOB AI: {signal.signal_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"LIVE MODE: Order executed successfully - Ticket {result.order}")
            
            # Return position object
            return Position(
                ticket=result.order,
                symbol=symbol,
                order_type=OrderType.BUY if signal.direction == "BUY" else OrderType.SELL,
                volume=volume,
                entry_price=result.price,
                stop_loss=validated_sl,
                take_profit=validated_tp,
                current_price=result.price,
                profit=0.0,
                open_time=datetime.now(),
                signal_id=f"{self.magic_number}_{result.order}",
                status=PositionStatus.OPEN,
                magic_number=self.magic_number
            )
        else:
            logger.error(f"LIVE MODE: Order failed - {result.comment} (Code: {result.retcode})")
            return None
    
    def manage_open_positions(self):
        """Manage all open positions with advanced exit strategies"""
        self.exit_manager.update_trailing_stops()
        self.exit_manager.apply_breakeven_stops()
        self.exit_manager.apply_time_based_exits()
    
    def get_portfolio_status(self) -> Dict:
        """Get comprehensive portfolio status"""
        positions = self.position_manager.get_open_positions()
        account_info = mt5.account_info()
        
        total_profit = sum(pos.profit for pos in positions)
        daily_pnl = self.risk_manager.calculate_daily_pnl()
        
        return {
            "account_balance": account_info.balance if account_info else 0,
            "account_equity": account_info.equity if account_info else 0,
            "open_positions": len(positions),
            "total_profit": total_profit,
            "daily_pnl": daily_pnl,
            "positions": [{
                "ticket": pos.ticket,
                "type": pos.order_type.value,
                "volume": pos.volume,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "profit": pos.profit,
                "open_time": pos.open_time.isoformat()
            } for pos in positions]
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize MT5
    if not mt5.initialize():
        print("Failed to initialize MT5")
        exit()
    
    try:
        # Load configuration
        config = ForexEngineConfig()
        
        # Create trade executor
        executor = TradeExecutor(config)
        
        # Get portfolio status
        status = executor.get_portfolio_status()
        print(f"Portfolio Status: {status}")
        
        # Manage existing positions
        executor.manage_open_positions()
        
        print("Trade executor initialized successfully")
        
    except Exception as e:
        logger.error(f"Error in trade executor: {e}")
    
    finally:
        mt5.shutdown()