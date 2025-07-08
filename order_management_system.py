import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    MODIFIED = "modified"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class OrderInfo:
    ticket: int
    symbol: str
    order_type: str  # "buy" or "sell"
    volume: float
    open_price: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    status: OrderStatus
    profit: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    comment: str = ""
    magic_number: int = 0
    
@dataclass
class RiskMetrics:
    current_drawdown: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    risk_reward_ratio: float
    exposure_percentage: float
    correlation_risk: float
    risk_level: RiskLevel

@dataclass
class PositionUpdate:
    ticket: int
    action: str  # "modify_sl", "modify_tp", "close", "partial_close"
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    close_volume: Optional[float] = None
    reason: str = ""

class OrderManagementSystem:
    def __init__(self, config):
        self.config = config
        self.active_orders: Dict[int, OrderInfo] = {}
        self.closed_orders: List[OrderInfo] = []
        self.risk_metrics = RiskMetrics(
            current_drawdown=0.0,
            max_drawdown=0.0,
            profit_factor=1.0,
            win_rate=0.0,
            risk_reward_ratio=1.0,
            exposure_percentage=0.0,
            correlation_risk=0.0,
            risk_level=RiskLevel.LOW
        )
        self.last_update = datetime.now()
        self.daily_stats = {
            'trades_opened': 0,
            'trades_closed': 0,
            'profit_loss': 0.0,
            'max_concurrent_trades': 0
        }
        
    def update_positions(self) -> bool:
        """Update all position information from MT5"""
        try:
            # Get all open positions
            positions = mt5.positions_get()
            if positions is None:
                logger.warning("Failed to get positions from MT5")
                return False
            
            current_tickets = set()
            
            # Update existing positions and add new ones
            for pos in positions:
                ticket = pos.ticket
                current_tickets.add(ticket)
                
                order_info = OrderInfo(
                    ticket=ticket,
                    symbol=pos.symbol,
                    order_type="buy" if pos.type == 0 else "sell",
                    volume=pos.volume,
                    open_price=pos.price_open,
                    stop_loss=pos.sl,
                    take_profit=pos.tp,
                    open_time=datetime.fromtimestamp(pos.time),
                    status=OrderStatus.ACTIVE,
                    profit=pos.profit,
                    swap=pos.swap,
                    commission=getattr(pos, 'commission', 0.0),
                    comment=pos.comment,
                    magic_number=pos.magic
                )
                
                # Check if this is a new position
                if ticket not in self.active_orders:
                    self.active_orders[ticket] = order_info
                    self.daily_stats['trades_opened'] += 1
                    logger.info(f"📈 New position tracked: {pos.symbol} {order_info.order_type.upper()} #{ticket}")
                else:
                    # Update existing position
                    self.active_orders[ticket] = order_info
            
            # Handle closed positions
            closed_tickets = set(self.active_orders.keys()) - current_tickets
            for ticket in closed_tickets:
                closed_order = self.active_orders.pop(ticket)
                closed_order.status = OrderStatus.CLOSED
                self.closed_orders.append(closed_order)
                self.daily_stats['trades_closed'] += 1
                logger.info(f"📉 Position closed: {closed_order.symbol} #{ticket} P&L: {closed_order.profit:.2f}")
            
            # Update daily stats
            self.daily_stats['max_concurrent_trades'] = max(
                self.daily_stats['max_concurrent_trades'],
                len(self.active_orders)
            )
            
            self.last_update = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            return False
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return self.risk_metrics
            
            balance = account_info.balance
            equity = account_info.equity
            margin = account_info.margin
            
            # Calculate current drawdown
            current_drawdown = max(0, (balance - equity) / balance * 100) if balance > 0 else 0
            
            # Update max drawdown
            self.risk_metrics.max_drawdown = max(self.risk_metrics.max_drawdown, current_drawdown)
            
            # Calculate exposure percentage
            total_volume = sum(order.volume for order in self.active_orders.values())
            exposure_percentage = (margin / equity * 100) if equity > 0 else 0
            
            # Calculate win rate and profit factor from closed trades
            if self.closed_orders:
                winning_trades = [order for order in self.closed_orders if order.profit > 0]
                losing_trades = [order for order in self.closed_orders if order.profit < 0]
                
                win_rate = len(winning_trades) / len(self.closed_orders) * 100
                
                total_wins = sum(order.profit for order in winning_trades)
                total_losses = abs(sum(order.profit for order in losing_trades))
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            else:
                win_rate = 0
                profit_factor = 1.0
            
            # Calculate correlation risk
            correlation_risk = self._calculate_correlation_risk()
            
            # Determine risk level
            risk_level = self._determine_risk_level(
                current_drawdown, exposure_percentage, correlation_risk
            )
            
            self.risk_metrics = RiskMetrics(
                current_drawdown=current_drawdown,
                max_drawdown=self.risk_metrics.max_drawdown,
                profit_factor=profit_factor,
                win_rate=win_rate,
                risk_reward_ratio=1.0,  # Will be calculated separately
                exposure_percentage=exposure_percentage,
                correlation_risk=correlation_risk,
                risk_level=risk_level
            )
            
            return self.risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return self.risk_metrics
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk between open positions"""
        if len(self.active_orders) < 2:
            return 0.0
        
        # Simple correlation risk based on currency exposure
        currency_exposure = {}
        
        for order in self.active_orders.values():
            symbol = order.symbol
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
            
            # Calculate exposure based on position direction
            if order.order_type == "buy":
                currency_exposure[base_currency] = currency_exposure.get(base_currency, 0) + order.volume
                currency_exposure[quote_currency] = currency_exposure.get(quote_currency, 0) - order.volume
            else:
                currency_exposure[base_currency] = currency_exposure.get(base_currency, 0) - order.volume
                currency_exposure[quote_currency] = currency_exposure.get(quote_currency, 0) + order.volume
        
        # Calculate correlation risk as the sum of absolute exposures
        total_exposure = sum(abs(exposure) for exposure in currency_exposure.values())
        max_single_exposure = max(abs(exposure) for exposure in currency_exposure.values()) if currency_exposure else 0
        
        correlation_risk = (max_single_exposure / total_exposure * 100) if total_exposure > 0 else 0
        return min(correlation_risk, 100.0)
    
    def _determine_risk_level(self, drawdown: float, exposure: float, correlation: float) -> RiskLevel:
        """Determine overall risk level based on metrics"""
        if drawdown > 15 or exposure > 80 or correlation > 70:
            return RiskLevel.CRITICAL
        elif drawdown > 10 or exposure > 60 or correlation > 50:
            return RiskLevel.HIGH
        elif drawdown > 5 or exposure > 40 or correlation > 30:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def apply_risk_management(self) -> List[PositionUpdate]:
        """Apply risk management rules and return position updates"""
        updates = []
        
        try:
            # Update risk metrics first
            self.calculate_risk_metrics()
            
            # Apply different risk management based on risk level
            if self.risk_metrics.risk_level == RiskLevel.CRITICAL:
                updates.extend(self._apply_critical_risk_management())
            elif self.risk_metrics.risk_level == RiskLevel.HIGH:
                updates.extend(self._apply_high_risk_management())
            elif self.risk_metrics.risk_level == RiskLevel.MEDIUM:
                updates.extend(self._apply_medium_risk_management())
            else:
                updates.extend(self._apply_normal_risk_management())
            
            # Apply trailing stops
            updates.extend(self._apply_trailing_stops())
            
            # Apply breakeven stops
            updates.extend(self._apply_breakeven_stops())
            
            return updates
            
        except Exception as e:
            logger.error(f"Error applying risk management: {e}")
            return []
    
    def _apply_critical_risk_management(self) -> List[PositionUpdate]:
        """Apply emergency risk management - close losing positions"""
        updates = []
        
        # Close all losing positions immediately
        for order in self.active_orders.values():
            if order.profit < -50:  # Close positions losing more than $50
                updates.append(PositionUpdate(
                    ticket=order.ticket,
                    action="close",
                    reason="Critical risk - emergency close"
                ))
        
        logger.warning(f"🚨 CRITICAL RISK LEVEL - Closing {len(updates)} losing positions")
        return updates
    
    def _apply_high_risk_management(self) -> List[PositionUpdate]:
        """Apply aggressive risk management"""
        updates = []
        
        # Tighten stop losses
        for order in self.active_orders.values():
            if order.profit < -30:  # Tighten SL for positions losing more than $30
                new_sl = self._calculate_tighter_stop_loss(order)
                if new_sl != order.stop_loss:
                    updates.append(PositionUpdate(
                        ticket=order.ticket,
                        action="modify_sl",
                        new_sl=new_sl,
                        reason="High risk - tightening stop loss"
                    ))
        
        return updates
    
    def _apply_medium_risk_management(self) -> List[PositionUpdate]:
        """Apply moderate risk management"""
        updates = []
        
        # Apply partial profit taking
        for order in self.active_orders.values():
            if order.profit > 50:  # Take partial profits on positions with $50+ profit
                partial_volume = order.volume * 0.5
                updates.append(PositionUpdate(
                    ticket=order.ticket,
                    action="partial_close",
                    close_volume=partial_volume,
                    reason="Medium risk - partial profit taking"
                ))
        
        return updates
    
    def _apply_normal_risk_management(self) -> List[PositionUpdate]:
        """Apply standard risk management"""
        updates = []
        
        # Standard trailing stops and breakeven management
        # This will be handled by separate methods
        
        return updates
    
    def _apply_trailing_stops(self) -> List[PositionUpdate]:
        """Apply trailing stop logic"""
        updates = []
        
        for order in self.active_orders.values():
            if order.profit > 20:  # Only apply trailing stops on profitable positions
                new_sl = self._calculate_trailing_stop(order)
                if new_sl != order.stop_loss:
                    updates.append(PositionUpdate(
                        ticket=order.ticket,
                        action="modify_sl",
                        new_sl=new_sl,
                        reason="Trailing stop adjustment"
                    ))
        
        return updates
    
    def _apply_breakeven_stops(self) -> List[PositionUpdate]:
        """Apply breakeven stop logic"""
        updates = []
        
        for order in self.active_orders.values():
            if order.profit > 10 and order.stop_loss != order.open_price:
                # Move stop loss to breakeven
                updates.append(PositionUpdate(
                    ticket=order.ticket,
                    action="modify_sl",
                    new_sl=order.open_price,
                    reason="Breakeven stop"
                ))
        
        return updates
    
    def _calculate_trailing_stop(self, order: OrderInfo) -> float:
        """Calculate trailing stop level with dynamic distances"""
        try:
            symbol_info = mt5.symbol_info(order.symbol)
            if not symbol_info:
                return order.stop_loss
            
            current_price = symbol_info.bid if order.order_type == "buy" else symbol_info.ask
            
            # Use dynamic trailing distance based on symbol
            trailing_pips = self._get_dynamic_trailing_distance(order.symbol)
            trailing_distance = trailing_pips * symbol_info.point
            
            # For JPY pairs, adjust the calculation
            if 'JPY' in order.symbol:
                trailing_distance = trailing_pips * 0.01
            else:
                trailing_distance = trailing_pips * 0.0001
            
            if order.order_type == "buy":
                new_sl = current_price - trailing_distance
                return max(new_sl, order.stop_loss)  # Only move SL up
            else:
                new_sl = current_price + trailing_distance
                return min(new_sl, order.stop_loss)  # Only move SL down
                
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {e}")
            return order.stop_loss
    
    def _calculate_tighter_stop_loss(self, order: OrderInfo) -> float:
        """Calculate a tighter stop loss for risk management with reasonable distances"""
        try:
            symbol_info = mt5.symbol_info(order.symbol)
            if not symbol_info:
                return order.stop_loss
            
            current_price = symbol_info.bid if order.order_type == "buy" else symbol_info.ask
            
            # Use dynamic tight stop distance based on symbol (75% of normal trailing distance)
            tight_pips = int(self._get_dynamic_trailing_distance(order.symbol) * 0.75)
            
            # For JPY pairs, adjust the calculation
            if 'JPY' in order.symbol:
                tight_distance = tight_pips * 0.01
            else:
                tight_distance = tight_pips * 0.0001
            
            if order.order_type == "buy":
                new_tight_sl = current_price - tight_distance
                # Only tighten if it's better than current SL
                return max(new_tight_sl, order.stop_loss)
            else:
                new_tight_sl = current_price + tight_distance
                # Only tighten if it's better than current SL
                return min(new_tight_sl, order.stop_loss)
                
        except Exception as e:
            logger.error(f"Error calculating tighter stop loss: {e}")
            return order.stop_loss
    
    def _get_dynamic_trailing_distance(self, symbol: str) -> int:
        """Get dynamic trailing distance in pips based on symbol characteristics"""
        try:
            # Base trailing distances by currency pair type (more conservative than initial SL)
            if 'JPY' in symbol:
                # JPY pairs - use smaller trailing distances
                base_pips = 25
            elif any(major in symbol for major in ['EUR', 'GBP', 'USD']):
                # Major pairs - moderate trailing
                base_pips = 30
            elif any(minor in symbol for minor in ['AUD', 'NZD', 'CAD']):
                # Commodity currencies - wider trailing
                base_pips = 35
            elif 'CHF' in symbol:
                # Swiss Franc - conservative trailing
                base_pips = 28
            else:
                # Default for exotic pairs
                base_pips = 40
            
            # Specific adjustments for high volatility pairs
            volatility_adjustments = {
                'GBPJPY': 45,  # Very volatile
                'EURJPY': 35,  # Moderate volatility
                'USDJPY': 30,  # Lower volatility
                'GBPUSD': 35,  # Cable volatility
                'EURUSD': 25,  # Most liquid
                'USDCHF': 30,  # Moderate volatility
                'AUDUSD': 35,  # Commodity currency
                'NZDUSD': 40,  # Higher volatility
                'GBPAUD': 45,  # Cross pair volatility
                'EURNZD': 40,  # Cross pair
                'AUDCAD': 38,  # Commodity cross
                'CHFJPY': 35,  # Safe haven cross
            }
            
            # Use specific adjustment if available
            final_pips = volatility_adjustments.get(symbol, base_pips)
            
            # Ensure minimum trailing distance
            return max(final_pips, 20)
            
        except Exception as e:
            logger.error(f"Error calculating dynamic trailing distance for {symbol}: {e}")
            return 30  # Safe default
    
    def execute_position_updates(self, updates: List[PositionUpdate]) -> bool:
        """Execute position updates in MT5"""
        success_count = 0
        
        for update in updates:
            try:
                if update.action == "close":
                    success = self._close_position(update.ticket)
                elif update.action == "partial_close":
                    success = self._partial_close_position(update.ticket, update.close_volume)
                elif update.action == "modify_sl":
                    success = self._modify_stop_loss(update.ticket, update.new_sl)
                elif update.action == "modify_tp":
                    success = self._modify_take_profit(update.ticket, update.new_tp)
                else:
                    logger.warning(f"Unknown update action: {update.action}")
                    continue
                
                if success:
                    success_count += 1
                    logger.info(f"✅ Position update executed: {update.action} #{update.ticket} - {update.reason}")
                else:
                    logger.error(f"❌ Failed to execute update: {update.action} #{update.ticket}")
                    
            except Exception as e:
                logger.error(f"Error executing position update: {e}")
        
        logger.info(f"📊 Position updates: {success_count}/{len(updates)} successful")
        return success_count == len(updates)
    
    def _close_position(self, ticket: int) -> bool:
        """Close a position completely"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            pos = position[0]
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "deviation": 20,
                "magic": pos.magic,
                "comment": "Risk management close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(close_request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def _partial_close_position(self, ticket: int, volume: float) -> bool:
        """Partially close a position"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            pos = position[0]
            if volume >= pos.volume:
                return self._close_position(ticket)
            
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "deviation": 20,
                "magic": pos.magic,
                "comment": "Partial close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(close_request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            logger.error(f"Error partially closing position {ticket}: {e}")
            return False
    
    def _modify_stop_loss(self, ticket: int, new_sl: float) -> bool:
        """Modify stop loss of a position"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            pos = position[0]
            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": pos.tp,
                "magic": pos.magic,
                "comment": "SL modification",
            }
            
            result = mt5.order_send(modify_request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            logger.error(f"Error modifying stop loss {ticket}: {e}")
            return False
    
    def _modify_take_profit(self, ticket: int, new_tp: float) -> bool:
        """Modify take profit of a position"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            pos = position[0]
            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": ticket,
                "sl": pos.sl,
                "tp": new_tp,
                "magic": pos.magic,
                "comment": "TP modification",
            }
            
            result = mt5.order_send(modify_request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            logger.error(f"Error modifying take profit {ticket}: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return {}
            
            total_profit = sum(order.profit for order in self.active_orders.values())
            total_volume = sum(order.volume for order in self.active_orders.values())
            
            return {
                'account_balance': account_info.balance,
                'account_equity': account_info.equity,
                'account_margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'active_positions': len(self.active_orders),
                'total_volume': total_volume,
                'unrealized_pnl': total_profit,
                'daily_stats': self.daily_stats.copy(),
                'risk_metrics': asdict(self.risk_metrics),
                'last_update': self.last_update.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def check_duplicate_trade(self, symbol: str, direction: str) -> bool:
        """Check if there's already an open position in the same direction for the symbol"""
        try:
            for order in self.active_orders.values():
                if order.symbol == symbol and order.order_type.lower() == direction.lower():
                    logger.warning(f"🚫 Duplicate trade prevention: {symbol} {direction.upper()} already open (#{order.ticket})")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking duplicate trade: {e}")
            return False
    
    def get_position_count_by_symbol(self, symbol: str) -> Dict[str, int]:
        """Get count of positions by direction for a symbol"""
        try:
            counts = {'buy': 0, 'sell': 0}
            for order in self.active_orders.values():
                if order.symbol == symbol:
                    counts[order.order_type] += 1
            return counts
        except Exception as e:
            logger.error(f"Error getting position count: {e}")
            return {'buy': 0, 'sell': 0}
    
    def should_allow_trade(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """Determine if a trade should be allowed based on existing positions"""
        try:
            # Check for duplicate trades in same direction
            if self.check_duplicate_trade(symbol, direction):
                return False, f"Already have {direction.upper()} position for {symbol}"
            
            # Check position counts
            counts = self.get_position_count_by_symbol(symbol)
            total_positions = counts['buy'] + counts['sell']
            
            # Limit to maximum 2 positions per symbol (1 buy + 1 sell max)
            if total_positions >= 2:
                return False, f"Maximum positions reached for {symbol} (Buy: {counts['buy']}, Sell: {counts['sell']})"
            
            # Check if we have opposite direction position
            opposite_direction = 'sell' if direction.lower() == 'buy' else 'buy'
            if counts[opposite_direction] > 0:
                logger.info(f"⚠️ Opening {direction.upper()} while {opposite_direction.upper()} position exists for {symbol}")
            
            return True, "Trade allowed"
            
        except Exception as e:
            logger.error(f"Error checking if trade should be allowed: {e}")
            return True, "Error in validation - allowing trade"
    
    def print_status_report(self):
        """Print comprehensive status report"""
        try:
            summary = self.get_portfolio_summary()
            
            print("\n" + "="*60)
            print("📊 ORDER MANAGEMENT SYSTEM STATUS REPORT")
            print("="*60)
            
            # Account Information
            print(f"💰 Account Balance: ${summary.get('account_balance', 0):.2f}")
            print(f"💎 Account Equity: ${summary.get('account_equity', 0):.2f}")
            print(f"📈 Unrealized P&L: ${summary.get('unrealized_pnl', 0):.2f}")
            print(f"🎯 Margin Level: {summary.get('margin_level', 0):.1f}%")
            
            # Position Information
            print(f"\n📋 Active Positions: {summary.get('active_positions', 0)}")
            print(f"📊 Total Volume: {summary.get('total_volume', 0):.2f} lots")
            
            # Risk Metrics
            risk = summary.get('risk_metrics', {})
            print(f"\n⚠️  Risk Level: {risk.get('risk_level', 'Unknown').upper()}")
            print(f"📉 Current Drawdown: {risk.get('current_drawdown', 0):.2f}%")
            print(f"📊 Exposure: {risk.get('exposure_percentage', 0):.1f}%")
            print(f"🔗 Correlation Risk: {risk.get('correlation_risk', 0):.1f}%")
            print(f"🎯 Win Rate: {risk.get('win_rate', 0):.1f}%")
            
            # Daily Statistics
            daily = summary.get('daily_stats', {})
            print(f"\n📅 Today's Statistics:")
            print(f"   Trades Opened: {daily.get('trades_opened', 0)}")
            print(f"   Trades Closed: {daily.get('trades_closed', 0)}")
            print(f"   Max Concurrent: {daily.get('max_concurrent_trades', 0)}")
            
            # Active Positions Detail
            if self.active_orders:
                print(f"\n🔍 Active Positions Detail:")
                symbol_groups = {}
                for order in self.active_orders.values():
                    if order.symbol not in symbol_groups:
                        symbol_groups[order.symbol] = []
                    symbol_groups[order.symbol].append(order)
                
                for symbol, orders in symbol_groups.items():
                    print(f"\n   📈 {symbol}:")
                    for order in orders:
                        profit_emoji = "🟢" if order.profit >= 0 else "🔴"
                        print(f"     {profit_emoji} #{order.ticket} {order.order_type.upper()} "
                              f"{order.volume} lots | P&L: ${order.profit:.2f}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error printing status report: {e}")