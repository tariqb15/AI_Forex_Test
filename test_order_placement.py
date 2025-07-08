#!/usr/bin/env python3
"""
Test Actual Order Placement
"""

import MetaTrader5 as mt5

def test_order_placement():
    print("=== Order Placement Test ===")
    
    # Initialize MT5
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
    
    try:
        # Check account and terminal info
        account = mt5.account_info()
        terminal = mt5.terminal_info()
        
        print(f"Account: {account.login}")
        print(f"Balance: {account.balance}")
        print(f"Trade allowed (account): {account.trade_allowed}")
        print(f"Trade expert (account): {account.trade_expert}")
        print(f"Terminal connected: {terminal.connected}")
        print(f"Trade allowed (terminal): {terminal.trade_allowed}")
        
        # Print all terminal attributes
        print("\nTerminal attributes:")
        for attr in dir(terminal):
            if not attr.startswith('_'):
                try:
                    value = getattr(terminal, attr)
                    print(f"  {attr}: {value}")
                except:
                    pass
        
        # Test symbol info
        symbol = "EURAUD"
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"Failed to get {symbol} info")
            return
        
        print(f"\nSymbol {symbol}:")
        print(f"  Trade mode: {symbol_info.trade_mode}")
        print(f"  Trade calc mode: {symbol_info.trade_calc_mode}")
        print(f"  Volume min: {symbol_info.volume_min}")
        print(f"  Volume step: {symbol_info.volume_step}")
        print(f"  Stops level: {symbol_info.trade_stops_level}")
        
        # Get current tick
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"Failed to get tick for {symbol}")
            return
        
        print(f"  Current bid: {tick.bid}")
        print(f"  Current ask: {tick.ask}")
        
        # Test a small order (0.01 lots)
        print("\n=== Testing Small Order ===")
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "magic": 123456,
            "comment": "Test order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        print(f"Order request: {request}")
        
        # Send order
        result = mt5.order_send(request)
        print(f"\nOrder result:")
        print(f"  Return code: {result.retcode}")
        print(f"  Comment: {result.comment}")
        print(f"  Order: {result.order}")
        print(f"  Deal: {result.deal}")
        print(f"  Volume: {result.volume}")
        print(f"  Price: {result.price}")
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print("✅ Order executed successfully!")
            
            # Check if position was created
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                print(f"✅ Position created: {len(positions)} positions found")
                for pos in positions:
                    print(f"  Ticket: {pos.ticket}, Volume: {pos.volume}, Price: {pos.price_open}")
            else:
                print("❌ No positions found after order")
        else:
            print(f"❌ Order failed: {result.comment} (Code: {result.retcode})")
            
            # Print common error codes
            error_codes = {
                10004: "TRADE_RETCODE_REQUOTE",
                10006: "TRADE_RETCODE_REJECT",
                10007: "TRADE_RETCODE_CANCEL",
                10008: "TRADE_RETCODE_PLACED",
                10009: "TRADE_RETCODE_DONE",
                10010: "TRADE_RETCODE_DONE_PARTIAL",
                10011: "TRADE_RETCODE_ERROR",
                10012: "TRADE_RETCODE_TIMEOUT",
                10013: "TRADE_RETCODE_INVALID",
                10014: "TRADE_RETCODE_INVALID_VOLUME",
                10015: "TRADE_RETCODE_INVALID_PRICE",
                10016: "TRADE_RETCODE_INVALID_STOPS",
                10017: "TRADE_RETCODE_TRADE_DISABLED",
                10018: "TRADE_RETCODE_MARKET_CLOSED",
                10019: "TRADE_RETCODE_NO_MONEY",
                10020: "TRADE_RETCODE_PRICE_CHANGED",
                10021: "TRADE_RETCODE_PRICE_OFF",
                10022: "TRADE_RETCODE_INVALID_EXPIRATION",
                10023: "TRADE_RETCODE_ORDER_CHANGED",
                10024: "TRADE_RETCODE_TOO_MANY_REQUESTS",
                10025: "TRADE_RETCODE_NO_CHANGES",
                10026: "TRADE_RETCODE_SERVER_DISABLES_AT",
                10027: "TRADE_RETCODE_CLIENT_DISABLES_AT",
                10028: "TRADE_RETCODE_LOCKED",
                10029: "TRADE_RETCODE_FROZEN",
                10030: "TRADE_RETCODE_INVALID_FILL",
                10031: "TRADE_RETCODE_CONNECTION",
                10032: "TRADE_RETCODE_ONLY_REAL",
                10033: "TRADE_RETCODE_LIMIT_ORDERS",
                10034: "TRADE_RETCODE_LIMIT_VOLUME",
                10035: "TRADE_RETCODE_INVALID_ORDER",
                10036: "TRADE_RETCODE_POSITION_CLOSED"
            }
            
            if result.retcode in error_codes:
                print(f"Error meaning: {error_codes[result.retcode]}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    test_order_placement()