import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def check_open_positions():
    """Check all currently open positions in MT5"""
    
    print("🔄 Starting position check...")
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"❌ Failed to initialize MT5: {mt5.last_error()}")
        return
    
    print("✅ MT5 initialized successfully")
    
    try:
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"📊 Account: {account_info.login} | Balance: {account_info.balance} | Equity: {account_info.equity}")
        
        # Get all open positions
        positions = mt5.positions_get()
        
        if positions is None:
            print("❌ Failed to get positions")
            return
        
        if len(positions) == 0:
            print("📭 No open positions")
        else:
            print(f"\n📈 Found {len(positions)} open position(s):")
            print("-" * 80)
            
            for pos in positions:
                profit_color = "🟢" if pos.profit >= 0 else "🔴"
                print(f"{profit_color} Ticket: {pos.ticket}")
                print(f"   Symbol: {pos.symbol}")
                print(f"   Type: {'BUY' if pos.type == 0 else 'SELL'}")
                print(f"   Volume: {pos.volume}")
                print(f"   Open Price: {pos.price_open}")
                print(f"   Current Price: {pos.price_current}")
                print(f"   Profit: {pos.profit:.2f}")
                print(f"   Open Time: {datetime.fromtimestamp(pos.time)}")
                print(f"   Comment: {pos.comment}")
                print("-" * 40)
        
        # Get recent trades (last 10)
        print("\n📋 Recent trades (last 10):")
        print("-" * 80)
        
        from datetime import datetime, timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)  # Last hour
        
        deals = mt5.history_deals_get(start_time, end_time)
        
        if deals:
            recent_deals = sorted(deals, key=lambda x: x.time, reverse=True)[:10]
            for deal in recent_deals:
                deal_type = "BUY" if deal.type == 0 else "SELL" if deal.type == 1 else "OTHER"
                print(f"🔄 Ticket: {deal.ticket} | {deal.symbol} {deal_type} | Volume: {deal.volume} | Price: {deal.price} | Time: {datetime.fromtimestamp(deal.time)}")
        else:
            print("No recent deals found")
    
    except Exception as e:
        print(f"❌ Error during position check: {e}")
    finally:
        mt5.shutdown()
        print("\n✅ MT5 connection closed")

if __name__ == "__main__":
    check_open_positions()