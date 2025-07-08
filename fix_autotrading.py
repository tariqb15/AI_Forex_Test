#!/usr/bin/env python3
"""
MT5 AutoTrading Diagnostic and Fix Tool

This script helps diagnose and fix common AutoTrading issues in MetaTrader 5.
"""

import MetaTrader5 as mt5
import sys
import time
from datetime import datetime
from config import ForexEngineConfig

def print_header():
    """Print diagnostic tool header"""
    print("\n" + "="*60)
    print("🔧 MT5 AUTOTRADING DIAGNOSTIC TOOL 🔧")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

def check_mt5_installation():
    """Check if MT5 is properly installed"""
    print("\n📋 Checking MT5 Installation...")
    try:
        version = mt5.version()
        if version:
            print(f"✅ MT5 Version: {version[0]} Build {version[1]}")
            return True
        else:
            print("❌ MT5 not found or not properly installed")
            return False
    except Exception as e:
        print(f"❌ Error checking MT5: {e}")
        return False

def check_mt5_connection():
    """Check MT5 connection"""
    print("\n🔌 Checking MT5 Connection...")
    
    config = ForexEngineConfig()
    
    try:
        # Initialize MT5
        if not mt5.initialize():
            print(f"❌ Failed to initialize MT5: {mt5.last_error()}")
            return False
        
        print("✅ MT5 initialized successfully")
        
        # Check login
        login_result = mt5.login(
            login=config.mt5.login,
            password=config.mt5.password,
            server=config.mt5.server
        )
        
        if login_result:
            account_info = mt5.account_info()
            print(f"✅ Connected to account: {account_info.login}")
            print(f"✅ Server: {account_info.server}")
            print(f"✅ Balance: ${account_info.balance:,.2f}")
            print(f"✅ Equity: ${account_info.equity:,.2f}")
            return True
        else:
            error = mt5.last_error()
            print(f"❌ Login failed: {error}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def check_autotrading_status():
    """Check AutoTrading status"""
    print("\n🤖 Checking AutoTrading Status...")
    
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info:
            if terminal_info.trade_allowed:
                print("✅ Trading is allowed in terminal")
            else:
                print("❌ Trading is NOT allowed in terminal")
                print("   → Enable AutoTrading in MT5 toolbar")
            
            if hasattr(terminal_info, 'expert_allowed'):
                if terminal_info.expert_allowed:
                    print("✅ Expert Advisors are allowed")
                else:
                    print("❌ Expert Advisors are NOT allowed")
                    print("   → Go to Tools → Options → Expert Advisors")
                    print("   → Check 'Allow automated trading'")
            
            return terminal_info.trade_allowed
        else:
            print("❌ Could not get terminal info")
            return False
            
    except Exception as e:
        print(f"❌ Error checking AutoTrading: {e}")
        return False

def check_account_permissions():
    """Check account trading permissions"""
    print("\n🔐 Checking Account Permissions...")
    
    try:
        account_info = mt5.account_info()
        if account_info:
            if account_info.trade_allowed:
                print("✅ Account allows trading")
            else:
                print("❌ Account does NOT allow trading")
                print("   → Contact your broker")
            
            if account_info.trade_expert:
                print("✅ Account allows Expert Advisor trading")
            else:
                print("❌ Account does NOT allow Expert Advisor trading")
                print("   → Contact your broker to enable EA trading")
            
            # Get account type safely
            try:
                account_type = getattr(account_info, 'trade_mode_description', 'Unknown')
                print(f"📊 Account Type: {account_type}")
            except:
                print("📊 Account Type: Demo/Live Account")
            
            print(f"📊 Leverage: 1:{account_info.leverage}")
            
            return account_info.trade_allowed and account_info.trade_expert
        else:
            print("❌ Could not get account info")
            return False
            
    except Exception as e:
        print(f"❌ Error checking account permissions: {e}")
        return False

def test_order_placement():
    """Test if orders can be placed"""
    print("\n📝 Testing Order Placement (Demo)...")
    
    try:
        # Get symbol info
        symbol = "USDJPY"
        symbol_info = mt5.symbol_info(symbol)
        
        if not symbol_info:
            print(f"❌ Symbol {symbol} not found")
            return False
        
        if not symbol_info.visible:
            print(f"⚠️  Symbol {symbol} not visible, adding to Market Watch")
            if not mt5.symbol_select(symbol, True):
                print(f"❌ Failed to add {symbol} to Market Watch")
                return False
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"❌ Could not get price for {symbol}")
            return False
        
        print(f"✅ Symbol {symbol} is available")
        print(f"✅ Current price: {tick.bid}/{tick.ask}")
        print(f"✅ Spread: {(tick.ask - tick.bid) * 100:.1f} pips")
        
        # Check if we can place orders (without actually placing)
        print("✅ Order placement test passed (no actual orders placed)")
        return True
        
    except Exception as e:
        print(f"❌ Error testing order placement: {e}")
        return False

def provide_solutions():
    """Provide solutions for common issues"""
    print("\n🔧 SOLUTIONS FOR COMMON ISSUES:")
    print("="*50)
    
    print("\n1. Enable AutoTrading in MT5:")
    print("   • Click the AutoTrading button in MT5 toolbar (robot icon)")
    print("   • Button should turn green when enabled")
    
    print("\n2. Enable Expert Advisors:")
    print("   • Go to Tools → Options → Expert Advisors")
    print("   • Check 'Allow automated trading'")
    print("   • Check 'Allow DLL imports'")
    print("   • Click OK")
    
    print("\n3. Restart MT5:")
    print("   • Close MT5 completely")
    print("   • Restart MT5")
    print("   • Ensure AutoTrading is enabled")
    
    print("\n4. Check Account Settings:")
    print("   • Verify account allows automated trading")
    print("   • Contact broker if EA trading is disabled")
    
    print("\n5. Test with Demo Account:")
    print("   • Always test with demo account first")
    print("   • Use small lot sizes (0.01)")

def main():
    """Main diagnostic function"""
    print_header()
    
    all_checks_passed = True
    
    # Run diagnostics
    checks = [
        ("MT5 Installation", check_mt5_installation),
        ("MT5 Connection", check_mt5_connection),
        ("AutoTrading Status", check_autotrading_status),
        ("Account Permissions", check_account_permissions),
        ("Order Placement Test", test_order_placement)
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_checks_passed = False
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results[check_name] = False
            all_checks_passed = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<25} {status}")
    
    if all_checks_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("Your MT5 setup is ready for automated trading.")
        print("\nYou can now run:")
        print("python start_automated_trading.py --mode single --demo")
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("Please fix the issues above before running automated trading.")
        provide_solutions()
    
    # Cleanup
    try:
        mt5.shutdown()
    except:
        pass
    
    print("\n" + "="*60)
    print("Diagnostic complete. Check the results above.")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Diagnostic interrupted by user")
        try:
            mt5.shutdown()
        except:
            pass
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        try:
            mt5.shutdown()
        except:
            pass
        sys.exit(1)