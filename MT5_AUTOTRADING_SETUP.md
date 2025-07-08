# MT5 AutoTrading Setup Guide

## Error: AutoTrading disabled by client (Code: 10027)

This error occurs when MetaTrader 5's AutoTrading feature is disabled. Here's how to fix it:

## 🔧 Step-by-Step Solution

### 1. Enable AutoTrading in MT5 Terminal

**Method 1: Using the Toolbar**
- Open your MetaTrader 5 terminal
- Look for the "AutoTrading" button in the toolbar (looks like a robot icon)
- Click the AutoTrading button to enable it
- The button should turn green when enabled

**Method 2: Using the Menu**
- Go to `Tools` → `Options`
- Click on the `Expert Advisors` tab
- Check the box "Allow automated trading"
- Check the box "Allow DLL imports"
- Click `OK`

### 2. Verify Expert Advisor Settings

1. In MT5, go to `Tools` → `Options` → `Expert Advisors`
2. Ensure these settings are enabled:
   - ✅ Allow automated trading
   - ✅ Allow DLL imports
   - ✅ Allow imports of external experts
   - ✅ Enable WebRequest for listed URLs (if using web features)

### 3. Check Account Permissions

- Verify your trading account allows automated trading
- Some demo accounts may have restrictions
- Contact your broker if automated trading is not allowed

### 4. Restart MT5 and Python Script

1. Close MetaTrader 5 completely
2. Restart MT5
3. Ensure AutoTrading is enabled (green button)
4. Run the automated trading script again

## 🚀 Quick Test

Run a single trading cycle to verify the fix:

```bash
python start_automated_trading.py --mode single --demo
 // use this
```

## 🔍 Troubleshooting

### If AutoTrading is still disabled:

1. **Check MT5 Version**: Ensure you're using a recent version of MT5
2. **Account Type**: Verify your account supports automated trading
3. **Broker Settings**: Some brokers disable EA trading by default
4. **Firewall/Antivirus**: Ensure MT5 and Python are not blocked

### Common Issues:

**Issue**: AutoTrading button keeps turning off
- **Solution**: Check if an EA is already running that disables AutoTrading

**Issue**: "Trade is disabled" error
- **Solution**: Check market hours and ensure the symbol is tradeable

**Issue**: "Not enough money" error
- **Solution**: Verify account balance and reduce lot size in config.py

## ⚙️ Configuration Verification

Ensure your `config.py` has correct settings:

```python
# In config.py, verify these settings:
class TradingConfig:
    lot_size: float = 0.01  # Start with small lot size
    max_spread_pips: float = 3.0
    slippage: int = 3
    magic_number: int = 12345  # Unique number for your EA
```

## 📊 Monitoring

Once AutoTrading is enabled:

1. **Check Logs**: Monitor `forex_engine.log` for any errors
2. **MT5 Journal**: Check MT5's Journal tab for trading activity
3. **Positions**: Monitor the Positions tab in MT5

## 🛡️ Safety Features

The automated trader includes these safety features:
- Stop Loss on every trade
- Take Profit targets
- Maximum position limits
- Risk management per trade
- Real-time monitoring

## 📞 Support

If you continue to experience issues:

1. Check MT5 Journal tab for detailed error messages
2. Verify broker allows automated trading
3. Ensure account has sufficient balance
4. Contact broker support if needed

---

**⚠️ Risk Warning**: Automated trading involves substantial risk. Always test with demo accounts first and never risk more than you can afford to lose.