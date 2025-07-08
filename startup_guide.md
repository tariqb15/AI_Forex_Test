# BLOB AI Startup & Continuous Running Guide

## 🚀 Quick Start Options

### Option 1: Interactive Control Panel
```bash
python start_blob_ai.py
```
**Features:**
- Interactive menu system
- Start/stop individual services
- View logs and status
- Manual control over all components

### Option 2: Direct Engine with Automated Analysis
```bash
python forex_engine.py
```
**When prompted "Start automated analysis? (y/n):", type `y`**

**Features:**
- Continuous market analysis
- Real-time signal generation
- Automated trading decisions
- Runs until manually stopped

### Option 3: Background Service Mode
```bash
# Windows PowerShell
Start-Process python -ArgumentList "forex_engine.py" -WindowStyle Hidden

# Or using nohup equivalent
python forex_engine.py > forex_output.log 2>&1 &
```

## 🔄 Continuous Running Modes

### 1. Automated Analysis Mode
When you run `forex_engine.py` and choose "y" for automated analysis:

- **Analysis Frequency**: Every 5 minutes (configurable)
- **Market Hours**: 24/7 monitoring
- **Signal Generation**: Real-time when conditions are met
- **Data Collection**: Continuous multi-timeframe updates
- **Logging**: All activities logged to `forex_engine.log`

### 2. Service-Based Running
Using the control panel (`start_blob_ai.py`):

1. **Start All Services** (Option 1)
   - Forex Engine
   - API Server (port 8000)
   - Dashboard (port 8501)

2. **Individual Service Control**
   - Start specific components
   - Monitor each service separately
   - Restart failed services

## ⚙️ Configuration for Continuous Operation

### Analysis Interval Settings
Edit `config.py` to adjust timing:

```python
class ForexEngineConfig:
    # Analysis frequency (seconds)
    analysis_interval = 300  # 5 minutes
    
    # Market hours (UTC)
    market_start_hour = 0    # Sunday 00:00 UTC
    market_end_hour = 24     # Friday 24:00 UTC
    
    # Auto-restart on errors
    auto_restart = True
    max_restart_attempts = 5
```

### Logging Configuration
```python
# Enable detailed logging
logging_level = "INFO"  # DEBUG, INFO, WARNING, ERROR
log_to_file = True
log_rotation = True  # Rotate logs daily
```

## 🖥️ Running as Windows Service

### Method 1: Using Task Scheduler
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: "At startup"
4. Action: Start program
5. Program: `python.exe`
6. Arguments: `C:\path\to\forex_engine.py`
7. Start in: `C:\path\to\BLOB AI\FXOpen`

### Method 2: Using NSSM (Non-Sucking Service Manager)
```bash
# Download NSSM and install
nssm install "BLOB AI Forex Engine"
# Path: C:\Python\python.exe
# Arguments: C:\path\to\forex_engine.py
# Startup directory: C:\path\to\BLOB AI\FXOpen
```

## 📊 Monitoring Continuous Operation

### 1. Log Files
```bash
# Real-time log monitoring
tail -f forex_engine.log

# Or on Windows PowerShell
Get-Content forex_engine.log -Wait
```

### 2. Web Dashboard
Access at: `http://localhost:8501`
- Real-time market data
- Signal history
- Performance metrics
- System status

### 3. API Endpoints
Access at: `http://localhost:8000`
- `/status` - System health
- `/signals` - Current signals
- `/analysis` - Latest analysis
- `/positions` - Open positions

## 🛡️ Error Handling & Recovery

### Automatic Restart Features
- **MT5 Connection Loss**: Auto-reconnect every 30 seconds
- **Analysis Errors**: Skip failed analysis, continue with next
- **Memory Issues**: Garbage collection after each analysis
- **Network Issues**: Retry mechanism with exponential backoff

### Manual Recovery Commands
```bash
# Check if process is running
tasklist | findstr python

# Kill stuck processes
taskkill /f /im python.exe

# Restart system
python start_blob_ai.py
```

## 🔧 Optimization for 24/7 Operation

### 1. Resource Management
```python
# In config.py
max_memory_usage = 512  # MB
data_retention_days = 7
auto_cleanup = True
```

### 2. Performance Settings
```python
# Reduce analysis frequency during low volatility
adaptive_analysis = True
min_interval = 60    # 1 minute during high volatility
max_interval = 900   # 15 minutes during low volatility
```

### 3. Network Optimization
```python
# Connection pooling
mt5_connection_pool = True
max_reconnect_attempts = 10
connection_timeout = 30
```

## 📋 Startup Checklist

### Before Starting Continuous Operation:
- [ ] MT5 terminal is running and logged in
- [ ] Demo account is active and funded
- [ ] Internet connection is stable
- [ ] All dependencies are installed
- [ ] Configuration is properly set
- [ ] Log directory has write permissions
- [ ] Firewall allows Python network access

### Recommended Startup Sequence:
1. **Start MT5 Terminal** and login
2. **Test Connection**: `python test_mt5_connection.py`
3. **Start BLOB AI**: `python forex_engine.py`
4. **Enable Automated Analysis**: Type `y` when prompted
5. **Monitor Initial Analysis**: Check logs for errors
6. **Verify Signal Generation**: Wait for market conditions

## 🚨 Emergency Stop Procedures

### Immediate Stop
```bash
# Ctrl+C in terminal
# Or kill process
taskkill /f /pid <process_id>
```

### Graceful Shutdown
- Use control panel option "Stop All Services"
- Wait for current analysis to complete
- Verify all positions are properly managed

## 📈 Performance Monitoring

### Key Metrics to Watch:
- **Analysis Speed**: Should complete in <1 second
- **Memory Usage**: Should stay under 500MB
- **Signal Quality**: Confidence scores >70%
- **Connection Stability**: <1% disconnection rate
- **Error Rate**: <0.1% analysis failures

### Alert Thresholds:
- Memory usage >80%
- Analysis time >5 seconds
- Connection failures >5 in 1 hour
- No signals generated for >24 hours (during active markets)

---

## 🎯 Recommended Setup for Continuous Trading

**For 24/7 automated operation:**

1. **Use automated analysis mode** (`python forex_engine.py` → `y`)
2. **Enable all logging** for monitoring
3. **Set up Windows service** for auto-start
4. **Monitor via dashboard** at `localhost:8501`
5. **Check logs daily** for any issues
6. **Review performance weekly** using analysis reports

This setup will provide continuous, autonomous forex trading with the BLOB AI system!