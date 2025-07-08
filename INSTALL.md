# BLOB AI Forex Engine - Installation Guide

🚀 **Complete setup guide for the Agentic Forex Analytics Engine**

## 📋 Prerequisites

### 1. System Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB free space
- **Internet**: Stable connection for real-time data

### 2. MetaTrader 5 Terminal
- Download and install [MetaTrader 5](https://www.metatrader5.com/en/download)
- Create a demo account or use your live account
- Ensure MT5 is running and logged in

## 🛠️ Installation Methods

### Method 1: Automated Setup (Recommended)

1. **Clone or Download the Repository**
   ```bash
   git clone <repository-url>
   cd blob-ai-forex-engine
   ```

2. **Run the Setup Script**
   ```bash
   python setup.py
   ```
   
   This will:
   - Check Python version
   - Install all dependencies
   - Create necessary directories
   - Generate configuration files
   - Run system tests

3. **Start the System**
   ```bash
   python start_blob_ai.py start
   ```

### Method 2: Manual Installation

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create Directory Structure**
   ```bash
   mkdir logs data config backups
   ```

3. **Configure the System**
   ```bash
   python config.py
   ```

4. **Test Installation**
   ```bash
   python test_system.py
   ```

## 🚀 Quick Start

### Option 1: Interactive Mode
```bash
python start_blob_ai.py
```
This opens an interactive control panel where you can:
- Start/stop services
- Monitor system status
- View logs
- Run tests

### Option 2: Direct Start
```bash
python start_blob_ai.py start
```
Starts all services and opens the interactive mode.

### Option 3: Headless Mode
```bash
python start_blob_ai.py headless
```
Runs all services in the background without interaction.

## 🌐 Access Points

Once started, you can access:

- **📊 Trading Dashboard**: http://localhost:8501
- **🔌 API Interface**: http://localhost:8000
- **📚 API Documentation**: http://localhost:8000/docs
- **📈 Real-time Signals**: WebSocket at ws://localhost:8000/ws

## ⚙️ Configuration

### Basic Configuration
Edit `config/trading_config.json`:
```json
{
  "mt5": {
    "server": "YourBroker-Demo",
    "login": 12345678,
    "password": "your_password"
  },
  "trading": {
    "symbol": "USDJPY",
    "lot_size": 0.01,
    "max_risk_per_trade": 0.02
  }
}
```

### Advanced Configuration
Use the configuration presets:
```python
from config import PresetConfigs

# Conservative trading
config = PresetConfigs.conservative()

# Aggressive trading
config = PresetConfigs.aggressive()

# Scalping setup
config = PresetConfigs.scalping()
```

## 🧪 Testing

### Run System Tests
```bash
python test_system.py
```

### Run Examples
```bash
python example_usage.py
```

### Test Individual Components
```python
# Test data collection
python -c "from forex_engine import ForexDataCollector; collector = ForexDataCollector(); print('Data collector OK')"

# Test Smart Money analysis
python -c "from forex_engine import SmartMoneyAnalyzer; smc = SmartMoneyAnalyzer(); print('SMC analyzer OK')"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. MetaTrader 5 Connection Failed
**Problem**: `MT5 initialization failed`

**Solutions**:
- Ensure MT5 terminal is running and logged in
- Check if MT5 allows algorithmic trading (Tools → Options → Expert Advisors)
- Verify your broker supports MT5 Python API
- Try restarting MT5 terminal

#### 2. Import Errors
**Problem**: `ModuleNotFoundError: No module named 'MetaTrader5'`

**Solutions**:
```bash
# Reinstall MetaTrader5 package
pip uninstall MetaTrader5
pip install MetaTrader5

# Or install specific version
pip install MetaTrader5==5.0.45
```

#### 3. Port Already in Use
**Problem**: `Port 8501 is already in use`

**Solutions**:
```bash
# Find and kill process using the port
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8501 | xargs kill -9
```

#### 4. Dashboard Not Loading
**Problem**: Dashboard shows blank page

**Solutions**:
- Clear browser cache
- Try incognito/private mode
- Check if Streamlit is properly installed
- Restart the dashboard service

#### 5. No Trading Signals
**Problem**: System runs but generates no signals

**Solutions**:
- Check if market is open (Forex trades 24/5)
- Verify MT5 data feed is active
- Ensure sufficient historical data (at least 1000 bars)
- Check configuration settings

### Log Files
Check these log files for detailed error information:
- `logs/startup.log` - System startup logs
- `logs/forex_engine/stdout.log` - Main engine logs
- `logs/dashboard/stdout.log` - Dashboard logs
- `logs/api_server/stdout.log` - API server logs

### Performance Issues

#### High CPU Usage
- Reduce data collection frequency in config
- Disable unnecessary timeframes
- Increase analysis intervals

#### High Memory Usage
- Reduce historical data retention period
- Clear old log files
- Restart services periodically

## 📊 System Monitoring

### Service Status
```bash
python start_blob_ai.py status
```

### Real-time Monitoring
The system includes built-in monitoring that:
- Automatically restarts failed services
- Logs all activities
- Monitors resource usage
- Sends alerts for critical issues

### Health Checks
Access health endpoints:
- **System Health**: http://localhost:8000/health
- **MT5 Connection**: http://localhost:8000/mt5/status
- **Data Feed**: http://localhost:8000/data/status

## 🔒 Security

### API Security
- Generate API keys: http://localhost:8000/auth/generate-key
- Use HTTPS in production
- Regularly rotate API keys
- Monitor API usage logs

### Trading Security
- Never share MT5 credentials
- Use demo accounts for testing
- Set appropriate risk limits
- Monitor all trades

## 📈 Production Deployment

### VPS Setup
1. **Choose a VPS Provider**
   - Recommended: AWS, DigitalOcean, Vultr
   - Location: Close to your broker's servers
   - Specs: 2+ CPU cores, 4GB+ RAM

2. **Install Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip
   pip3 install -r requirements.txt
   ```

3. **Setup as Service**
   ```bash
   # Create systemd service
   sudo nano /etc/systemd/system/blob-ai.service
   ```
   
   Service file content:
   ```ini
   [Unit]
   Description=BLOB AI Forex Engine
   After=network.target
   
   [Service]
   Type=simple
   User=your_user
   WorkingDirectory=/path/to/blob-ai
   ExecStart=/usr/bin/python3 start_blob_ai.py headless
   Restart=always
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   Enable and start:
   ```bash
   sudo systemctl enable blob-ai
   sudo systemctl start blob-ai
   ```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000 8501

CMD ["python", "start_blob_ai.py", "headless"]
```

## 🆘 Support

### Getting Help
1. **Check Documentation**: README.md and this guide
2. **Run Diagnostics**: `python test_system.py`
3. **Check Logs**: Look in the `logs/` directory
4. **Community Support**: [GitHub Issues](link-to-issues)

### Reporting Issues
When reporting issues, include:
- Operating system and Python version
- Complete error message
- Relevant log files
- Steps to reproduce
- Configuration settings (without sensitive data)

### Feature Requests
We welcome feature requests! Please include:
- Detailed description of the feature
- Use case and benefits
- Any relevant examples or mockups

## 📚 Next Steps

After successful installation:

1. **📖 Read the Documentation**: Familiarize yourself with all features
2. **🧪 Run Examples**: Execute `python example_usage.py`
3. **⚙️ Customize Configuration**: Adjust settings for your trading style
4. **📊 Explore Dashboard**: Learn all dashboard features
5. **🔌 Test API**: Try the API endpoints
6. **📈 Paper Trade**: Test with demo account first
7. **🎯 Go Live**: Deploy with real account (at your own risk)

## 🎉 Congratulations!

You now have a fully functional agentic Forex analytics engine! 

The system will:
- ✅ Collect multi-timeframe data every 5 minutes
- ✅ Analyze Smart Money Concepts in real-time
- ✅ Generate high-probability trading signals
- ✅ Provide institutional-grade market analysis
- ✅ Monitor volatility and session behaviors
- ✅ Manage risk automatically

**Happy Trading! 🚀📈**

---

*Remember: Trading involves risk. Never trade with money you can't afford to lose. This system is for educational and research purposes. Always test thoroughly before live trading.*