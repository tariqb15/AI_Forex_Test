#!/usr/bin/env python3
"""
Configuration settings for the Agentic Forex Engine

Author: BLOB AI Trading System
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum

class TradingMode(Enum):
    DEMO = "demo"
    LIVE = "live"
    BACKTEST = "backtest"

class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class MT5Config:
    """MetaTrader 5 connection configuration"""
    server: str = "FXOpen-MT5"  # MT5 server name
    login: int = 26007926    # Account number
    password: str = "(!f#Q7UV"  # Account password
    timeout: int = 60000  # Connection timeout in milliseconds
    portable: bool = False  # Use portable mode
    
    def __init__(self):
        self.server = "FXOpen-MT5"  # MT5 server name
        self.login = 26007926  # Your MT5 login
        self.password = "(!f#Q7UV"  # Your MT5 password

@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    symbol: str = "USDJPY"
    lot_size: float = 0.01  # Standard lot size
    max_spread_pips: float = 3.0  # Maximum acceptable spread in pips
    slippage: int = 3  # Maximum slippage in points
    magic_number: int = 12345  # EA magic number
    
    def get_max_spread_price(self, symbol: str) -> float:
        """Get maximum spread in price units based on currency pair"""
        if 'JPY' in symbol:
            return self.max_spread_pips * 0.01  # JPY pairs: 1 pip = 0.01
        elif any(x in symbol for x in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return self.max_spread_pips * 0.1   # Precious metals: 1 pip = 0.1
        else:
            return self.max_spread_pips * 0.0001  # Major pairs: 1 pip = 0.0001
    
    # Risk management
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_daily_loss: float = 0.06  # 6% maximum daily loss
    max_open_positions: int = 3  # Maximum concurrent positions
    
    # Signal filtering
    min_confidence: float = 0.6  # Minimum signal confidence
    min_risk_reward: float = 1.5  # Minimum risk/reward ratio
    
@dataclass
class AnalysisConfig:
    """Analysis parameters configuration"""
    # Data collection
    analysis_interval: int = 300  # Analysis interval in seconds (5 minutes)
    data_history_bars: int = 200  # Number of historical bars to analyze (reduced from 1000 for faster analysis)
    
    # Smart Money Concepts
    structure_break_lookback: int = 20  # Periods to look back for structure breaks
    order_block_strength_threshold: float = 0.4  # Minimum order block strength
    fvg_min_size_pips: float = 1.0  # Minimum FVG size in pips
    liquidity_sweep_buffer_pips: float = 2.0  # Buffer for liquidity sweeps in pips
    
    def get_fvg_min_size(self, symbol: str) -> float:
        """Get minimum FVG size in price units based on currency pair"""
        if 'JPY' in symbol:
            return self.fvg_min_size_pips * 0.01  # JPY pairs: 1 pip = 0.01
        elif any(x in symbol for x in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return self.fvg_min_size_pips * 0.1   # Precious metals: 1 pip = 0.1
        else:
            return self.fvg_min_size_pips * 0.0001  # Major pairs: 1 pip = 0.0001
    
    def get_liquidity_sweep_buffer(self, symbol: str) -> float:
        """Get liquidity sweep buffer in price units based on currency pair"""
        if 'JPY' in symbol:
            return self.liquidity_sweep_buffer_pips * 0.01  # JPY pairs: 1 pip = 0.01
        elif any(x in symbol for x in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return self.liquidity_sweep_buffer_pips * 0.1   # Precious metals: 1 pip = 0.1
        else:
            return self.liquidity_sweep_buffer_pips * 0.0001  # Major pairs: 1 pip = 0.0001
    
    # Volatility analysis
    atr_period: int = 14  # ATR calculation period
    volatility_compression_threshold: float = 0.3  # Compression detection threshold
    volatility_expansion_threshold: float = 2.0  # Expansion detection threshold
    compression_min_duration: int = 5  # Minimum compression duration
    
    # Session analysis
    session_lookback_days: int = 30  # Days to analyze for session statistics
    breakout_significance_multiplier: float = 1.5  # Range multiplier for significant moves
    
@dataclass
class NotificationConfig:
    """Notification settings"""
    enable_email: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = None
    
    enable_telegram: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    enable_discord: bool = False
    discord_webhook_url: str = ""
    
    # Notification triggers
    notify_on_signals: bool = True
    notify_on_trades: bool = True
    notify_on_errors: bool = True
    min_signal_strength_for_notification: int = 3  # SignalStrength.STRONG

@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_to_file: bool = True
    log_file_path: str = "forex_engine.log"
    log_rotation: str = "1 day"  # Log rotation interval
    log_retention: str = "30 days"  # Log retention period
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"

@dataclass
class DatabaseConfig:
    """Database configuration for storing analysis results"""
    enable_database: bool = False
    db_type: str = "sqlite"  # sqlite, postgresql, mysql
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "forex_engine"
    db_username: str = ""
    db_password: str = ""
    db_file_path: str = "forex_engine.db"  # For SQLite

@dataclass
class WebConfig:
    """Web dashboard configuration"""
    enable_dashboard: bool = True
    dashboard_host: str = "localhost"
    dashboard_port: int = 8501
    dashboard_auto_refresh: bool = True
    dashboard_refresh_interval: int = 300  # seconds
    
    enable_api: bool = False
    api_host: str = "localhost"
    api_port: int = 8000
    api_key: str = ""  # API authentication key

class ForexEngineConfig:
    """Main configuration class for the Forex Engine"""
    
    def __init__(self):
        self.trading_mode = TradingMode.LIVE
        self.risk_level = RiskLevel.MODERATE
        
        # Component configurations
        self.mt5 = MT5Config()
        self.trading = TradingConfig()
        self.analysis = AnalysisConfig()
        self.notifications = NotificationConfig()
        self.logging = LoggingConfig()
        self.database = DatabaseConfig()
        self.web = WebConfig()
        
        # Apply risk level adjustments
        self._apply_risk_level_settings()
    
    def _apply_risk_level_settings(self):
        """Apply risk level specific settings"""
        if self.risk_level == RiskLevel.CONSERVATIVE:
            self.trading.risk_per_trade = 0.01  # 1%
            self.trading.max_daily_loss = 0.03  # 3%
            self.trading.max_open_positions = 1
            self.trading.min_confidence = 0.8
            self.trading.min_risk_reward = 2.0
            
        elif self.risk_level == RiskLevel.MODERATE:
            self.trading.risk_per_trade = 0.02  # 2%
            self.trading.max_daily_loss = 0.06  # 6%
            self.trading.max_open_positions = 3
            self.trading.min_confidence = 0.4  # Lowered from 0.6 to allow more signals
            self.trading.min_risk_reward = 1.2  # Lowered from 1.5 to allow more signals
            
        elif self.risk_level == RiskLevel.AGGRESSIVE:
            self.trading.risk_per_trade = 0.03  # 3%
            self.trading.max_daily_loss = 0.10  # 10%
            self.trading.max_open_positions = 5
            self.trading.min_confidence = 0.3  # Lowered from 0.5 to allow more signals
            self.trading.min_risk_reward = 1.0  # Lowered from 1.2 to allow more signals
    
    def set_risk_level(self, risk_level: RiskLevel):
        """Change risk level and apply settings"""
        self.risk_level = risk_level
        self._apply_risk_level_settings()
    
    def set_trading_mode(self, mode: TradingMode):
        """Change trading mode"""
        self.trading_mode = mode
        
        if mode == TradingMode.DEMO:
            # Demo mode settings
            self.trading.lot_size = 0.01
            self.notifications.notify_on_trades = False
            
        elif mode == TradingMode.LIVE:
            # Live mode settings - more conservative
            self.trading.min_confidence = max(self.trading.min_confidence, 0.7)
            self.notifications.notify_on_trades = True
            
        elif mode == TradingMode.BACKTEST:
            # Backtest mode settings
            self.analysis.analysis_interval = 60  # Faster analysis for backtesting
            self.notifications.notify_on_signals = False
            self.notifications.notify_on_trades = False
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # MT5 validation
        if self.trading_mode in [TradingMode.DEMO, TradingMode.LIVE]:
            if not self.mt5.login:
                issues.append("MT5 login not configured")
            if not self.mt5.password:
                issues.append("MT5 password not configured")
        
        # Trading validation
        if self.trading.risk_per_trade <= 0 or self.trading.risk_per_trade > 0.1:
            issues.append("Risk per trade should be between 0% and 10%")
        
        if self.trading.max_daily_loss <= 0 or self.trading.max_daily_loss > 0.2:
            issues.append("Max daily loss should be between 0% and 20%")
        
        if self.trading.min_confidence < 0 or self.trading.min_confidence > 1:
            issues.append("Min confidence should be between 0 and 1")
        
        if self.trading.min_risk_reward < 1:
            issues.append("Min risk/reward should be at least 1:1")
        
        # Notification validation
        if self.notifications.enable_email:
            if not self.notifications.email_username:
                issues.append("Email username not configured")
            if not self.notifications.email_password:
                issues.append("Email password not configured")
            if not self.notifications.email_recipients:
                issues.append("Email recipients not configured")
        
        if self.notifications.enable_telegram:
            if not self.notifications.telegram_bot_token:
                issues.append("Telegram bot token not configured")
            if not self.notifications.telegram_chat_id:
                issues.append("Telegram chat ID not configured")
        
        if self.notifications.enable_discord:
            if not self.notifications.discord_webhook_url:
                issues.append("Discord webhook URL not configured")
        
        return issues
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'trading_mode': self.trading_mode.value,
            'risk_level': self.risk_level.value,
            'mt5': self.mt5.__dict__,
            'trading': self.trading.__dict__,
            'analysis': self.analysis.__dict__,
            'notifications': self.notifications.__dict__,
            'logging': self.logging.__dict__,
            'database': self.database.__dict__,
            'web': self.web.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ForexEngineConfig':
        """Create configuration from dictionary"""
        config = cls()
        
        # Set basic properties
        if 'trading_mode' in config_dict:
            config.trading_mode = TradingMode(config_dict['trading_mode'])
        if 'risk_level' in config_dict:
            config.risk_level = RiskLevel(config_dict['risk_level'])
        
        # Update component configurations
        for component_name, component_config in config_dict.items():
            if hasattr(config, component_name) and isinstance(component_config, dict):
                component = getattr(config, component_name)
                for key, value in component_config.items():
                    if hasattr(component, key):
                        setattr(component, key, value)
        
        return config
    
    def save_to_file(self, file_path: str = "config.json"):
        """Save configuration to JSON file"""
        import json
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str = "config.json") -> 'ForexEngineConfig':
        """Load configuration from JSON file"""
        import json
        import os
        
        if not os.path.exists(file_path):
            # Create default config file
            config = cls()
            config.save_to_file(file_path)
            return config
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)

# Predefined configurations for different use cases
class PresetConfigs:
    """Predefined configuration presets"""
    
    @staticmethod
    def demo_conservative() -> ForexEngineConfig:
        """Conservative demo trading configuration"""
        config = ForexEngineConfig()
        config.set_trading_mode(TradingMode.DEMO)
        config.set_risk_level(RiskLevel.CONSERVATIVE)
        return config
    
    @staticmethod
    def demo_aggressive() -> ForexEngineConfig:
        """Aggressive demo trading configuration"""
        config = ForexEngineConfig()
        config.set_trading_mode(TradingMode.DEMO)
        config.set_risk_level(RiskLevel.AGGRESSIVE)
        return config
    
    @staticmethod
    def live_conservative() -> ForexEngineConfig:
        """Conservative live trading configuration"""
        config = ForexEngineConfig()
        config.set_trading_mode(TradingMode.LIVE)
        config.set_risk_level(RiskLevel.CONSERVATIVE)
        
        # Enable all notifications for live trading
        config.notifications.notify_on_signals = True
        config.notifications.notify_on_trades = True
        config.notifications.notify_on_errors = True
        
        return config
    
    @staticmethod
    def backtest_fast() -> ForexEngineConfig:
        """Fast backtesting configuration"""
        config = ForexEngineConfig()
        config.set_trading_mode(TradingMode.BACKTEST)
        
        # Optimize for speed
        config.analysis.analysis_interval = 60  # 1 minute
        config.analysis.data_history_bars = 500  # Less historical data
        
        # Disable unnecessary features
        config.notifications.notify_on_signals = False
        config.notifications.notify_on_trades = False
        config.web.enable_dashboard = False
        config.database.enable_database = False
        
        return config
    
    @staticmethod
    def research_mode() -> ForexEngineConfig:
        """Configuration for research and analysis"""
        config = ForexEngineConfig()
        config.set_trading_mode(TradingMode.DEMO)
        
        # Extended analysis
        config.analysis.data_history_bars = 2000  # More historical data
        config.analysis.session_lookback_days = 90  # Longer session analysis
        
        # Enable database for research
        config.database.enable_database = True
        
        # Detailed logging
        config.logging.log_level = "DEBUG"
        
        return config

# Global configuration instance
CONFIG = ForexEngineConfig()

# Configuration validation and setup functions
def setup_config(config_file: str = "config.json") -> ForexEngineConfig:
    """Setup and validate configuration"""
    global CONFIG
    
    try:
        CONFIG = ForexEngineConfig.load_from_file(config_file)
        
        # Validate configuration
        issues = CONFIG.validate_config()
        if issues:
            print("⚠️ Configuration Issues:")
            for issue in issues:
                print(f"  - {issue}")
            print("\nPlease update your configuration file.")
        else:
            print("✅ Configuration validated successfully")
        
        return CONFIG
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        print("Creating default configuration...")
        
        CONFIG = ForexEngineConfig()
        CONFIG.save_to_file(config_file)
        
        return CONFIG

def get_config() -> ForexEngineConfig:
    """Get the global configuration instance"""
    return CONFIG

if __name__ == "__main__":
    # Example usage and testing
    print("🔧 Forex Engine Configuration")
    print("=" * 40)
    
    # Create and test different configurations
    configs = {
        "Demo Conservative": PresetConfigs.demo_conservative(),
        "Demo Aggressive": PresetConfigs.demo_aggressive(),
        "Live Conservative": PresetConfigs.live_conservative(),
        "Backtest Fast": PresetConfigs.backtest_fast(),
        "Research Mode": PresetConfigs.research_mode()
    }
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  Trading Mode: {config.trading_mode.value}")
        print(f"  Risk Level: {config.risk_level.value}")
        print(f"  Risk per Trade: {config.trading.risk_per_trade:.1%}")
        print(f"  Min Confidence: {config.trading.min_confidence:.1%}")
        print(f"  Max Positions: {config.trading.max_open_positions}")
        
        # Validate
        issues = config.validate_config()
        if issues:
            print(f"  Issues: {len(issues)}")
        else:
            print("  ✅ Valid")
    
    # Save example configuration
    example_config = PresetConfigs.demo_conservative()
    example_config.save_to_file("example_config.json")
    print("\n💾 Example configuration saved to 'example_config.json'")