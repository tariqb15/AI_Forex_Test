#!/usr/bin/env python3
"""
Setup script for the Agentic Forex Engine
Handles installation, configuration, and initial setup

Author: BLOB AI Trading System
Version: 1.0.0
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
import platform

class ForexEngineSetup:
    """Setup and installation manager for the Forex Engine"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_executable = sys.executable
        self.os_type = platform.system().lower()
        
        print("🚀 BLOB AI - Forex Engine Setup")
        print("=" * 40)
        print(f"Project Root: {self.project_root}")
        print(f"Python: {self.python_executable}")
        print(f"OS: {platform.system()} {platform.release()}")
        print()
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        print("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python {version.major}.{version.minor} is not supported")
            print("   Minimum required: Python 3.8")
            return False
        
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    
    def install_dependencies(self) -> bool:
        """Install required Python packages"""
        print("📦 Installing dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            # Upgrade pip first
            subprocess.run([
                self.python_executable, "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, capture_output=True)
            
            # Install requirements
            result = subprocess.run([
                self.python_executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True, capture_output=True, text=True)
            
            print("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            return False
    
    def check_mt5_installation(self) -> bool:
        """Check if MetaTrader 5 is installed and accessible"""
        print("🔍 Checking MetaTrader 5 installation...")
        
        try:
            import MetaTrader5 as mt5
            
            # Try to initialize MT5
            if mt5.initialize():
                version = mt5.version()
                print(f"✅ MetaTrader 5 found - Version: {version}")
                mt5.shutdown()
                return True
            else:
                print("⚠️ MetaTrader 5 found but failed to initialize")
                print("   Make sure MT5 is installed and not running")
                return False
                
        except ImportError:
            print("❌ MetaTrader5 package not available")
            print("   Install with: pip install MetaTrader5")
            return False
        except Exception as e:
            print(f"❌ Error checking MT5: {e}")
            return False
    
    def create_config_file(self) -> bool:
        """Create initial configuration file"""
        print("⚙️ Creating configuration file...")
        
        config_file = self.project_root / "config.json"
        
        try:
            from config import ForexEngineConfig, PresetConfigs
            
            # Ask user for configuration type
            print("\nSelect configuration type:")
            print("1. Demo Conservative (Recommended for beginners)")
            print("2. Demo Aggressive")
            print("3. Research Mode")
            print("4. Custom")
            
            while True:
                choice = input("\nEnter choice (1-4): ").strip()
                
                if choice == "1":
                    config = PresetConfigs.demo_conservative()
                    break
                elif choice == "2":
                    config = PresetConfigs.demo_aggressive()
                    break
                elif choice == "3":
                    config = PresetConfigs.research_mode()
                    break
                elif choice == "4":
                    config = self._create_custom_config()
                    break
                else:
                    print("Invalid choice. Please enter 1-4.")
            
            # Save configuration
            config.save_to_file(str(config_file))
            print(f"✅ Configuration saved to {config_file}")
            
            # Validate configuration
            issues = config.validate_config()
            if issues:
                print("\n⚠️ Configuration issues (can be fixed later):")
                for issue in issues:
                    print(f"  - {issue}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create configuration: {e}")
            return False
    
    def _create_custom_config(self):
        """Create custom configuration through user input"""
        from config import ForexEngineConfig, TradingMode, RiskLevel
        
        config = ForexEngineConfig()
        
        print("\n🔧 Custom Configuration Setup")
        print("-" * 30)
        
        # Trading mode
        print("\nTrading Mode:")
        print("1. Demo")
        print("2. Live")
        print("3. Backtest")
        
        mode_choice = input("Select mode (1-3): ").strip()
        mode_map = {"1": TradingMode.DEMO, "2": TradingMode.LIVE, "3": TradingMode.BACKTEST}
        config.set_trading_mode(mode_map.get(mode_choice, TradingMode.DEMO))
        
        # Risk level
        print("\nRisk Level:")
        print("1. Conservative")
        print("2. Moderate")
        print("3. Aggressive")
        
        risk_choice = input("Select risk level (1-3): ").strip()
        risk_map = {"1": RiskLevel.CONSERVATIVE, "2": RiskLevel.MODERATE, "3": RiskLevel.AGGRESSIVE}
        config.set_risk_level(risk_map.get(risk_choice, RiskLevel.MODERATE))
        
        # Analysis interval
        try:
            interval = int(input("\nAnalysis interval in minutes (default 5): ") or "5")
            config.analysis.analysis_interval = interval * 60
        except ValueError:
            pass
        
        # Dashboard settings
        enable_dashboard = input("\nEnable web dashboard? (y/n, default y): ").strip().lower()
        config.web.enable_dashboard = enable_dashboard != 'n'
        
        if config.web.enable_dashboard:
            try:
                port = int(input("Dashboard port (default 8501): ") or "8501")
                config.web.dashboard_port = port
            except ValueError:
                pass
        
        return config
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        print("📁 Creating directories...")
        
        directories = [
            "logs",
            "data",
            "exports",
            "backups"
        ]
        
        try:
            for dir_name in directories:
                dir_path = self.project_root / dir_name
                dir_path.mkdir(exist_ok=True)
                print(f"  ✅ {dir_name}/")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create directories: {e}")
            return False
    
    def create_startup_scripts(self) -> bool:
        """Create startup scripts for different platforms"""
        print("📜 Creating startup scripts...")
        
        try:
            # Windows batch script
            if self.os_type == "windows":
                self._create_windows_scripts()
            
            # Unix shell script
            else:
                self._create_unix_scripts()
            
            print("✅ Startup scripts created")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create startup scripts: {e}")
            return False
    
    def _create_windows_scripts(self):
        """Create Windows batch scripts"""
        # Main engine script
        engine_script = self.project_root / "start_engine.bat"
        with open(engine_script, 'w') as f:
            f.write(f"@echo off\n")
            f.write(f"echo Starting BLOB AI Forex Engine...\n")
            f.write(f"cd /d \"{self.project_root}\"\n")
            f.write(f"\"{self.python_executable}\" forex_engine.py\n")
            f.write(f"pause\n")
        
        # Dashboard script
        dashboard_script = self.project_root / "start_dashboard.bat"
        with open(dashboard_script, 'w') as f:
            f.write(f"@echo off\n")
            f.write(f"echo Starting BLOB AI Dashboard...\n")
            f.write(f"cd /d \"{self.project_root}\"\n")
            f.write(f"streamlit run dashboard.py\n")
            f.write(f"pause\n")
        
        print("  ✅ start_engine.bat")
        print("  ✅ start_dashboard.bat")
    
    def _create_unix_scripts(self):
        """Create Unix shell scripts"""
        # Main engine script
        engine_script = self.project_root / "start_engine.sh"
        with open(engine_script, 'w') as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"echo \"Starting BLOB AI Forex Engine...\"\n")
            f.write(f"cd \"{self.project_root}\"\n")
            f.write(f"\"{self.python_executable}\" forex_engine.py\n")
        
        # Dashboard script
        dashboard_script = self.project_root / "start_dashboard.sh"
        with open(dashboard_script, 'w') as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"echo \"Starting BLOB AI Dashboard...\"\n")
            f.write(f"cd \"{self.project_root}\"\n")
            f.write(f"streamlit run dashboard.py\n")
        
        # Make scripts executable
        os.chmod(engine_script, 0o755)
        os.chmod(dashboard_script, 0o755)
        
        print("  ✅ start_engine.sh")
        print("  ✅ start_dashboard.sh")
    
    def run_initial_test(self) -> bool:
        """Run initial system test"""
        print("🧪 Running initial system test...")
        
        try:
            # Test imports
            print("  Testing imports...")
            import pandas as pd
            import numpy as np
            import plotly.graph_objects as go
            import streamlit as st
            from forex_engine import AgenticForexEngine
            from config import ForexEngineConfig
            print("    ✅ All imports successful")
            
            # Test configuration
            print("  Testing configuration...")
            config = ForexEngineConfig.load_from_file("config.json")
            issues = config.validate_config()
            if not issues:
                print("    ✅ Configuration valid")
            else:
                print(f"    ⚠️ Configuration has {len(issues)} issues")
            
            # Test engine initialization (without MT5 connection)
            print("  Testing engine initialization...")
            engine = AgenticForexEngine("USDJPY")
            print("    ✅ Engine initialized successfully")
            
            print("✅ Initial system test passed")
            return True
            
        except Exception as e:
            print(f"❌ System test failed: {e}")
            return False
    
    def display_next_steps(self):
        """Display next steps for the user"""
        print("\n🎉 Setup Complete!")
        print("=" * 40)
        print("\nNext Steps:")
        print("\n1. 📊 Configure MetaTrader 5:")
        print("   - Install MetaTrader 5 if not already installed")
        print("   - Open a demo account with a broker")
        print("   - Update config.json with your MT5 credentials")
        
        print("\n2. 🚀 Start the Engine:")
        if self.os_type == "windows":
            print("   - Double-click 'start_engine.bat' to run the main engine")
            print("   - Double-click 'start_dashboard.bat' to open the web dashboard")
        else:
            print("   - Run './start_engine.sh' to start the main engine")
            print("   - Run './start_dashboard.sh' to open the web dashboard")
        
        print("\n3. 📈 Access Dashboard:")
        print("   - Open http://localhost:8501 in your browser")
        print("   - Monitor real-time analysis and trading signals")
        
        print("\n4. ⚙️ Customize Settings:")
        print("   - Edit config.json to adjust trading parameters")
        print("   - Modify risk levels and analysis intervals")
        print("   - Configure notifications (email, Telegram, Discord)")
        
        print("\n5. 📚 Documentation:")
        print("   - Read README.md for detailed usage instructions")
        print("   - Check the logs/ directory for system logs")
        print("   - Use the exports/ directory for analysis results")
        
        print("\n⚠️ Important Notes:")
        print("   - Start with demo trading to test the system")
        print("   - Never risk more than you can afford to lose")
        print("   - Monitor the system regularly and adjust settings as needed")
        
        print("\n🆘 Support:")
        print("   - Check logs for error messages")
        print("   - Ensure MT5 is properly configured")
        print("   - Verify internet connection for data feeds")
        
        print("\n" + "=" * 40)
        print("Happy Trading! 🚀📈")
    
    def run_setup(self) -> bool:
        """Run the complete setup process"""
        steps = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.install_dependencies),
            ("MetaTrader 5", self.check_mt5_installation),
            ("Directories", self.create_directories),
            ("Configuration", self.create_config_file),
            ("Startup Scripts", self.create_startup_scripts),
            ("System Test", self.run_initial_test)
        ]
        
        print("Starting setup process...\n")
        
        for step_name, step_function in steps:
            print(f"\n{'='*50}")
            print(f"Step: {step_name}")
            print(f"{'='*50}")
            
            if not step_function():
                print(f"\n❌ Setup failed at step: {step_name}")
                print("Please fix the issues and run setup again.")
                return False
        
        print(f"\n{'='*50}")
        self.display_next_steps()
        return True

def main():
    """Main setup execution"""
    setup = ForexEngineSetup()
    
    try:
        success = setup.run_setup()
        if success:
            print("\n✅ Setup completed successfully!")
            return 0
        else:
            print("\n❌ Setup failed!")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n🛑 Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())