#!/usr/bin/env python3
"""
BLOB AI Forex Engine - Master Startup Script

This script provides a unified interface to start all components of the
BLOB AI Forex Analytics Engine system.

Author: BLOB AI Trading System
Version: 1.0.0
Date: January 2024
"""

import asyncio
import logging
import os
import sys
import subprocess
import time
import signal
from datetime import datetime
from typing import Dict, List, Optional
import threading
import webbrowser
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BLOBAILauncher:
    """
    Master launcher for BLOB AI Forex Engine
    """
    
    def __init__(self):
        self.processes = {}
        self.running = False
        self.start_time = datetime.now()
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutdown signal received. Stopping all services...")
        self.stop_all_services()
        sys.exit(0)
    
    def print_banner(self):
        """Print startup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ██████╗ ██╗      ██████╗ ██████╗      █████╗ ██╗                          ║
║  ██╔══██╗██║     ██╔═══██╗██╔══██╗    ██╔══██╗██║                          ║
║  ██████╔╝██║     ██║   ██║██████╔╝    ███████║██║                          ║
║  ██╔══██╗██║     ██║   ██║██╔══██╗    ██╔══██║██║                          ║
║  ██████╔╝███████╗╚██████╔╝██████╔╝    ██║  ██║██║                          ║
║  ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝     ╚═╝  ╚═╝╚═╝                          ║
║                                                                              ║
║                    🚀 AGENTIC FOREX ANALYTICS ENGINE 🚀                     ║
║                                                                              ║
║  💡 Think like a top-performing prop trader                                  ║
║  📊 Smart Money Concepts + Institutional Logic                               ║
║  ⚡ Real-time Multi-timeframe Analysis                                       ║
║  🎯 High-probability Signal Generation                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"🕐 Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Working directory: {os.getcwd()}")
        print("="*80)
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        print("\n🔍 Checking dependencies...")
        
        required_packages = [
            ('MetaTrader5', 'MetaTrader5'),
            ('pandas', 'pandas'),
            ('numpy', 'numpy'),
            ('plotly', 'plotly'),
            ('streamlit', 'streamlit'),
            ('fastapi', 'fastapi'),
            ('uvicorn', 'uvicorn'),
            ('scikit-learn', 'sklearn')
        ]
        
        missing_packages = []
        
        for display_name, import_name in required_packages:
            try:
                __import__(import_name)
                print(f"  ✅ {display_name}")
            except ImportError:
                print(f"  ❌ {display_name} - MISSING")
                missing_packages.append(display_name)
        
        if missing_packages:
            print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
            print("💡 Run 'python setup.py' to install missing dependencies")
            return False
        
        print("\n✅ All dependencies satisfied!")
        return True
    
    def check_files(self) -> bool:
        """Check if all required files exist"""
        print("\n🔍 Checking required files...")
        
        required_files = [
            'forex_engine.py',
            'config.py',
            'dashboard.py',
            'api.py',
            'requirements.txt'
        ]
        
        missing_files = []
        
        for file_name in required_files:
            if os.path.exists(file_name):
                print(f"  ✅ {file_name}")
            else:
                print(f"  ❌ {file_name} - MISSING")
                missing_files.append(file_name)
        
        if missing_files:
            print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
            return False
        
        print("\n✅ All required files present!")
        return True
    
    def start_service(self, name: str, command: List[str], cwd: str = None) -> bool:
        """Start a service process"""
        try:
            print(f"🚀 Starting {name}...")
            
            # Create logs directory for this service
            service_log_dir = f"logs/{name.lower().replace(' ', '_')}"
            os.makedirs(service_log_dir, exist_ok=True)
            
            # Open log files
            stdout_log = open(f"{service_log_dir}/stdout.log", "w")
            stderr_log = open(f"{service_log_dir}/stderr.log", "w")
            
            # Start the process
            process = subprocess.Popen(
                command,
                stdout=stdout_log,
                stderr=stderr_log,
                cwd=cwd or os.getcwd(),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            self.processes[name] = {
                'process': process,
                'command': command,
                'stdout_log': stdout_log,
                'stderr_log': stderr_log,
                'start_time': datetime.now()
            }
            
            # Give the process a moment to start
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                print(f"  ✅ {name} started successfully (PID: {process.pid})")
                return True
            else:
                print(f"  ❌ {name} failed to start")
                return False
                
        except Exception as e:
            print(f"  ❌ Error starting {name}: {e}")
            return False
    
    def stop_service(self, name: str):
        """Stop a service process"""
        if name in self.processes:
            process_info = self.processes[name]
            process = process_info['process']
            
            try:
                print(f"🛑 Stopping {name}...")
                
                # Terminate the process
                if os.name == 'nt':
                    # Windows
                    process.terminate()
                else:
                    # Unix-like
                    process.terminate()
                
                # Wait for process to terminate
                try:
                    process.wait(timeout=10)
                    print(f"  ✅ {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f"  ⚠️ {name} didn't stop gracefully, forcing...")
                    process.kill()
                    process.wait()
                    print(f"  ✅ {name} force stopped")
                
                # Close log files
                process_info['stdout_log'].close()
                process_info['stderr_log'].close()
                
                del self.processes[name]
                
            except Exception as e:
                print(f"  ❌ Error stopping {name}: {e}")
    
    def start_forex_engine(self) -> bool:
        """Start the main forex engine"""
        return self.start_service(
            "Forex Engine",
            [sys.executable, "forex_engine.py"]
        )
    
    def start_dashboard(self) -> bool:
        """Start the Streamlit dashboard"""
        return self.start_service(
            "Dashboard",
            [sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.port=8502", "--server.headless=true"]
        )
    
    def start_api_server(self) -> bool:
        """Start the FastAPI server"""
        return self.start_service(
            "API Server",
            [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
        )
    
    def monitor_services(self):
        """Monitor running services"""
        while self.running:
            try:
                # Check each service
                for name, info in list(self.processes.items()):
                    process = info['process']
                    
                    if process.poll() is not None:
                        # Process has stopped
                        print(f"⚠️ {name} has stopped unexpectedly (exit code: {process.returncode})")
                        
                        # Attempt to restart
                        print(f"🔄 Attempting to restart {name}...")
                        self.stop_service(name)  # Clean up
                        
                        # Restart based on service type
                        if name == "Forex Engine":
                            self.start_forex_engine()
                        elif name == "Dashboard":
                            self.start_dashboard()
                        elif name == "API Server":
                            self.start_api_server()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in service monitoring: {e}")
                time.sleep(60)  # Wait longer on error
    
    def show_status(self):
        """Show status of all services"""
        print("\n📊 SERVICE STATUS:")
        print("="*50)
        
        if not self.processes:
            print("  No services running")
            return
        
        for name, info in self.processes.items():
            process = info['process']
            start_time = info['start_time']
            uptime = datetime.now() - start_time
            
            if process.poll() is None:
                status = "🟢 RUNNING"
            else:
                status = f"🔴 STOPPED (exit code: {process.returncode})"
            
            print(f"  {name:15} | {status:20} | PID: {process.pid:8} | Uptime: {uptime}")
        
        print("\n🌐 ACCESS POINTS:")
        print("  Dashboard:  http://localhost:8502")
        print("  API:        http://localhost:8001")
        print("  API Docs:   http://localhost:8001/docs")
    
    def open_browser_tabs(self):
        """Open browser tabs for dashboard and API docs"""
        try:
            print("\n🌐 Opening browser tabs...")
            
            # Wait a moment for services to fully start
            time.sleep(5)
            
            # Open dashboard
            webbrowser.open('http://localhost:8502')
            time.sleep(1)
            
            # Open API documentation
            webbrowser.open('http://localhost:8001/docs')
            
            print("  ✅ Browser tabs opened")
            
        except Exception as e:
            print(f"  ⚠️ Could not open browser: {e}")
    
    def stop_all_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping all services...")
        
        self.running = False
        
        # Stop all services
        for name in list(self.processes.keys()):
            self.stop_service(name)
        
        print("\n✅ All services stopped")
    
    def run_interactive_mode(self):
        """Run in interactive mode with menu"""
        while True:
            print("\n" + "="*50)
            print("BLOB AI FOREX ENGINE - CONTROL PANEL")
            print("="*50)
            print("1. Start All Services")
            print("2. Stop All Services")
            print("3. Show Service Status")
            print("4. Start Individual Service")
            print("5. Stop Individual Service")
            print("6. View Logs")
            print("7. Open Browser Tabs")
            print("8. Run System Test")
            print("9. Run Examples")
            print("0. Exit")
            
            try:
                choice = input("\nEnter your choice (0-9): ").strip()
                
                if choice == "1":
                    self.start_all_services()
                elif choice == "2":
                    self.stop_all_services()
                elif choice == "3":
                    self.show_status()
                elif choice == "4":
                    self.start_individual_service()
                elif choice == "5":
                    self.stop_individual_service()
                elif choice == "6":
                    self.view_logs()
                elif choice == "7":
                    self.open_browser_tabs()
                elif choice == "8":
                    self.run_system_test()
                elif choice == "9":
                    self.run_examples()
                elif choice == "0":
                    self.stop_all_services()
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted by user")
                self.stop_all_services()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def start_individual_service(self):
        """Start an individual service"""
        print("\nAvailable services:")
        print("1. Forex Engine")
        print("2. Dashboard")
        print("3. API Server")
        
        choice = input("Enter service number: ").strip()
        
        if choice == "1":
            self.start_forex_engine()
        elif choice == "2":
            self.start_dashboard()
        elif choice == "3":
            self.start_api_server()
        else:
            print("❌ Invalid choice")
    
    def stop_individual_service(self):
        """Stop an individual service"""
        if not self.processes:
            print("No services running")
            return
        
        print("\nRunning services:")
        services = list(self.processes.keys())
        for i, name in enumerate(services, 1):
            print(f"{i}. {name}")
        
        try:
            choice = int(input("Enter service number: ").strip()) - 1
            if 0 <= choice < len(services):
                self.stop_service(services[choice])
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Invalid input")
    
    def view_logs(self):
        """View service logs"""
        print("\nAvailable log files:")
        
        log_files = []
        for root, dirs, files in os.walk('logs'):
            for file in files:
                if file.endswith('.log'):
                    log_files.append(os.path.join(root, file))
        
        if not log_files:
            print("No log files found")
            return
        
        for i, log_file in enumerate(log_files, 1):
            print(f"{i}. {log_file}")
        
        try:
            choice = int(input("Enter log file number (0 to cancel): ").strip())
            if choice == 0:
                return
            elif 1 <= choice <= len(log_files):
                log_file = log_files[choice - 1]
                print(f"\n📄 Last 50 lines of {log_file}:")
                print("-" * 60)
                
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-50:]:
                            print(line.rstrip())
                except Exception as e:
                    print(f"Error reading log file: {e}")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Invalid input")
    
    def run_system_test(self):
        """Run system tests"""
        print("\n🧪 Running system tests...")
        try:
            result = subprocess.run([sys.executable, "test_system.py"], 
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
        except Exception as e:
            print(f"Error running tests: {e}")
    
    def run_examples(self):
        """Run example usage script"""
        print("\n📚 Running examples...")
        try:
            subprocess.run([sys.executable, "example_usage.py"])
        except Exception as e:
            print(f"Error running examples: {e}")
    
    def start_all_services(self, open_browser: bool = True):
        """Start all services"""
        print("\n🚀 Starting all BLOB AI services...")
        
        # Check prerequisites
        if not self.check_dependencies() or not self.check_files():
            print("\n❌ Prerequisites not met. Please fix issues and try again.")
            return False
        
        success_count = 0
        total_services = 3
        
        # Start services in order
        if self.start_forex_engine():
            success_count += 1
        
        if self.start_api_server():
            success_count += 1
        
        if self.start_dashboard():
            success_count += 1
        
        if success_count == total_services:
            print(f"\n🎉 All {total_services} services started successfully!")
            
            # Start monitoring thread
            self.running = True
            monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
            monitor_thread.start()
            
            # Show status
            time.sleep(3)
            self.show_status()
            
            # Open browser tabs
            if open_browser:
                self.open_browser_tabs()
            
            return True
        else:
            print(f"\n⚠️ Only {success_count}/{total_services} services started successfully")
            return False
    
    def run_headless(self):
        """Run in headless mode (no interaction)"""
        print("\n🤖 Running in headless mode...")
        
        if self.start_all_services(open_browser=False):
            print("\n✅ All services running. Press Ctrl+C to stop.")
            
            try:
                # Keep running until interrupted
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n🛑 Shutdown requested")
            finally:
                self.stop_all_services()
        else:
            print("\n❌ Failed to start services")
            sys.exit(1)

def main():
    """Main function"""
    launcher = BLOBAILauncher()
    launcher.print_banner()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "start":
            launcher.start_all_services()
            launcher.run_interactive_mode()
        elif command == "headless":
            launcher.run_headless()
        elif command == "test":
            launcher.run_system_test()
        elif command == "examples":
            launcher.run_examples()
        elif command == "status":
            launcher.show_status()
        else:
            print(f"\n❌ Unknown command: {command}")
            print("\nAvailable commands:")
            print("  start     - Start all services with interactive mode")
            print("  headless  - Start all services in background")
            print("  test      - Run system tests")
            print("  examples  - Run example usage")
            print("  status    - Show service status")
    else:
        # Interactive mode
        launcher.run_interactive_mode()

if __name__ == "__main__":
    main()