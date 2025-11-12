#!/usr/bin/env python3
"""
CLOUD DEPLOYMENT PACKAGE - Trading Bot
Optimized for Railway, Render, Heroku, and other free cloud platforms
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import and run the main trading bot
try:
    from quality_live_trading_bot import QualityTradingBot
    
    def main():
        """Main function for cloud deployment"""
        print("🚀 Starting Quality Trading Bot...")
        print(f"📍 Running in: {os.getcwd()}")
        print(f"🐍 Python version: {sys.version}")
        
        # Initialize and run the bot
        bot = QualityTradingBot()
        bot.run()
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📁 Current directory contents:")
    for item in os.listdir('.'):
        print(f"  - {item}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Fatal error: {e}")
    sys.exit(1)