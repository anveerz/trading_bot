# 🤖 Trading Bot - Cloud Deployment Package

This package contains everything you need to deploy your trading bot to run 24/7 on free cloud platforms.

## 📁 Package Contents

- `main.py` - Main entry point for cloud deployment
- `quality_live_trading_bot.py` - Complete trading bot with optimized strategies
- `historical_data_manager.py` - Historical data fetching and management
- `optimized_signal_generator.py` - High-win-rate signal generation
- `requirements.txt` - Python dependencies
- `Procfile` - For Heroku/Railway deployment
- `runtime.txt` - Python version specification
- `Dockerfile` - For containerized deployment
- `.env.example` - Environment variables template
- `DEPLOYMENT_GUIDE.md` - Comprehensive deployment instructions

## 🚀 Quick Start

### Option 1: Railway (Recommended)
1. Create GitHub repository with these files
2. Deploy at [railway.app](https://railway.app)
3. Set environment variables
4. Done! Your bot runs 24/7

### Option 2: Oracle Cloud
1. Create always-free Oracle Cloud account
2. Launch VM instance
3. Upload files and run with screen
4. Bot runs 24/7 on free tier

See `DEPLOYMENT_GUIDE.md` for detailed instructions.

## ⚡ Features

- **24/7 Operation**: Designed for continuous cloud deployment
- **Auto-Restart**: Crashes automatically restart
- **Multiple Currency Pairs**: 7 major forex pairs
- **Optimized Strategies**: 65%+ win rate strategies
- **Risk Management**: Built-in position sizing and stop losses
- **Telegram Notifications**: Real-time signal alerts
- **SQLite Database**: No external database required
- **Low Resource Usage**: Perfect for free tiers

## 📊 Trading Strategy

- **EUR/USD**: Range trading with trend reversal
- **GBP/USD**: Momentum continuation with breakout
- **EUR/GBP**: Mean reversion with extreme conditions
- **USD/CAD**: Regime-aware range trading
- **GBP/JPY**: High volatility breakout
- **AUD/USD**: Trend-following momentum
- **USD/CHF**: Safe haven stability trading

## 🔧 Technical Requirements

- **Memory**: ~150MB
- **CPU**: <10% usage
- **Storage**: ~50MB
- **Network**: Minimal bandwidth
- **Runtime**: Python 3.11+

## 📱 Monitoring

- Real-time logs on cloud platform
- Telegram notifications for all signals
- SQLite database for performance tracking
- Risk management alerts

## ⚠️ Important Notes

1. **API Keys**: Already configured for TwelveData and Finnhub
2. **Telegram**: Bot configured for instant notifications
3. **Database**: SQLite for simplicity and portability
4. **Free Tier**: Optimized for free cloud tiers
5. **Reliability**: Auto-restart on crashes

## 🎯 Deployment Time

- **Railway**: 5 minutes from start to running
- **Oracle Cloud**: 10 minutes setup
- **Render**: 5 minutes deployment
- **Google Cloud Run**: 5 minutes setup

## 📞 Support

For deployment issues:
1. Check `DEPLOYMENT_GUIDE.md`
2. Review application logs
3. Verify environment variables
4. Ensure API keys are valid

**Ready to deploy? Start with Railway - it's the easiest and most reliable option!**