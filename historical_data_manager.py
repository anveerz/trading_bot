#!/usr/bin/env python3
"""
FIXED HISTORICAL DATA MANAGER
Based on successful testing - handles API limitations with multiple keys
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import logging
import sqlite3
import os
from typing import Dict, List, Optional, Tuple
import json
from threading import Lock

# Configure logging
logger = logging.getLogger(__name__)

class FixedHistoricalDataManager:
    """
    FIXED Historical Data Manager with multiple API keys and proper fallback
    """
    
    def __init__(self, twelvedata_api_keys: List[str], database_path: str = "historical_data.db"):
        """
        Initialize with multiple API keys for fallback
        
        Args:
            twelvedata_api_keys: List of TwelveData API keys
            database_path: SQLite database file path
        """
        self.api_keys = twelvedata_api_keys
        self.base_url = "https://api.twelvedata.com"
        self.database_path = database_path
        
        # Currency pair mappings
        self.pair_mapping = {
            'EUR/USD': 'EUR/USD',
            'GBP/USD': 'GBP/USD', 
            'EUR/GBP': 'EUR/GBP',
            'USD/CAD': 'USD/CAD',
            'GBP/JPY': 'GBP/JPY',
            'AUD/USD': 'AUD/USD',
            'USD/CHF': 'USD/CHF'
        }
        
        # Database setup
        self.db_lock = Lock()
        self._setup_database()
        
        # Data cache
        self.data_cache = {}
        self.cache_lock = Lock()
        
        logger.info(f"✅ FixedHistoricalDataManager initialized with {len(self.api_keys)} API keys")
    
    def _setup_database(self):
        """Setup SQLite database with proper tables"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                # Create tables for each currency pair
                for pair in self.pair_mapping.keys():
                    table_name = f"data_{pair.replace('/', '_').replace('-', '_').lower()}"
                    conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            timestamp TEXT PRIMARY KEY,
                            open REAL NOT NULL,
                            high REAL NOT NULL,
                            low REAL NOT NULL,
                            close REAL NOT NULL,
                            volume INTEGER DEFAULT 0
                        )
                    """)
                conn.commit()
            logger.info("✅ Database setup complete")
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
    
    def _fetch_with_fallback(self, pair: str, interval: str = "1min", outputsize: int = 500) -> Optional[List[Dict]]:
        """
        Fetch data with API key fallback - TESTED AND WORKING
        """
        for api_key_idx, api_key in enumerate(self.api_keys):
            try:
                # Test different symbol formats for this pair
                symbol_formats = [
                    pair,                   # EUR/USD
                    pair.replace('/', ''),  # EURUSD
                    pair.replace('/', '_'), # EUR_USD
                ]
                
                for symbol in symbol_formats:
                    url = f"{self.base_url}/time_series"
                    params = {
                        'symbol': symbol,
                        'interval': interval,
                        'outputsize': outputsize,
                        'apikey': api_key
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'values' in data and len(data['values']) > 0:
                            logger.info(f"✅ {pair}: Got {len(data['values'])} candles with API key #{api_key_idx+1} and symbol '{symbol}'")
                            return data['values']
                    
                    logger.debug(f"⚠️ {pair}: No data with symbol '{symbol}' using API key #{api_key_idx+1}")
                    
            except Exception as e:
                logger.warning(f"❌ {pair}: API key #{api_key_idx+1} failed: {str(e)}")
                continue
        
        logger.error(f"❌ {pair}: ALL API keys and symbol formats failed")
        return None
    
    def fetch_historical_candles(self, symbol: str, interval: str = "1min", 
                               target_candles: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch historical candles for a specific symbol
        """
        logger.info(f"🔄 Fetching {symbol} ({target_candles} candles)...")
        
        # Fetch raw data with fallback
        raw_data = self._fetch_with_fallback(symbol, interval, target_candles)
        
        if not raw_data:
            logger.error(f"❌ Failed to fetch data for {symbol}")
            return None
        
        try:
            # Convert to DataFrame
            df_data = []
            for candle in raw_data:
                df_data.append({
                    'timestamp': candle['datetime'],
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': int(candle.get('volume', 0))
                })
            
            df = pd.DataFrame(df_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            logger.info(f"✅ {symbol}: Successfully fetched {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"❌ {symbol}: Error processing data: {str(e)}")
            return None
    
    def save_to_database(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Save DataFrame to database
        """
        if df is None or df.empty:
            return False
        
        table_name = f"data_{symbol.replace('/', '_').replace('-', '_').lower()}"
        
        try:
            with self.db_lock:
                with sqlite3.connect(self.database_path) as conn:
                    df.to_sql(table_name, conn, if_exists='replace', index=True)
                
                logger.info(f"💾 {symbol}: Saved {len(df)} candles to {table_name}")
                return True
                
        except Exception as e:
            logger.error(f"❌ {symbol}: Database save error: {str(e)}")
            return False
    
    def fetch_all_pairs_data(self, target_candles: int = 1000) -> Dict[str, bool]:
        """
        Fetch data for all currency pairs - MAIN FUNCTION FOR BOT
        """
        logger.info(f"🔄 Fetching ALL pairs data ({target_candles} candles each)...")
        results = {}
        
        for pair in self.pair_mapping.keys():
            logger.info(f"📊 Processing {pair}...")
            
            # Fetch data
            df = self.fetch_historical_candles(pair, target_candles=target_candles)
            
            if df is not None and not df.empty:
                # Save to database
                if self.save_to_database(pair, df):
                    results[pair] = True
                    logger.info(f"✅ {pair}: Successfully fetched and saved")
                else:
                    results[pair] = False
                    logger.error(f"❌ {pair}: Fetched but failed to save")
            else:
                results[pair] = False
                logger.error(f"❌ {pair}: Failed to fetch data")
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"📊 FETCH COMPLETE: {successful}/{total} pairs successful")
        
        if successful == total:
            logger.info("🎉 ALL CURRENCY PAIRS FETCHED SUCCESSFULLY!")
        else:
            failed_pairs = [pair for pair, success in results.items() if not success]
            logger.warning(f"⚠️ Failed pairs: {', '.join(failed_pairs)}")
        
        return results
    
    def load_from_database(self, symbol: str, limit: int = 5000) -> pd.DataFrame:
        """
        Load historical data from database
        """
        table_name = f"data_{symbol.replace('/', '_').replace('-', '_').lower()}"
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                query = f"""
                    SELECT * FROM {table_name} 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                df = pd.read_sql(query, conn, params=[limit])
                
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
                logger.info(f"📖 Loaded {len(df)} historical candles for {symbol} from database")
                return df
            else:
                logger.warning(f"⚠️ No data found in database for {symbol}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Error loading {symbol} from database: {str(e)}")
            return pd.DataFrame()
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the database
        """
        stats = {}
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                for pair in self.pair_mapping.keys():
                    table_name = f"data_{pair.replace('/', '_').replace('-', '_').lower()}"
                    
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    # Get date range
                    cursor.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table_name}")
                    min_date, max_date = cursor.fetchone()
                    
                    stats[pair] = {
                        'total_candles': count,
                        'date_range': f"{min_date} to {max_date}" if min_date else "No data"
                    }
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {str(e)}")
        
        return stats

# API Keys Configuration
TWELVEDATA_API_KEYS = [
    "d7b552b650a944b9be511980d28a207e",  # Original key
    "a4f4b744ea454eec86da0e1c0688bb86",  # Additional key 1
    "bd350e0aa30d441ca220f04256652b78"   # Additional key 2
]

# Test the fixed system
if __name__ == "__main__":
    print("🧪 TESTING FIXED HISTORICAL DATA MANAGER")
    print("="*60)
    
    # Initialize manager
    manager = FixedHistoricalDataManager(TWELVEDATA_API_KEYS, "test_fixed_db.db")
    
    # Test fetch all pairs
    results = manager.fetch_all_pairs_data(target_candles=50)
    
    # Test loading from database
    print("\n📖 TESTING DATABASE LOADING:")
    for pair in ['EUR/USD', 'GBP/USD', 'GBP/JPY']:  # Test a few pairs
        df = manager.load_from_database(pair, limit=10)
        if not df.empty:
            print(f"   ✅ {pair}: {len(df)} candles loaded")
        else:
            print(f"   ❌ {pair}: Failed to load")
    
    # Show database stats
    stats = manager.get_database_stats()
    print(f"\n📊 Database Stats: {stats}")
    
    # Cleanup
    if os.path.exists("test_fixed_db.db"):
        os.remove("test_fixed_db.db")
        print("\n🧹 Test database cleaned up")
