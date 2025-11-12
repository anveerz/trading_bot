#!/usr/bin/env python3
"""
HISTORICAL DATA MANAGER
Fetches and manages historical forex data from TwelveData API
Integrates with live trading system for comprehensive analysis
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

class HistoricalDataManager:
    """
    Manages historical data fetching, storage, and integration with live data
    """
    
    def __init__(self, twelvedata_api_key: str, database_path: str = "historical_data.db"):
        """
        Initialize the historical data manager
        
        Args:
            twelvedata_api_key: TwelveData API key
            database_path: SQLite database file path
        """
        self.api_key = twelvedata_api_key
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
        
    def _setup_database(self):
        """Create SQLite database tables for storing historical data"""
        with sqlite3.connect(self.database_path) as conn:
            # Create table for each currency pair
            for pair in self.pair_mapping.keys():
                table_name = f"data_{pair.replace('/', '_').replace('-', '_').lower()}"
                
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        timestamp DATETIME PRIMARY KEY,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for faster queries
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name}(timestamp)")
    
    def fetch_historical_candles(self, symbol: str, interval: str = "1min", 
                               target_count: int = 2500) -> pd.DataFrame:
        """
        Fetch historical candles from TwelveData API with enhanced rate limiting
        
        Args:
            symbol: Currency pair (e.g., "EUR/USD")
            interval: Time interval (1min, 5min, etc.)
            target_count: Target number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        if symbol not in self.pair_mapping:
            raise ValueError(f"Unsupported symbol: {symbol}")
        
        api_symbol = self.pair_mapping[symbol]
        
        # Calculate date range based on interval
        interval_minutes = self._get_interval_minutes(interval)
        estimated_minutes = target_count * interval_minutes
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(minutes=estimated_minutes)
        
        logger.info(f"Fetching {target_count} {interval} candles for {symbol}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        all_data = []
        current_start = start_date
        api_call_count = 0
        max_api_calls_per_minute = 8  # TwelveData free plan limit
        
        # Fetch data in chunks to handle rate limits
        while current_start < end_date:
            try:
                # Check rate limits
                if api_call_count >= max_api_calls_per_minute:
                    logger.warning(f"Rate limit reached for {symbol} - waiting 60 seconds")
                    time.sleep(60)  # Wait for rate limit reset
                    api_call_count = 0
                
                # Calculate chunk size (limit API calls)
                chunk_end = min(current_start + timedelta(days=3), end_date)  # Reduced to 3 days max per request
                
                params = {
                    'symbol': api_symbol,
                    'interval': interval,
                    'apikey': self.api_key,
                    'format': 'JSON',
                    'outputsize': 500,  # Limit response size
                    'start_date': int(current_start.timestamp()),
                    'end_date': int(chunk_end.timestamp())
                }
                
                url = f"{self.base_url}/time_series"
                response = requests.get(url, params=params, timeout=30)
                api_call_count += 1
                
                if response.status_code == 429:
                    logger.warning(f"Rate limit hit (429) for {symbol} - waiting 60 seconds")
                    time.sleep(60)
                    api_call_count = 0
                    continue
                
                if response.status_code != 200:
                    logger.error(f"API request failed: {response.status_code} - {response.text}")
                    break
                
                data = response.json()
                
                # Check for API errors
                if 'code' in data and data.get('code') != 200:
                    logger.error(f"API returned error: {data}")
                    if 'credits' in data.get('message', '').lower():
                        logger.warning("API credits issue - waiting 60 seconds")
                        time.sleep(60)
                        api_call_count = 0
                        continue
                    break
                
                if 'values' not in data or not data['values']:
                    logger.warning(f"No data returned for {symbol} - {interval}")
                    break
                
                # Process the data
                candles = []
                for value in data['values']:
                    try:
                        candle = {
                            'timestamp': pd.to_datetime(value['datetime']),
                            'open': float(value['open']),
                            'high': float(value['high']),
                            'low': float(value['low']),
                            'close': float(value['close']),
                            'volume': int(value.get('volume', 0))
                        }
                        candles.append(candle)
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Skipping invalid candle data: {e}")
                        continue
                
                if candles:
                    all_data.extend(candles)
                    logger.info(f"Fetched {len(candles)} candles for {symbol} - {len(all_data)} total")
                
                # Update start for next iteration
                current_start = chunk_end
                
                # Enhanced rate limiting (7.5 seconds between calls to be safe)
                time.sleep(7.5)
                
                # Check if we have enough data
                if len(all_data) >= target_count:
                    break
                    
            except requests.RequestException as e:
                logger.error(f"Request error for {symbol}: {e}")
                time.sleep(10)  # Wait longer on error
                break
            except Exception as e:
                logger.error(f"Unexpected error fetching data for {symbol}: {e}")
                break
        
        if not all_data:
            logger.error(f"No data fetched for {symbol}")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        df = df.set_index('timestamp').sort_index()
        
        # Take the most recent data
        df = df.tail(target_count)
        
        logger.info(f"Successfully fetched {len(df)} historical candles for {symbol}")
        return df
    
    def _get_interval_minutes(self, interval: str) -> int:
        """Convert interval string to minutes"""
        mapping = {
            '1min': 1,
            '5min': 5,
            '15min': 15,
            '30min': 30,
            '60min': 60,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return mapping.get(interval, 1)
    
    def save_to_database(self, df: pd.DataFrame, symbol: str, interval: str = "1min"):
        """
        Save historical data to SQLite database
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Currency pair
            interval: Time interval
        """
        table_name = f"data_{symbol.replace('/', '_').replace('-', '_').lower()}_{interval.replace('min', 'min')}"
        
        with self.db_lock, sqlite3.connect(self.database_path) as conn:
            df_reset = df.reset_index()
            df_reset.to_sql(table_name, conn, if_exists='replace', index=False)
            
        logger.info(f"Saved {len(df)} {interval} candles to database for {symbol} in table {table_name}")
    
    def load_from_database(self, symbol: str, limit: int = 5000) -> pd.DataFrame:
        """
        Load historical data from database
        
        Args:
            symbol: Currency pair
            limit: Maximum number of candles to load
            
        Returns:
            DataFrame with historical data
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
                logger.info(f"Loaded {len(df)} historical candles for {symbol} from database")
                return df
            else:
                logger.warning(f"No data found in database for {symbol}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading data from database for {symbol}: {e}")
            return pd.DataFrame()
    
    def load_5min_from_database(self, symbol: str, limit: int = 5000) -> pd.DataFrame:
        """
        Load 5-minute historical data from database
        
        Args:
            symbol: Currency pair
            limit: Maximum number of candles to load
            
        Returns:
            DataFrame with 5-minute historical data
        """
        table_name = f"data_{symbol.replace('/', '_').replace('-', '_').lower()}_5min"
        
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
                logger.info(f"Loaded {len(df)} 5-minute historical candles for {symbol} from database")
                return df
            else:
                logger.warning(f"No 5-minute data found in database for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading data from database for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_combined_data(self, symbol: str, live_data: Optional[pd.DataFrame] = None, 
                         target_total: int = 3000) -> pd.DataFrame:
        """
        Get combined historical + live data for comprehensive analysis
        
        Args:
            symbol: Currency pair
            live_data: Optional DataFrame with live data
            target_total: Target total candles (historical + live)
            
        Returns:
            DataFrame with combined data
        """
        # Load historical data
        historical = self.load_from_database(symbol, limit=min(target_total, 2500))
        
        if live_data is not None and not live_data.empty:
            # Combine historical + live data
            combined = pd.concat([historical, live_data]).drop_duplicates()
            combined = combined.sort_index()
            
            # Take the most recent data
            combined = combined.tail(target_total)
            
            logger.info(f"Combined data: {len(historical)} historical + {len(live_data)} live = {len(combined)} total for {symbol}")
            return combined
        else:
            logger.info(f"Using only historical data: {len(historical)} candles for {symbol}")
            return historical
    
    def resample_to_5min(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 1-minute data to 5-minute candles
        
        Args:
            df: DataFrame with 1-minute data
            
        Returns:
            DataFrame with 5-minute OHLCV data
        """
        if df.empty or 'close' not in df.columns:
            return pd.DataFrame()
        
        # Resample to 5-minute candles
        resampled = df.resample('5T').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def create_comprehensive_dataset(self, symbol: str, live_tick_data: Dict) -> Dict[str, pd.DataFrame]:
        """
        Create comprehensive dataset combining historical data with live tick updates
        
        Args:
            symbol: Currency pair
            live_tick_data: Dictionary with current price and timestamp
            
        Returns:
            Dictionary with different timeframe DataFrames
        """
        try:
            # Get historical data
            historical = self.load_from_database(symbol, limit=2500)
            
            if historical.empty:
                logger.warning(f"No historical data available for {symbol}")
                return {}
            
            # Add live price to create new candle
            current_time = pd.to_datetime(live_tick_data.get('timestamp', datetime.now(timezone.utc)))
            current_price = live_tick_data.get('price')
            
            if current_price is None:
                return self._create_timeframe_datasets(historical)
            
            # Get current 1-minute candle
            minute_bucket = current_time.replace(second=0, microsecond=0)
            
            # Check if we need to create a new candle or update existing
            if not historical.empty and historical.index[-1].floor('1T') >= minute_bucket:
                # Update existing candle
                latest = historical.iloc[-1].copy()
                latest['high'] = max(latest['high'], current_price)
                latest['low'] = min(latest['low'], current_price)
                latest['close'] = current_price
                latest_volume = historical.iloc[-1]['volume'] + 1
                latest['volume'] = latest_volume
                
                # Update the last candle
                historical.iloc[-1] = latest
            else:
                # Create new candle
                open_price = current_price
                if len(historical) > 0:
                    open_price = historical.iloc[-1]['close']
                
                new_candle = pd.Series({
                    'open': open_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': 1
                }, name=minute_bucket)
                
                historical = pd.concat([historical, new_candle.to_frame().T])
            
            # Keep only recent data
            historical = historical.tail(3000)
            
            # Create different timeframes
            datasets = self._create_timeframe_datasets(historical)
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dataset for {symbol}: {e}")
            return {}
    
    def _create_timeframe_datasets(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create datasets for different timeframes from 1-minute data"""
        if df.empty:
            return {}
        
        datasets = {
            '1min': df,
            '5min': df.resample('5T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min', 
                'close': 'last',
                'volume': 'sum'
            }).dropna(),
            '15min': df.resample('15T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last', 
                'volume': 'sum'
            }).dropna(),
            '20min': df.resample('20T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna(),
            '30min': df.resample('30T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna(),
            '60min': df.resample('60T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        }
        
        return {k: v for k, v in datasets.items() if not v.empty}
    
    def fetch_all_pairs_data(self, target_candles: int = 5000) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all currency pairs
        
        Args:
            target_candles: Target number of candles per pair
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in self.pair_mapping.keys():
            try:
                logger.info(f"Fetching historical data for {symbol}...")
                
                # Fetch 5-minute data
                df_5min = self.fetch_historical_candles(symbol, "5min", target_candles)
                
                if not df_5min.empty:
                    # Save to database
                    self.save_to_database(df_5min, symbol, "5min")
                    results[symbol] = df_5min
                    
                    logger.info(f"✅ Successfully fetched {len(df_5min)} 5-minute candles for {symbol}")
                else:
                    logger.error(f"❌ Failed to fetch data for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
        
        return results

# Test function
if __name__ == "__main__":
    # Test the historical data manager
    API_KEY = "d7b552b650a944b9be511980d28a207e"  # Use the actual API key from config
    
    manager = HistoricalDataManager(API_KEY)
    
    # Test fetching data for EUR/USD
    print("Testing historical data fetch for EUR/USD...")
    df = manager.fetch_historical_candles("EUR/USD", "1min", 100)  # Test with 100 candles first
    
    if not df.empty:
        print(f"✅ Successfully fetched {len(df)} candles")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Price range: {df['close'].min():.5f} to {df['close'].max():.5f}")
        
        # Test saving to database
        manager.save_to_database(df, "EUR/USD")
        print("✅ Data saved to database")
        
        # Test loading from database
        loaded = manager.load_from_database("EUR/USD")
        print(f"✅ Loaded {len(loaded)} candles from database")
        
    else:
        print("❌ Failed to fetch data")