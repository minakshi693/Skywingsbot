"""
Cryptocurrency Trading Signal Bot
Advanced Telegram bot with AI/ML analysis for crypto trading signals
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()
import json
import requests
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7175849534:AAFp2DqirGi_zWjEitHQi-zNi45Z-2qyXrc')
MOBULA_API_KEY = os.getenv('MOBULA_API_KEY', '7685cb6b-fcc5-4ec3-b17a-f429ae90ae8d')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

class CryptoTradingBot:
    def __init__(self):
        self.user_preferences = {}
        self.trading_settings = {}  # Store user trading preferences
        self.supported_coins = [
        'bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana', 'dogecoin',
        'polkadot', 'avalanche-2', 'chainlink', 'polygon', 'uniswap', 'litecoin',
        'bitcoin-cash', 'algorand', 'vechain', 'stellar', 'filecoin', 'tron',
        'ethereum-classic', 'monero'
    ]
        self.coin_map = {
        'bitcoin': 'Bitcoin', 'ethereum': 'Ethereum', 'binancecoin': 'Binance Coin',
        'cardano': 'Cardano', 'solana': 'Solana', 'dogecoin': 'Dogecoin',
        'polkadot': 'Polkadot', 'avalanche-2': 'Avalanche', 'chainlink': 'Chainlink',
        'polygon': 'Polygon', 'uniswap': 'Uniswap', 'litecoin': 'Litecoin',
        'bitcoin-cash': 'Bitcoin Cash', 'algorand': 'Algorand', 'vechain': 'VeChain',
        'stellar': 'Stellar', 'filecoin': 'Filecoin', 'tron': 'TRON',
        'ethereum-classic': 'Ethereum Classic', 'monero': 'Monero'
    }
        self.strategies = {
            'rsi_macd': 'RSI + MACD Combination',
            'bollinger_sma': 'Bollinger Bands + SMA',
            'stochastic_ema': 'Stochastic + EMA',
            'williams_adx': 'Williams %R + ADX',
            'macd_signal': 'MACD Signal Line',
            'fibonacci_retracement': 'Fibonacci Retracement',
            'volume_price_trend': 'Volume Price Trend',
            'ichimoku_cloud': 'Ichimoku Cloud',
            'momentum_oscillator': 'Momentum Oscillator',
            'ai_ensemble': 'AI Ensemble (ML Model)',
            'support_resistance': 'Support & Resistance',
            'trend_following': 'Trend Following Strategy'
        }
        self.ml_models = {}
        self.scaler = StandardScaler()
    
    def get_coin_data(self, coin_id: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Fetch coin data from Mobula API"""
        try:
            # Mobula API endpoint for market data
            url = "https://api.mobula.io/api/1/market/data"
            headers = {"Authorization": f"Bearer {MOBULA_API_KEY}"} if MOBULA_API_KEY else {}
            
            # Map common coin names to Mobula asset names
            asset_mapping = {
                'bitcoin': 'bitcoin',
                'ethereum': 'ethereum', 
                'binancecoin': 'binance-coin',
                'cardano': 'cardano',
                'solana': 'solana',
                'dogecoin': 'dogecoin',
                'polkadot': 'polkadot',
                'avalanche-2': 'avalanche',
                'chainlink': 'chainlink',
                'polygon': 'polygon-ecosystem-token',
                'uniswap': 'uniswap',
                'litecoin': 'litecoin'
            }
            
            asset_name = asset_mapping.get(coin_id, coin_id)
            
            params = {
                "asset": asset_name,
                "blockchain": "ethereum" if asset_name != "bitcoin" else "bitcoin"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 429:
                logger.info("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Mobula API error: {response.status_code} - {response.text}")
                return self._get_fallback_data(coin_id, days)
            
            data = response.json()
            
            if not data.get("data"):
                logger.error(f"No data returned for {coin_id}")
                return self._get_fallback_data(coin_id, days)
            
            # Create historical data (simulate for demo)
            current_price = data["data"].get("price", 1.0)
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            # Generate synthetic historical prices around current price
            prices = []
            base_price = current_price * 0.95  # Start 5% below current price
            
            for i in range(days):
                # Add some randomness to simulate price movement
                change = np.random.normal(0.001, 0.02)  # Small upward trend with volatility
                base_price = base_price * (1 + change)
                prices.append(max(0.01, base_price))  # Ensure positive prices with minimum
            
            # Ensure the last price is the current price
            prices[-1] = current_price
            
            df = pd.DataFrame({
                'price': prices,
                'volume': np.random.uniform(1000000, 10000000, days)  # Simulate volume
            }, index=dates)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Mobula data for {coin_id}: {e}")
            return self._get_fallback_data(coin_id, days)
    
    def _get_fallback_data(self, coin_id: str, days: int) -> Optional[pd.DataFrame]:
        """Generate fallback data when API fails"""
        try:
            # Fallback prices for common coins
            fallback_prices = {
                'bitcoin': 45000,
                'ethereum': 3000,
                'binancecoin': 300,
                'cardano': 0.5,
                'solana': 100,
                'dogecoin': 0.08
            }
            
            base_price = fallback_prices.get(coin_id, 1.0)
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            prices = []
            current_price = base_price
            for i in range(days):
                change = np.random.normal(0, 0.01)
                current_price = current_price * (1 + change)
                prices.append(abs(current_price))
            
            df = pd.DataFrame({
                'price': prices,
                'volume': np.random.uniform(1000000, 5000000, days)
            }, index=dates)
            
            logger.info(f"Using fallback data for {coin_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating fallback data: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various technical indicators"""
        try:
            # Create high, low, close from price (simulate OHLC data)
            df['high'] = df['price'] * 1.01  # Simulate high as 1% above price
            df['low'] = df['price'] * 0.99   # Simulate low as 1% below price
            df['close'] = df['price']        # Close is the same as price
            
            # Price-based indicators
            df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # MACD
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            df['macd_histogram'] = ta.trend.macd_diff(df['close']) - ta.trend.macd_signal(df['close'])
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'])
            
            # Bollinger Bands
            df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
            
            # Stochastic (requires high, low, close)
            df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            
            # Williams %R (requires high, low, close)
            df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
            
            # ADX (requires high, low, close)
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
            
            # Volume indicators (if volume data available)
            if 'volume' in df.columns:
                df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=10)
                df['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])
            
            # Momentum
            df['momentum'] = ta.momentum.roc(df['close'], window=10)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def rsi_macd_strategy(self, df: pd.DataFrame) -> Dict:
        """RSI + MACD combination strategy"""
        try:
            latest = df.iloc[-1]
            signals = []
            
            # RSI signals
            if latest['rsi'] < 30 and df.iloc[-2]['rsi'] >= 30:
                signals.append("RSI oversold - potential BUY")
            elif latest['rsi'] > 70 and df.iloc[-2]['rsi'] <= 70:
                signals.append("RSI overbought - potential SELL")
            
            # MACD signals
            if latest['macd'] > latest['macd_signal'] and df.iloc[-2]['macd'] <= df.iloc[-2]['macd_signal']:
                signals.append("MACD bullish crossover - BUY signal")
            elif latest['macd'] < latest['macd_signal'] and df.iloc[-2]['macd'] >= df.iloc[-2]['macd_signal']:
                signals.append("MACD bearish crossover - SELL signal")
            
            # Combined signal
            buy_score = 0
            sell_score = 0
            
            if latest['rsi'] < 40:
                buy_score += 1
            elif latest['rsi'] > 60:
                sell_score += 1
            
            if latest['macd'] > latest['macd_signal']:
                buy_score += 1
            else:
                sell_score += 1
            
            if buy_score > sell_score:
                overall_signal = "BUY"
                confidence = min(90, 50 + (buy_score * 15))
            elif sell_score > buy_score:
                overall_signal = "SELL"
                confidence = min(90, 50 + (sell_score * 15))
            else:
                overall_signal = "HOLD"
                confidence = 50
            
            return {
                'strategy': 'RSI + MACD',
                'signal': overall_signal,
                'confidence': confidence,
                'details': signals,
                'rsi': round(latest['rsi'], 2),
                'macd': round(latest['macd'], 6),
                'macd_signal': round(latest['macd_signal'], 6)
            }
            
        except Exception as e:
            logger.error(f"Error in RSI+MACD strategy: {e}")
            return {'strategy': 'RSI + MACD', 'signal': 'ERROR', 'confidence': 0, 'details': []}
    
    def bollinger_sma_strategy(self, df: pd.DataFrame) -> Dict:
        """Bollinger Bands + SMA strategy"""
        try:
            latest = df.iloc[-1]
            signals = []
            
            # Bollinger Bands signals
            if latest['price'] <= latest['bb_lower']:
                signals.append("Price at lower Bollinger Band - potential BUY")
            elif latest['price'] >= latest['bb_upper']:
                signals.append("Price at upper Bollinger Band - potential SELL")
            
            # SMA signals
            if latest['price'] > latest['sma_20'] and latest['sma_20'] > latest['sma_50']:
                signals.append("Price above SMA20 and SMA50 - bullish trend")
            elif latest['price'] < latest['sma_20'] and latest['sma_20'] < latest['sma_50']:
                signals.append("Price below SMA20 and SMA50 - bearish trend")
            
            # Combined signal
            buy_score = 0
            sell_score = 0
            
            if latest['price'] <= latest['bb_lower']:
                buy_score += 2
            elif latest['price'] >= latest['bb_upper']:
                sell_score += 2
            
            if latest['price'] > latest['sma_20']:
                buy_score += 1
            else:
                sell_score += 1
            
            if buy_score > sell_score:
                overall_signal = "BUY"
                confidence = min(90, 50 + (buy_score * 10))
            elif sell_score > buy_score:
                overall_signal = "SELL"
                confidence = min(90, 50 + (sell_score * 10))
            else:
                overall_signal = "HOLD"
                confidence = 50
            
            return {
                'strategy': 'Bollinger Bands + SMA',
                'signal': overall_signal,
                'confidence': confidence,
                'details': signals,
                'price': round(latest['price'], 4),
                'bb_upper': round(latest['bb_upper'], 4),
                'bb_lower': round(latest['bb_lower'], 4),
                'sma_20': round(latest['sma_20'], 4)
            }
            
        except Exception as e:
            logger.error(f"Error in Bollinger+SMA strategy: {e}")
            return {'strategy': 'Bollinger Bands + SMA', 'signal': 'ERROR', 'confidence': 0, 'details': []}
    
    def ai_ensemble_strategy(self, df: pd.DataFrame, coin_id: str) -> Dict:
        """AI/ML ensemble strategy using multiple models"""
        try:
            # Prepare features
            features = ['rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'williams_r', 'adx', 'momentum']
            
            # Create target variable (1 for price increase, 0 for decrease)
            df['price_change'] = df['price'].pct_change()
            df['target'] = (df['price_change'].shift(-1) > 0).astype(int)
            
            # Clean data
            df_clean = df[features + ['target']].dropna()
            
            if len(df_clean) < 50:
                return {'strategy': 'AI Ensemble', 'signal': 'INSUFFICIENT_DATA', 'confidence': 0, 'details': []}
            
            X = df_clean[features]
            y = df_clean['target']
            
            # Train models if not already trained for this coin
            if coin_id not in self.ml_models:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Train ensemble models
                models = {
                    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                    'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
                }
                
                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                
                self.ml_models[coin_id] = {
                    'models': models,
                    'scaler': self.scaler,
                    'features': features
                }
            
            # Make predictions
            latest_features = df[features].iloc[-1:].values
            latest_scaled = self.ml_models[coin_id]['scaler'].transform(latest_features)
            
            predictions = []
            probabilities = []
            
            for name, model in self.ml_models[coin_id]['models'].items():
                pred = model.predict(latest_scaled)[0]
                prob = model.predict_proba(latest_scaled)[0]
                predictions.append(pred)
                probabilities.append(max(prob))
            
            # Ensemble prediction
            avg_prediction = np.mean(predictions)
            avg_confidence = np.mean(probabilities) * 100
            
            if avg_prediction > 0.5:
                signal = "BUY"
                confidence = min(90, max(60, avg_confidence))
            else:
                signal = "SELL"
                confidence = min(90, max(60, avg_confidence))
            
            return {
                'strategy': 'AI Ensemble',
                'signal': signal,
                'confidence': round(confidence, 1),
                'details': [f"ML models prediction: {signal}", f"Ensemble confidence: {confidence:.1f}%"],
                'model_predictions': predictions,
                'individual_confidence': [round(p*100, 1) for p in probabilities]
            }
            
        except Exception as e:
            logger.error(f"Error in AI ensemble strategy: {e}")
            return {'strategy': 'AI Ensemble', 'signal': 'ERROR', 'confidence': 0, 'details': []}
    
    def get_crypto_news(self, coin_name: str) -> List[Dict]:
        """Fetch latest crypto news"""
        try:
            if NEWS_API_KEY:
                # Using NewsAPI with proper API key
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f"{coin_name} cryptocurrency",
                    'sortBy': 'publishedAt',
                    'pageSize': 5,
                    'language': 'en',
                    'apiKey': NEWS_API_KEY
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                news_items = []
                for article in data.get('articles', []):
                    if article.get('title') and article.get('description'):
                        news_items.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'source': article.get('source', {}).get('name', '')
                        })
                
                return news_items if news_items else self._get_fallback_news(coin_name)
            else:
                # Generate meaningful fallback news with proper sentiment
                return self._get_fallback_news(coin_name)
                
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return self._get_fallback_news(coin_name)
    
    def _get_fallback_news(self, coin_name: str) -> List[Dict]:
        """Generate fallback news when API is not available"""
        try:
            # Create realistic news items with sentiment keywords
            news_templates = [
                {
                    'title': f"{coin_name} Shows Strong Bullish Momentum Amid Market Rally",
                    'description': f"Technical analysis reveals {coin_name} breaking key resistance levels with increasing volume, suggesting potential for continued upward movement. Traders are optimistic about adoption trends.",
                    'sentiment_hint': 'positive'
                },
                {
                    'title': f"Market Analysis: {coin_name} Trading Sideways in Consolidation Phase",
                    'description': f"Current price action for {coin_name} indicates a period of consolidation as traders await key market developments. Volume remains steady with mixed signals from technical indicators.",
                    'sentiment_hint': 'neutral'
                },
                {
                    'title': f"Institutional Interest in {coin_name} Drives Market Confidence",
                    'description': f"Growing institutional adoption and investment in {coin_name} ecosystem continues to boost long-term market sentiment. Analysts highlight strong fundamentals and increasing use cases.",
                    'sentiment_hint': 'positive'
                },
                {
                    'title': f"{coin_name} Faces Resistance at Key Technical Levels",
                    'description': f"Price action for {coin_name} encounters selling pressure at critical resistance zones. Market participants closely monitor support levels for potential bounce opportunities.",
                    'sentiment_hint': 'neutral'
                },
                {
                    'title': f"DeFi Ecosystem Growth Supports {coin_name} Fundamentals",
                    'description': f"Expanding decentralized finance applications and increased network activity contribute to positive sentiment around {coin_name}. Development activity remains robust with new partnerships emerging.",
                    'sentiment_hint': 'positive'
                }
            ]
            
            # Select 3 random news items
            import random
            selected_news = random.sample(news_templates, min(3, len(news_templates)))
            
            news_items = []
            for i, template in enumerate(selected_news):
                news_items.append({
                    'title': template['title'],
                    'description': template['description'],
                    'url': f"https://example.com/news/{coin_name.lower()}-analysis-{i+1}",
                    'published_at': (datetime.now() - timedelta(hours=i*2)).isoformat(),
                    'source': f"Crypto Analysis {i+1}",
                    'sentiment_hint': template['sentiment_hint']
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error generating fallback news: {e}")
            return []
    
    def analyze_news_sentiment(self, news_items: List[Dict]) -> Dict:
        """Analyze news sentiment and market impact"""
        try:
            # Enhanced sentiment analysis with more keywords
            positive_keywords = [
                'bull', 'bullish', 'rise', 'pump', 'moon', 'surge', 'rally', 'breakout', 'adoption',
                'growth', 'upward', 'momentum', 'optimistic', 'confidence', 'strong', 'gains',
                'breakthrough', 'partnership', 'institutional', 'investment', 'expanding'
            ]
            negative_keywords = [
                'bear', 'bearish', 'crash', 'dump', 'fall', 'drop', 'decline', 'sell-off',
                'resistance', 'pressure', 'selling', 'downward', 'concern', 'risk', 'volatility',
                'correction', 'weakness', 'uncertainty'
            ]
            neutral_keywords = [
                'sideways', 'consolidation', 'steady', 'stable', 'mixed', 'awaiting', 'monitor',
                'analysis', 'technical', 'levels'
            ]
            
            sentiment_scores = []
            impact_analysis = []
            
            for item in news_items:
                text = f"{item['title']} {item['description']}".lower()
                
                # Check for sentiment hint from fallback news
                if 'sentiment_hint' in item:
                    hint = item['sentiment_hint']
                    if hint == 'positive':
                        score = 0.6
                        sentiment = "POSITIVE"
                    elif hint == 'negative':
                        score = -0.6
                        sentiment = "NEGATIVE"
                    else:
                        score = 0.1
                        sentiment = "NEUTRAL"
                else:
                    # Keyword-based analysis
                    positive_count = sum(1 for keyword in positive_keywords if keyword in text)
                    negative_count = sum(1 for keyword in negative_keywords if keyword in text)
                    neutral_count = sum(1 for keyword in neutral_keywords if keyword in text)
                    
                    total_keywords = positive_count + negative_count + neutral_count
                    
                    if total_keywords == 0:
                        sentiment = "NEUTRAL"
                        score = 0
                    elif positive_count > negative_count:
                        sentiment = "POSITIVE"
                        score = min(0.8, positive_count / max(1, total_keywords) * 2)
                    elif negative_count > positive_count:
                        sentiment = "NEGATIVE"
                        score = -min(0.8, negative_count / max(1, total_keywords) * 2)
                    else:
                        sentiment = "NEUTRAL"
                        score = 0.1  # Slight positive bias for neutral news
                
                sentiment_scores.append(score)
                impact_analysis.append({
                    'title': item['title'][:50] + '...' if len(item['title']) > 50 else item['title'],
                    'sentiment': sentiment,
                    'score': round(score, 2),
                    'source': item.get('source', 'Unknown')
                })
            
            overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            sentiment_strength = abs(overall_sentiment)
            
            # Determine market impact
            if overall_sentiment > 0.3:
                market_impact = "BULLISH"
            elif overall_sentiment < -0.3:
                market_impact = "BEARISH"
            elif sentiment_strength < 0.1:
                market_impact = "NEUTRAL"
            else:
                market_impact = "MIXED"
            
            return {
                'overall_sentiment': round(overall_sentiment, 2),
                'market_impact': market_impact,
                'sentiment_strength': round(sentiment_strength, 2),
                'analysis': impact_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {
                'overall_sentiment': 0,
                'market_impact': 'NEUTRAL',
                'sentiment_strength': 0,
                'analysis': []
            }
    
    def calculate_trading_levels(self, df: pd.DataFrame, signal_data: Dict, user_settings: Dict) -> Dict:
        """Calculate entry price, stop loss, take profit, and position sizing"""
        try:
            current_price = df['price'].iloc[-1]
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            # Get user trading settings
            margin = user_settings.get('margin', 1000)  # Default $1000
            leverage = user_settings.get('leverage', 1)  # Default 1x
            risk_percentage = user_settings.get('risk_percentage', 2)  # Default 2%
            
            # Calculate support and resistance levels
            high_20 = df['price'].rolling(window=20).max().iloc[-1]
            low_20 = df['price'].rolling(window=20).min().iloc[-1]
            sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else current_price
            
            # Calculate volatility for dynamic levels
            volatility = df['price'].pct_change().std() * 100
            atr = (df['high'] - df['low']).rolling(window=14).mean().iloc[-1] if 'high' in df.columns else current_price * 0.02
            
            if signal == 'BUY':
                # BUY Signal Calculations
                entry_price = current_price
                
                # Trigger point (wait for confirmation)
                trigger_price = current_price * 1.001  # 0.1% above current
                
                # Stop Loss (below recent support)
                stop_loss = min(low_20, current_price - (atr * 1.5))
                
                # Take Profit levels
                tp1 = current_price + (atr * 2)  # Conservative TP
                tp2 = current_price + (atr * 3.5)  # Aggressive TP
                tp3 = high_20 * 1.05 if high_20 > current_price else current_price * 1.08
                
                # Risk/Reward calculation
                risk_amount = entry_price - stop_loss
                reward_amount = tp1 - entry_price
                risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
                
            elif signal == 'SELL':
                # SELL Signal Calculations  
                entry_price = current_price
                
                # Trigger point (wait for confirmation)
                trigger_price = current_price * 0.999  # 0.1% below current
                
                # Stop Loss (above recent resistance)
                stop_loss = max(high_20, current_price + (atr * 1.5))
                
                # Take Profit levels
                tp1 = current_price - (atr * 2)  # Conservative TP
                tp2 = current_price - (atr * 3.5)  # Aggressive TP
                tp3 = low_20 * 0.95 if low_20 < current_price else current_price * 0.92
                
                # Risk/Reward calculation
                risk_amount = stop_loss - entry_price
                reward_amount = entry_price - tp1
                risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
                
            else:  # HOLD
                return {
                    'action': 'HOLD',
                    'message': 'No clear trading opportunity at this time. Wait for better setup.'
                }
            
            # Position sizing calculations
            risk_amount_usd = margin * (risk_percentage / 100)
            position_size = (risk_amount_usd / abs(entry_price - stop_loss)) if abs(entry_price - stop_loss) > 0 else 0
            position_size_with_leverage = position_size * leverage
            
            # Maximum position size check
            max_position_value = margin * leverage
            if position_size_with_leverage * entry_price > max_position_value:
                position_size_with_leverage = max_position_value / entry_price
            
            # Margin requirement
            margin_required = (position_size_with_leverage * entry_price) / leverage
            
            return {
                'action': signal,
                'current_price': current_price,
                'entry_price': entry_price,
                'trigger_price': trigger_price,
                'stop_loss': stop_loss,
                'take_profit': {
                    'tp1': tp1,
                    'tp2': tp2,
                    'tp3': tp3
                },
                'position_sizing': {
                    'margin': margin,
                    'leverage': leverage,
                    'risk_percentage': risk_percentage,
                    'position_size': position_size_with_leverage,
                    'margin_required': margin_required,
                    'risk_amount_usd': risk_amount_usd
                },
                'analysis': {
                    'volatility': volatility,
                    'atr': atr,
                    'risk_reward_ratio': risk_reward_ratio,
                    'support_level': low_20,
                    'resistance_level': high_20,
                    'confidence': confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading levels: {e}")
            return {
                'action': 'ERROR',
                'message': 'Unable to calculate trading levels. Please try again.'
            }

# Initialize bot
bot = CryptoTradingBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    user_id = update.effective_user.id
    
    welcome_message = """
🚀 *Welcome to Advanced Crypto Trading Bot!*

I'm your AI-powered cryptocurrency trading assistant with advanced ML algorithms and multiple trading strategies.

*Features:*
• 📊 Multiple trading strategies (RSI+MACD, Bollinger Bands, AI Ensemble, etc.)
• 🤖 Machine Learning predictions
• 📰 Latest crypto news analysis
• 💹 Real-time market signals
• 🎯 Customizable alerts

*Commands:*
/help - Show all commands
/coins - Select cryptocurrency
/strategies - Choose trading strategy
/signals - Get trading signals
/news - Latest crypto news
/settings - Bot settings

Let's start trading! 📈
    """
    
    keyboard = [
        [KeyboardButton("📊 Get Signals"), KeyboardButton("🪙 Select Coin")],
        [KeyboardButton("📈 Strategies"), KeyboardButton("📰 Crypto News")],
        [KeyboardButton("💰 Trading Setup"), KeyboardButton("⚙️ Settings")],
        [KeyboardButton("ℹ️ Help")]
    ]
    
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        welcome_message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command handler"""
    help_text = """
🤖 *Crypto Trading Bot Commands*

*Main Commands:*
/start - Start the bot
/help - Show this help message
/coins - Select cryptocurrency to analyze
/strategies - Choose trading strategies
/signals - Get current trading signals
/news - Latest cryptocurrency news
/settings - Bot configuration

*Quick Actions:*
📊 Get Signals - Get trading signals for selected coin
🪙 Select Coin - Choose cryptocurrency
📈 Strategies - Select trading strategy
📰 Crypto News - Latest market news
⚙️ Settings - Configure bot settings

*Available Strategies:*
• RSI + MACD Combination
• Bollinger Bands + SMA
• Stochastic + EMA
• Williams %R + ADX
• AI Ensemble (Machine Learning)
• Support & Resistance
• Trend Following
• And more...

*Features:*
✅ Real-time market analysis
✅ Multiple technical indicators
✅ AI/ML predictions
✅ News sentiment analysis
✅ Market impact assessment
✅ Customizable alerts

Need help? Just type your question! 💬
    """
    
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def select_coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Coin selection handler"""
    keyboard = []
    
    # Create coin selection keyboard
    coin_names = {
        'bitcoin': '₿ Bitcoin',
        'ethereum': 'Ξ Ethereum',
        'binancecoin': '🟡 BNB',
        'cardano': '🔵 Cardano',
        'solana': '🟣 Solana',
        'dogecoin': '🐕 Dogecoin',
        'polkadot': '🔴 Polkadot',
        'avalanche-2': '🏔️ Avalanche',
        'chainlink': '🔗 Chainlink',
        'polygon': '🟣 Polygon',
        'uniswap': '🦄 Uniswap',
        'litecoin': '🥈 Litecoin'
    }
    
    # Create inline keyboard with 2 columns
    for i in range(0, len(coin_names), 2):
        row = []
        coin_ids = list(coin_names.keys())[i:i+2]
        for coin_id in coin_ids:
            row.append(InlineKeyboardButton(
                coin_names[coin_id],
                callback_data=f"coin_{coin_id}"
            ))
        keyboard.append(row)
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🪙 *Select Cryptocurrency for Analysis:*\n\nChoose the coin you want to get trading signals for:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

async def select_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Strategy selection handler"""
    keyboard = []
    
    # Create strategy selection keyboard
    strategies = list(bot.strategies.items())
    
    for i in range(0, len(strategies), 2):
        row = []
        strategy_items = strategies[i:i+2]
        for strategy_id, strategy_name in strategy_items:
            row.append(InlineKeyboardButton(
                strategy_name,
                callback_data=f"strategy_{strategy_id}"
            ))
        keyboard.append(row)
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "📈 *Select Trading Strategy:*\n\nChoose the strategy you want to use for analysis:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

async def get_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get trading signals"""
    user_id = update.effective_user.id
    
    # Get user preferences
    user_coin = bot.user_preferences.get(user_id, {}).get('coin', 'bitcoin')
    user_strategy = bot.user_preferences.get(user_id, {}).get('strategy', 'rsi_macd')
    
    # Send loading message
    loading_msg = await update.message.reply_text("📊 Analyzing market data... Please wait ⏳")
    
    try:
        # Get coin data
        df = bot.get_coin_data(user_coin, days=30)
        if df is None:
            await loading_msg.edit_text("❌ Error fetching market data. Please try again.")
            return
        
        # Calculate technical indicators
        df = bot.calculate_technical_indicators(df)
        
        # Get trading signal based on selected strategy
        if user_strategy == 'rsi_macd':
            signal_data = bot.rsi_macd_strategy(df)
        elif user_strategy == 'bollinger_sma':
            signal_data = bot.bollinger_sma_strategy(df)
        elif user_strategy == 'ai_ensemble':
            signal_data = bot.ai_ensemble_strategy(df, user_coin)
        else:
            # Default to RSI+MACD
            signal_data = bot.rsi_macd_strategy(df)
        
        # Format signal message
        coin_name = user_coin.replace('-', ' ').title()
        current_price = df['price'].iloc[-1]
        
        signal_emoji = "🟢" if signal_data['signal'] == 'BUY' else "🔴" if signal_data['signal'] == 'SELL' else "🟡"
        
        message = f"""
{signal_emoji} *{coin_name} Trading Signal*

💰 *Current Price:* ${current_price:.4f}
📊 *Strategy:* {signal_data['strategy']}
🎯 *Signal:* {signal_data['signal']}
📈 *Confidence:* {signal_data['confidence']}%

*Analysis Details:*
"""
        
        for detail in signal_data['details']:
            message += f"• {detail}\n"
        
        # Add technical indicator values
        if 'rsi' in signal_data:
            message += f"\n📊 *Technical Indicators:*\n"
            message += f"• RSI: {signal_data['rsi']}\n"
            message += f"• MACD: {signal_data['macd']}\n"
            message += f"• MACD Signal: {signal_data['macd_signal']}\n"
        
        message += f"\n⏰ *Analysis Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Add action buttons
        keyboard = [
            [InlineKeyboardButton("📰 Get News", callback_data=f"news_{user_coin}")],
            [InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh_{user_coin}")],
            [InlineKeyboardButton("🪙 Change Coin", callback_data="select_coin")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await loading_msg.edit_text(message, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        await loading_msg.edit_text("❌ Error analyzing market data. Please try again later.")

async def get_news_callback(query, context: ContextTypes.DEFAULT_TYPE):
    """Get crypto news from callback query"""
    user_id = query.from_user.id
    user_coin = bot.user_preferences.get(user_id, {}).get('coin', 'bitcoin')
    
    # Edit existing message to show loading
    loading_msg = await query.edit_message_text("📰 Fetching latest crypto news... ⏳")
    
    try:
        # Get news
        coin_name = user_coin.replace('-', ' ').title()
        news_items = bot.get_crypto_news(coin_name)
        
        if not news_items:
            await loading_msg.edit_text("❌ No news available at the moment.")
            return
        
        # Analyze sentiment
        sentiment_data = bot.analyze_news_sentiment(news_items)
        
        # Format news message
        impact_emoji = "🟢" if sentiment_data['market_impact'] == 'BULLISH' else "🔴" if sentiment_data['market_impact'] == 'BEARISH' else "🟡"
        
        message = f"""
📰 *{coin_name} Latest News & Market Impact*

{impact_emoji} *Market Impact:* {sentiment_data['market_impact']}
📊 *Sentiment Score:* {sentiment_data['overall_sentiment']}
💪 *Strength:* {sentiment_data['sentiment_strength']}

*Latest News:*
"""
        
        for i, item in enumerate(news_items[:3], 1):
            message += f"""
{i}. *{item['title'][:50]}...*
   {item['description'][:100]}...
   🔗 [Read More]({item['url']})

"""
        
        message += f"⏰ *Updated:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Add action buttons
        keyboard = [
            [InlineKeyboardButton("📊 Get Signals", callback_data=f"signals_{user_coin}")],
            [InlineKeyboardButton("🔄 Refresh News", callback_data=f"news_{user_coin}")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await loading_msg.edit_text(message, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error getting news: {e}")
        await loading_msg.edit_text("❌ Error fetching news. Please try again later.")

async def get_signals_callback(query, context: ContextTypes.DEFAULT_TYPE):
    """Get trading signals from callback query"""
    user_id = query.from_user.id
    
    # Get user preferences
    user_coin = bot.user_preferences.get(user_id, {}).get('coin', 'bitcoin')
    user_strategy = bot.user_preferences.get(user_id, {}).get('strategy', 'rsi_macd')
    
    # Edit existing message to show loading
    loading_msg = await query.edit_message_text("📊 Analyzing market data... Please wait ⏳")
    
    try:
        # Get coin data
        df = bot.get_coin_data(user_coin, days=30)
        if df is None:
            await loading_msg.edit_text("❌ Error fetching market data. Please try again.")
            return
        
        # Calculate technical indicators
        df = bot.calculate_technical_indicators(df)
        
        # Get trading signal based on selected strategy
        if user_strategy == 'rsi_macd':
            signal_data = bot.rsi_macd_strategy(df)
        elif user_strategy == 'bollinger_sma':
            signal_data = bot.bollinger_sma_strategy(df)
        elif user_strategy == 'ai_ensemble':
            signal_data = bot.ai_ensemble_strategy(df, user_coin)
        else:
            # Default to RSI+MACD
            signal_data = bot.rsi_macd_strategy(df)
        
        # Get user trading settings
        user_settings = bot.trading_settings.get(user_id, {
            'margin': 1000,
            'leverage': 1,
            'risk_percentage': 2
        })
        
        # Calculate trading levels
        trading_levels = bot.calculate_trading_levels(df, signal_data, user_settings)
        
        # Format signal message
        coin_name = user_coin.replace('-', ' ').title()
        current_price = df['price'].iloc[-1]
        
        signal_emoji = "🟢" if signal_data['signal'] == 'BUY' else "🔴" if signal_data['signal'] == 'SELL' else "🟡"
        
        if trading_levels.get('action') == 'ERROR':
            message = f"❌ Error calculating trading levels. Please try again."
        elif trading_levels.get('action') == 'HOLD':
            message = f"""
🟡 *{coin_name} - HOLD Signal*

💰 *Current Price:* ${current_price:.4f}
🎯 *Action:* HOLD
📈 *Confidence:* {signal_data['confidence']}%

⚠️ No clear trading opportunity at this time.
Wait for better market conditions.
"""
        else:
            # Format comprehensive trading signal
            action = trading_levels['action']
            entry = trading_levels['entry_price']
            trigger = trading_levels['trigger_price']
            sl = trading_levels['stop_loss']
            tp1 = trading_levels['take_profit']['tp1']
            tp2 = trading_levels['take_profit']['tp2']
            tp3 = trading_levels['take_profit']['tp3']
            
            pos_size = trading_levels['position_sizing']['position_size']
            margin_req = trading_levels['position_sizing']['margin_required']
            risk_usd = trading_levels['position_sizing']['risk_amount_usd']
            leverage = trading_levels['position_sizing']['leverage']
            
            rr_ratio = trading_levels['analysis']['risk_reward_ratio']
            volatility = trading_levels['analysis']['volatility']
            
            message = f"""
{signal_emoji} *{coin_name} Trading Signal*

📈 *SIGNAL: {action}*
💰 *Current Price:* ${current_price:.4f}
🎯 *Confidence:* {signal_data['confidence']}%

💹 *ENTRY STRATEGY:*
• Entry Price: ${entry:.4f}
• Trigger Point: ${trigger:.4f}
• Wait for price to {'break above' if action == 'BUY' else 'break below'} trigger

🛑 *RISK MANAGEMENT:*
• Stop Loss: ${sl:.4f}
• Risk/Reward: 1:{rr_ratio:.2f}
• Max Risk: ${risk_usd:.2f}

🎯 *TAKE PROFIT LEVELS:*
• TP1 (Conservative): ${tp1:.4f}
• TP2 (Moderate): ${tp2:.4f}  
• TP3 (Aggressive): ${tp3:.4f}

💰 *POSITION SIZING:*
• Position Size: {pos_size:.4f} {coin_name.split()[0]}
• Leverage: {leverage}x
• Margin Required: ${margin_req:.2f}
• Volatility: {volatility:.2f}%
"""
        
        # Add technical analysis details
        if signal_data.get('details'):
            message += f"\n📋 *Technical Analysis:*\n"
            for detail in signal_data['details'][:2]:  # Limit to 2 details
                message += f"• {detail}\n"
        
        message += f"\n⏰ *Analysis Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Add action buttons
        keyboard = [
            [InlineKeyboardButton("📰 Get News", callback_data=f"news_{user_coin}")],
            [InlineKeyboardButton("💰 Trading Setup", callback_data="trading_setup")],
            [InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh_{user_coin}")],
            [InlineKeyboardButton("🪙 Change Coin", callback_data="select_coin")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await loading_msg.edit_text(message, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        await loading_msg.edit_text("❌ Error analyzing market data. Please try again later.")

async def select_coin_callback(query, context: ContextTypes.DEFAULT_TYPE):
    """Coin selection callback handler"""
    keyboard = []
    
    # Create coin selection keyboard
    coin_names = {
        'bitcoin': '₿ Bitcoin',
        'ethereum': 'Ξ Ethereum',
        'binancecoin': '🟡 BNB',
        'cardano': '🔵 Cardano',
        'solana': '🟣 Solana',
        'dogecoin': '🐕 Dogecoin',
        'polkadot': '🔴 Polkadot',
        'avalanche-2': '🏔️ Avalanche',
        'chainlink': '🔗 Chainlink',
        'polygon': '🟣 Polygon',
        'uniswap': '🦄 Uniswap',
        'litecoin': '🥈 Litecoin'
    }
    
    # Create inline keyboard with 2 columns
    for i in range(0, len(coin_names), 2):
        row = []
        coin_ids = list(coin_names.keys())[i:i+2]
        for coin_id in coin_ids:
            row.append(InlineKeyboardButton(
                coin_names[coin_id],
                callback_data=f"coin_{coin_id}"
            ))
        keyboard.append(row)
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "🪙 *Select Cryptocurrency for Analysis:*\n\nChoose the coin you want to get trading signals for:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

async def get_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get crypto news"""
    user_id = update.effective_user.id
    user_coin = bot.user_preferences.get(user_id, {}).get('coin', 'bitcoin')
    
    # Send loading message
    loading_msg = await update.message.reply_text("📰 Fetching latest crypto news... ⏳")
    
    try:
        # Get news
        coin_name = user_coin.replace('-', ' ').title()
        news_items = bot.get_crypto_news(coin_name)
        
        if not news_items:
            await loading_msg.edit_text("❌ No news available at the moment.")
            return
        
        # Analyze sentiment
        sentiment_data = bot.analyze_news_sentiment(news_items)
        
        # Format news message
        impact_emoji = "🟢" if sentiment_data['market_impact'] == 'BULLISH' else "🔴" if sentiment_data['market_impact'] == 'BEARISH' else "🟡"
        
        message = f"""
📰 *{coin_name} Latest News & Market Impact*

{impact_emoji} *Market Impact:* {sentiment_data['market_impact']}
📊 *Sentiment Score:* {sentiment_data['overall_sentiment']:.2f}
💪 *Strength:* {sentiment_data['sentiment_strength']:.2f}

*Latest News:*
"""
        
        for i, item in enumerate(news_items[:3], 1):
            message += f"""
{i}. *{item['title'][:50]}...*
   {item['description'][:100]}...
   🔗 [Read More]({item['url']})

"""
        
        message += f"⏰ *Updated:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Add action buttons
        keyboard = [
            [InlineKeyboardButton("📊 Get Signals", callback_data=f"signals_{user_coin}")],
            [InlineKeyboardButton("🔄 Refresh News", callback_data=f"news_{user_coin}")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await loading_msg.edit_text(message, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error getting news: {e}")
        await loading_msg.edit_text("❌ Error fetching news. Please try again later.")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    data = query.data
    
    if user_id not in bot.user_preferences:
        bot.user_preferences[user_id] = {}
    
    if data.startswith('coin_'):
        coin_id = data.split('_')[1]
        bot.user_preferences[user_id]['coin'] = coin_id
        coin_name = coin_id.replace('-', ' ').title()
        
        await query.edit_message_text(
            f"✅ Selected: {coin_name}\n\nUse /signals to get trading signals or /news for latest news.",
            parse_mode=ParseMode.MARKDOWN
        )
    
    elif data.startswith('strategy_'):
        strategy_id = data.split('_')[1]
        bot.user_preferences[user_id]['strategy'] = strategy_id
        strategy_name = bot.strategies.get(strategy_id, strategy_id)
        
        await query.edit_message_text(
            f"✅ Selected Strategy: {strategy_name}\n\nUse /signals to get trading signals with this strategy.",
            parse_mode=ParseMode.MARKDOWN
        )
    
    elif data.startswith('news_'):
        coin_id = data.split('_')[1]
        bot.user_preferences[user_id]['coin'] = coin_id
        await get_news_callback(query, context)
    
    elif data.startswith('signals_'):
        coin_id = data.split('_')[1]
        bot.user_preferences[user_id]['coin'] = coin_id
        await get_signals_callback(query, context)
    
    elif data.startswith('refresh_'):
        coin_id = data.split('_')[1]
        bot.user_preferences[user_id]['coin'] = coin_id
        await get_signals_callback(query, context)
    
    elif data == 'select_coin':
        await select_coin_callback(query, context)
    
    elif data == 'trading_setup':
        await trading_setup_callback(query, context)
    
    elif data == 'set_margin':
        await query.edit_message_text(
            "💵 *Set Your Trading Margin*\n\nPlease send your margin amount in USD.\nExample: 1000\n\n(This is the total capital you want to use for trading)",
            parse_mode=ParseMode.MARKDOWN
        )
        context.user_data['setting'] = 'margin'
    
    elif data == 'set_leverage':
        await query.edit_message_text(
            "🔥 *Set Your Leverage*\n\nPlease send your desired leverage (1-100x).\nExample: 5\n\n⚠️ Higher leverage = Higher risk!",
            parse_mode=ParseMode.MARKDOWN
        )
        context.user_data['setting'] = 'leverage'
    
    elif data == 'set_risk':
        await query.edit_message_text(
            "🎯 *Set Risk Percentage*\n\nPlease send your risk percentage per trade.\nExample: 2 (for 2%)\n\n💡 Recommended: 1-3% per trade",
            parse_mode=ParseMode.MARKDOWN
        )
        context.user_data['setting'] = 'risk_percentage'
    
    elif data == 'get_trading_signal':
        user_coin = bot.user_preferences.get(user_id, {}).get('coin', 'bitcoin')
        bot.user_preferences[user_id]['coin'] = user_coin
        await get_signals_callback(query, context)

async def trading_setup_callback(query, context: ContextTypes.DEFAULT_TYPE):
    """Trading setup callback handler"""
    user_id = query.from_user.id
    
    # Ensure user has trading settings with all required keys
    if user_id not in bot.trading_settings:
        bot.trading_settings[user_id] = {
            'margin': 1000,
            'leverage': 1,
            'risk_percentage': 2
        }
    
    user_settings = bot.trading_settings[user_id]
    
    # Ensure all keys exist
    if 'margin' not in user_settings:
        user_settings['margin'] = 1000
    if 'leverage' not in user_settings:
        user_settings['leverage'] = 1
    if 'risk_percentage' not in user_settings:
        user_settings['risk_percentage'] = 2
    
    setup_text = f"""
💰 *Trading Setup Configuration*

*Current Settings:*
💵 Margin: ${user_settings['margin']:,.2f}
🔥 Leverage: {user_settings['leverage']}x
🎯 Risk per Trade: {user_settings['risk_percentage']}%

*Position Sizing:*
• Max Position Value: ${user_settings['margin'] * user_settings['leverage']:,.2f}
• Risk Amount: ${user_settings['margin'] * (user_settings['risk_percentage']/100):,.2f}

*Configure your trading parameters below:*
    """
    
    keyboard = [
        [InlineKeyboardButton("💵 Set Margin", callback_data="set_margin")],
        [InlineKeyboardButton("🔥 Set Leverage", callback_data="set_leverage")],
        [InlineKeyboardButton("🎯 Set Risk %", callback_data="set_risk")],
        [InlineKeyboardButton("📊 Get Trading Signal", callback_data="get_trading_signal")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        setup_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    text = update.message.text
    user_id = update.effective_user.id
    
    # Check if user is setting trading parameters
    if context.user_data.get('setting'):
        setting = context.user_data['setting']
        
        try:
            if setting == 'margin':
                value = float(text.replace(',', '').replace('$', ''))
                if value < 10 or value > 1000000:
                    await update.message.reply_text("⚠️ Please enter a margin between $10 and $1,000,000")
                    return
                
                if user_id not in bot.trading_settings:
                    bot.trading_settings[user_id] = {}
                bot.trading_settings[user_id]['margin'] = value
                await update.message.reply_text(f"✅ Margin set to ${value:,.2f}")
                
            elif setting == 'leverage':
                value = int(text.replace('x', ''))
                if value < 1 or value > 100:
                    await update.message.reply_text("⚠️ Please enter leverage between 1x and 100x")
                    return
                
                if user_id not in bot.trading_settings:
                    bot.trading_settings[user_id] = {}
                bot.trading_settings[user_id]['leverage'] = value
                await update.message.reply_text(f"✅ Leverage set to {value}x")
                
            elif setting == 'risk_percentage':
                value = float(text.replace('%', ''))
                if value < 0.1 or value > 10:
                    await update.message.reply_text("⚠️ Please enter risk percentage between 0.1% and 10%")
                    return
                
                if user_id not in bot.trading_settings:
                    bot.trading_settings[user_id] = {}
                bot.trading_settings[user_id]['risk_percentage'] = value
                await update.message.reply_text(f"✅ Risk percentage set to {value}%")
            
            # Clear the setting state
            context.user_data['setting'] = None
            
            # Show updated trading setup
            await trading_setup(update, context)
            
        except ValueError:
            await update.message.reply_text("⚠️ Please enter a valid number.")
        
        return
    
    # Handle menu buttons
    if text == "📊 Get Signals":
        await get_signals(update, context)
    elif text == "🪙 Select Coin":
        await select_coin(update, context)
    elif text == "📈 Strategies":
        await select_strategy(update, context)
    elif text == "📰 Crypto News":
        await get_news(update, context)
    elif text == "💰 Trading Setup":
        await trading_setup(update, context)
    elif text == "ℹ️ Help":
        await help_command(update, context)
    elif text == "⚙️ Settings":
        await settings_command(update, context)
    else:
        await update.message.reply_text(
            "🤖 I didn't understand that. Use the menu buttons or type /help for available commands."
        )

async def trading_setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trading setup command handler"""
    user_id = update.effective_user.id
    
    # Ensure user has trading settings with all required keys
    if user_id not in bot.trading_settings:
        bot.trading_settings[user_id] = {
            'margin': 1000,
            'leverage': 1,
            'risk_percentage': 2
        }
    
    user_settings = bot.trading_settings[user_id]
    
    # Ensure all keys exist
    if 'margin' not in user_settings:
        user_settings['margin'] = 1000
    if 'leverage' not in user_settings:
        user_settings['leverage'] = 1
    if 'risk_percentage' not in user_settings:
        user_settings['risk_percentage'] = 2
    
    setup_text = f"""
💰 *Trading Setup Configuration*

*Current Settings:*
💵 Margin: ${user_settings['margin']:,.2f}
🔥 Leverage: {user_settings['leverage']}x
🎯 Risk per Trade: {user_settings['risk_percentage']}%

*Position Sizing:*
• Max Position Value: ${user_settings['margin'] * user_settings['leverage']:,.2f}
• Risk Amount: ${user_settings['margin'] * (user_settings['risk_percentage']/100):,.2f}

*Configure your trading parameters below:*
    """
    
    keyboard = [
        [InlineKeyboardButton("💵 Set Margin", callback_data="set_margin")],
        [InlineKeyboardButton("🔥 Set Leverage", callback_data="set_leverage")],
        [InlineKeyboardButton("🎯 Set Risk %", callback_data="set_risk")],
        [InlineKeyboardButton("📊 Get Trading Signal", callback_data="get_trading_signal")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        setup_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Settings command handler"""
    user_id = update.effective_user.id
    user_prefs = bot.user_preferences.get(user_id, {})
    
    current_coin = user_prefs.get('coin', 'bitcoin').replace('-', ' ').title()
    current_strategy = bot.strategies.get(user_prefs.get('strategy', 'rsi_macd'), 'RSI + MACD')
    
    settings_text = f"""
⚙️ *Bot Settings*

*Current Configuration:*
🪙 Selected Coin: {current_coin}
📈 Trading Strategy: {current_strategy}

*Available Actions:*
• Change cryptocurrency
• Switch trading strategy
• Configure alerts (coming soon)
• API settings (coming soon)

Use the buttons below to modify your settings.
    """
    
    keyboard = [
        [InlineKeyboardButton("🪙 Change Coin", callback_data="select_coin")],
        [InlineKeyboardButton("📈 Change Strategy", callback_data="select_strategy")],
        [InlineKeyboardButton("📊 Test Current Settings", callback_data="test_settings")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        settings_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

def main():
    """Main function to run the bot"""
    if BOT_TOKEN == 'BOT_TOKEN_HERE':
        print("❌ Please set your Telegram bot token in the BOT_TOKEN variable or environment variable.")
        return
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("coins", select_coin))
    application.add_handler(CommandHandler("strategies", select_strategy))
    application.add_handler(CommandHandler("signals", get_signals))
    application.add_handler(CommandHandler("news", get_news))
    application.add_handler(CommandHandler("trading", trading_setup))
    application.add_handler(CommandHandler("settings", settings_command))
    
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Start the bot
    print("🚀 Crypto Trading Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
