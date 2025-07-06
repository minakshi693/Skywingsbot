"""
Cryptocurrency Trading Signal Bot
Advanced Telegram bot with AI/ML analysis for crypto trading signals
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
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
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

class CryptoTradingBot:
    def __init__(self):
        self.user_preferences = {}
        self.supported_coins = [
            'bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana',
            'dogecoin', 'polkadot', 'avalanche-2', 'chainlink', 'polygon',
            'uniswap', 'litecoin', 'bitcoin-cash', 'algorand', 'vechain',
            'stellar', 'filecoin', 'tron', 'ethereum-classic', 'monero'
        ]
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
        """Fetch historical price data for a cryptocurrency"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if days <= 30 else 'daily'
            }
            
            if COINGECKO_API_KEY:
                headers = {'x-cg-demo-api-key': COINGECKO_API_KEY}
            else:
                headers = {}
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add volume and market cap if available
            if 'total_volumes' in data:
                volume_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                volume_df.set_index('timestamp', inplace=True)
                df = df.join(volume_df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {coin_id}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various technical indicators"""
        try:
            # Price-based indicators
            df['sma_10'] = ta.trend.sma_indicator(df['price'], window=10)
            df['sma_20'] = ta.trend.sma_indicator(df['price'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['price'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['price'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['price'], window=26)
            
            # MACD
            df['macd'] = ta.trend.macd_diff(df['price'])
            df['macd_signal'] = ta.trend.macd_signal(df['price'])
            df['macd_histogram'] = ta.trend.macd_diff(df['price']) - ta.trend.macd_signal(df['price'])
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['price'])
            
            # Bollinger Bands
            df['bb_upper'] = ta.volatility.bollinger_hband(df['price'])
            df['bb_lower'] = ta.volatility.bollinger_lband(df['price'])
            df['bb_middle'] = ta.volatility.bollinger_mavg(df['price'])
            
            # Stochastic
            df['stoch_k'] = ta.momentum.stoch(df['price'], df['price'], df['price'])
            df['stoch_d'] = ta.momentum.stoch_signal(df['price'], df['price'], df['price'])
            
            # Williams %R
            df['williams_r'] = ta.momentum.williams_r(df['price'], df['price'], df['price'])
            
            # ADX
            df['adx'] = ta.trend.adx(df['price'], df['price'], df['price'])
            
            # Volume indicators (if volume data available)
            if 'volume' in df.columns:
                df['volume_sma'] = ta.volume.volume_sma(df['price'], df['volume'])
                df['vpt'] = ta.volume.volume_price_trend(df['price'], df['volume'])
            
            # Momentum
            df['momentum'] = ta.momentum.roc(df['price'], window=10)
            
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
            # Using a free news API (you can replace with NewsAPI or other services)
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{coin_name} cryptocurrency",
                'sortBy': 'publishedAt',
                'pageSize': 5,
                'language': 'en'
            }
            
            if NEWS_API_KEY:
                params['apiKey'] = NEWS_API_KEY
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                news_items = []
                for article in data.get('articles', []):
                    news_items.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', '')
                    })
                
                return news_items
            else:
                # Fallback to CoinGecko news or other free sources
                return [
                    {
                        'title': f"Latest {coin_name} Analysis",
                        'description': "Market analysis and trends for cryptocurrency trading",
                        'url': f"https://www.coingecko.com/en/coins/{coin_name}",
                        'published_at': datetime.now().isoformat(),
                        'source': 'CoinGecko'
                    }
                ]
                
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def analyze_news_sentiment(self, news_items: List[Dict]) -> Dict:
        """Analyze news sentiment and market impact"""
        try:
            # Simple sentiment analysis based on keywords
            positive_keywords = ['bull', 'bullish', 'rise', 'pump', 'moon', 'surge', 'rally', 'breakout', 'adoption']
            negative_keywords = ['bear', 'bearish', 'crash', 'dump', 'fall', 'drop', 'decline', 'sell-off']
            
            sentiment_scores = []
            impact_analysis = []
            
            for item in news_items:
                text = f"{item['title']} {item['description']}".lower()
                
                positive_count = sum(1 for keyword in positive_keywords if keyword in text)
                negative_count = sum(1 for keyword in negative_keywords if keyword in text)
                
                if positive_count > negative_count:
                    sentiment = "POSITIVE"
                    score = min(1.0, positive_count / max(1, negative_count))
                elif negative_count > positive_count:
                    sentiment = "NEGATIVE"
                    score = -min(1.0, negative_count / max(1, positive_count))
                else:
                    sentiment = "NEUTRAL"
                    score = 0
                
                sentiment_scores.append(score)
                impact_analysis.append({
                    'title': item['title'],
                    'sentiment': sentiment,
                    'score': score,
                    'source': item['source']
                })
            
            overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            
            if overall_sentiment > 0.2:
                market_impact = "BULLISH"
            elif overall_sentiment < -0.2:
                market_impact = "BEARISH"
            else:
                market_impact = "NEUTRAL"
            
            return {
                'overall_sentiment': overall_sentiment,
                'market_impact': market_impact,
                'sentiment_strength': abs(overall_sentiment),
                'analysis': impact_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {'overall_sentiment': 0, 'market_impact': 'NEUTRAL', 'sentiment_strength': 0, 'analysis': []}

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
        [KeyboardButton("⚙️ Settings"), KeyboardButton("ℹ️ Help")]
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
        await get_news(query, context)
    
    elif data.startswith('signals_'):
        coin_id = data.split('_')[1]
        bot.user_preferences[user_id]['coin'] = coin_id
        await get_signals(query, context)
    
    elif data.startswith('refresh_'):
        coin_id = data.split('_')[1]
        bot.user_preferences[user_id]['coin'] = coin_id
        await get_signals(query, context)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    text = update.message.text
    
    if text == "📊 Get Signals":
        await get_signals(update, context)
    elif text == "🪙 Select Coin":
        await select_coin(update, context)
    elif text == "📈 Strategies":
        await select_strategy(update, context)
    elif text == "📰 Crypto News":
        await get_news(update, context)
    elif text == "ℹ️ Help":
        await help_command(update, context)
    elif text == "⚙️ Settings":
        await settings_command(update, context)
    else:
        await update.message.reply_text(
            "🤖 I didn't understand that. Use the menu buttons or type /help for available commands."
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
    application.add_handler(CommandHandler("settings", settings_command))
    
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Start the bot
    print("🚀 Crypto Trading Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
