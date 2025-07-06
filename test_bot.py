#!/usr/bin/env python3
"""
Simple test script to verify crypto trading bot functionality
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from skywingsbot import CryptoTradingBot
import pandas as pd

async def test_bot_functions():
    """Test core bot functions"""
    print("🧪 Testing Crypto Trading Bot Functions...")
    
    # Initialize bot
    bot = CryptoTradingBot()
    
    # Test 1: Market data fetching
    print("\n1️⃣ Testing market data fetching...")
    try:
        df = bot.get_coin_data('bitcoin', days=10)
        if df is not None and not df.empty:
            print(f"✅ Market data: {len(df)} data points fetched")
            print(f"   Latest price: ${df['price'].iloc[-1]:.2f}")
        else:
            print("❌ No market data received")
    except Exception as e:
        print(f"❌ Market data error: {e}")
    
    # Test 2: Technical indicators
    print("\n2️⃣ Testing technical indicators...")
    try:
        if df is not None:
            df = bot.calculate_technical_indicators(df)
            indicators = ['rsi', 'macd', 'sma_20', 'bb_upper', 'bb_lower']
            available = [ind for ind in indicators if ind in df.columns]
            print(f"✅ Technical indicators: {len(available)} calculated")
            print(f"   Available: {', '.join(available)}")
        else:
            print("❌ Cannot test indicators without market data")
    except Exception as e:
        print(f"❌ Technical indicators error: {e}")
    
    # Test 3: Trading strategy
    print("\n3️⃣ Testing trading strategy...")
    try:
        if df is not None and len(df) > 0:
            signal = bot.rsi_macd_strategy(df)
            print(f"✅ Strategy signal: {signal['signal']}")
            print(f"   Confidence: {signal['confidence']}%")
            print(f"   Strategy: {signal['strategy']}")
        else:
            print("❌ Cannot test strategy without data")
    except Exception as e:
        print(f"❌ Strategy error: {e}")
    
    # Test 4: News fetching
    print("\n4️⃣ Testing news fetching...")
    try:
        news = bot.get_crypto_news('Bitcoin')
        if news:
            print(f"✅ News: {len(news)} articles fetched")
            print(f"   First headline: {news[0]['title'][:50]}...")
        else:
            print("❌ No news fetched")
    except Exception as e:
        print(f"❌ News error: {e}")
    
    # Test 5: Sentiment analysis
    print("\n5️⃣ Testing sentiment analysis...")
    try:
        if news:
            sentiment = bot.analyze_news_sentiment(news)
            print(f"✅ Sentiment analysis complete")
            print(f"   Market impact: {sentiment['market_impact']}")
            print(f"   Sentiment score: {sentiment['overall_sentiment']}")
            print(f"   Strength: {sentiment['sentiment_strength']}")
        else:
            print("❌ Cannot test sentiment without news")
    except Exception as e:
        print(f"❌ Sentiment error: {e}")
    
    print("\n🎉 Bot functionality test complete!")

if __name__ == "__main__":
    asyncio.run(test_bot_functions())
