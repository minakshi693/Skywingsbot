#!/usr/bin/env python3
"""
Test script for the new trading features
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from skywingsbot import CryptoTradingBot
import pandas as pd

def test_trading_features():
    """Test the new trading features"""
    print("🧪 Testing Advanced Trading Features...")
    
    # Initialize bot
    bot = CryptoTradingBot()
    
    # Test 1: Get market data
    print("\n1️⃣ Testing market data...")
    df = bot.get_coin_data('bitcoin', days=30)
    if df is not None:
        print(f"✅ Market data: {len(df)} days")
        current_price = df['price'].iloc[-1]
        print(f"   Current price: ${current_price:.2f}")
    
    # Test 2: Calculate technical indicators
    print("\n2️⃣ Testing technical indicators...")
    df = bot.calculate_technical_indicators(df)
    print(f"✅ Technical indicators calculated")
    print(f"   RSI: {df['rsi'].iloc[-1]:.2f}")
    print(f"   MACD: {df['macd'].iloc[-1]:.6f}")
    
    # Test 3: Get trading signal
    print("\n3️⃣ Testing trading strategy...")
    signal_data = bot.rsi_macd_strategy(df)
    print(f"✅ Signal: {signal_data['signal']}")
    print(f"   Confidence: {signal_data['confidence']}%")
    
    # Test 4: Trading settings
    print("\n4️⃣ Testing trading calculations...")
    user_settings = {
        'margin': 5000,      # $5,000 capital
        'leverage': 3,       # 3x leverage
        'risk_percentage': 2 # 2% risk per trade
    }
    
    trading_levels = bot.calculate_trading_levels(df, signal_data, user_settings)
    
    if trading_levels.get('action') != 'ERROR':
        print(f"✅ Trading levels calculated")
        print(f"   Signal: {trading_levels['action']}")
        print(f"   Entry Price: ${trading_levels['entry_price']:.2f}")
        print(f"   Stop Loss: ${trading_levels['stop_loss']:.2f}")
        print(f"   Take Profit 1: ${trading_levels['take_profit']['tp1']:.2f}")
        print(f"   Position Size: {trading_levels['position_sizing']['position_size']:.4f} BTC")
        print(f"   Margin Required: ${trading_levels['position_sizing']['margin_required']:.2f}")
        print(f"   Risk Amount: ${trading_levels['position_sizing']['risk_amount_usd']:.2f}")
        print(f"   Risk/Reward Ratio: 1:{trading_levels['analysis']['risk_reward_ratio']:.2f}")
    else:
        print("❌ Error calculating trading levels")
    
    print("\n🎉 Trading features test complete!")
    
    # Example trading scenarios
    print("\n📋 Example Trading Scenarios:")
    
    scenarios = [
        {'margin': 1000, 'leverage': 1, 'risk_percentage': 1},
        {'margin': 5000, 'leverage': 5, 'risk_percentage': 2},
        {'margin': 10000, 'leverage': 10, 'risk_percentage': 3}
    ]
    
    for i, settings in enumerate(scenarios, 1):
        trading_calc = bot.calculate_trading_levels(df, signal_data, settings)
        if trading_calc.get('action') != 'ERROR':
            print(f"\n{i}. Margin: ${settings['margin']}, Leverage: {settings['leverage']}x, Risk: {settings['risk_percentage']}%")
            print(f"   Position Size: {trading_calc['position_sizing']['position_size']:.4f} BTC")
            print(f"   Max Risk: ${trading_calc['position_sizing']['risk_amount_usd']:.2f}")
            print(f"   R/R Ratio: 1:{trading_calc['analysis']['risk_reward_ratio']:.2f}")

if __name__ == "__main__":
    test_trading_features()
