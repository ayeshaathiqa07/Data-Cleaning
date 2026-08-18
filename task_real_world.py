import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Set visual style
sns.set_theme(style="whitegrid")

# 1. Fetch Real-World Financial Data (Apple Inc. - AAPL)
print("=== FETCHING REAL-WORLD FINANCIAL DATA (AAPL) ===")
ticker = "AAPL"
df = yf.download(ticker, start="2023-01-01", end="2024-01-01")

# Clean column names if multi-indexed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. Data Cleaning & Feature Engineering
df.reset_index(inplace=True)
df['SMA_20'] = df['Close'].rolling(window=20).mean() # 20-Day Simple Moving Average
df['SMA_50'] = df['Close'].rolling(window=50).mean() # 50-Day Simple Moving Average

# 3. Statistical Overview
print("\n--- Summary Statistics ---")
print(df[['Close', 'Volume', 'SMA_20', 'SMA_50']].describe())

# 4. Visualizations
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: Stock Price & Moving Averages
axes[0].plot(df['Date'], df['Close'], label='Close Price', color='blue', alpha=0.6)
axes[0].plot(df['Date'], df['SMA_20'], label='20-Day SMA', color='orange', linestyle='--')
axes[0].plot(df['Date'], df['SMA_50'], label='50-Day SMA', color='red', linestyle='--')
axes[0].set_title(f'{ticker} Stock Price Analysis & Moving Averages (2023)')
axes[0].set_ylabel('Price (USD)')
axes[0].legend(loc='upper left')

# Plot 2: Trading Volume
axes[1].bar(df['Date'], df['Volume'], color='purple', alpha=0.5)
axes[1].set_title(f'{ticker} Daily Trading Volume')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Volume')

plt.tight_layout()
plt.show()

print("\n=== KEY INSIGHTS ===")
print("1. Successfully performed end-to-end stock trend analysis using real market data.")
print("2. The 20-day and 50-day Simple Moving Averages highlight key bullish/bearish crossover points.")