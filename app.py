# ==========================================
# 老陳 AI 交易系統 V24.1 - 顯示錯誤修復版
# 修復內容：
# 1. 解決 st.dataframe 格式化 None 值時的崩潰問題
# 2. 優化交易紀錄顯示，未平倉單會顯示 "-" 而非報錯
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
from scipy.stats import norm 
from datetime import datetime

st.set_page_config(page_title="老陳 V24.1 (修復版)", layout="wide", page_icon="🦋")

# --- 0. 全局設定 ---
PRESETS = {
    "自行輸入": "MHI",
    "🏙️ 收租三寶": {"823 領展": "823", "5 匯豐": "5", "941 中移動": "941"},
    "🚀 科技龍頭": {"700 騰訊": "700", "9988 阿里": "9988", "3690 美團": "3690"},
    "🇺🇸 美股 ETF": {"QQQ 納指": "QQQ", "SPY 標普": "SPY", "TLT 美債": "TLT", "NVDA": "NVDA"}
}

# --- 1. 核心函數 ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if S <= 0 or K <= 0 or T <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0.01)

@st.cache_data(ttl=3600)
def get_stooq_data(symbol):
    raw_sym = symbol.upper().strip()
    clean_sym = raw_sym 
    if raw_sym in ["HSI", "^HSI", "MHI", "HK50"]: clean_sym = "2800.HK"
    elif raw_sym in ["HHI", "^HHI", "MCH"]: clean_sym = "2828.HK"
    elif raw_sym.isdigit(): clean_sym = f"{int(raw_sym)}.HK"
    elif raw_sym.isalpha() and "." not in raw_sym: clean_sym = f"{raw_sym}.US"
        
    url = f"https://stooq.com/q/d/l/?s={clean_sym}&i=d"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return None, clean_sym
        file_content = response.content.decode('utf-8')
        if "No data" in file_content or len(file_content) < 50: return None, clean_sym
        df = pd.read_csv(io.StringIO(file_content))
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        return df, clean_sym
    except:
        return None, clean_sym

def calculate_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    low_9 = df['Low'].rolling(9).min()
    high_9 = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    pos_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    pos_mf = pd.Series(pos_flow).rolling(14).sum()
    neg_mf = pd.Series(neg_flow).rolling(14).sum()
    mfi_ratio = np.divide(pos_mf, neg_mf, out=np.zeros_like(pos_mf), where=neg_mf!=0)
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    return df

def generate_signals(df, buy_thresh, sell_thresh):
    df['Signal'] = 0 
    buy_cond = (df['J'] < buy_thresh) & (df['J'] > df['J'].shift(1))
    sell_cond = (df['J'] > sell_thresh) & (df['J'] < df['J'].shift(1))
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

# --- 2. 顯示模組 ---
def render_market_scan(df, real_sym):
    st.header(f"📊 報價與資金流: {real_sym}")
    last = df.iloc[-1]
    prev = df.iloc[-2]
    change = last['Close'] - prev['Close']
    pct = (change / prev['Close']) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新價", f"{last['Close']:,.2f}", f"{change:+.2f} ({pct:.2f}%)")
    c2.metric("MFI", f"{last['MFI']:.1f}")
    c3.metric("J 線", f"{last['J']:.1f}")
    c4.metric("量比", f"x{(last['Volume']/df['Volume'].rolling(20).mean().iloc[-1]):.1f}")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue'), name='MA60'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['J'], line=dict(color='magenta'), name='J線'), row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", row=3, col=1, line_color="green")
    fig.add_hline(y=80, line_dash="dot", row=3, col=1, line_color="red")
    fig.update_layout(height=800, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_strategy_lab(df, real_sym):
    st.header(f"🦅 策略工廠: {real_sym}")
    last = df.iloc[-1]
    st.info(f"價格: {last['Close']:.2f} | J線: {last['J']:.1f}")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("趨勢策略")
        if last['J'] < 20:
            st.success("🚀 看升 (Bullish)")
            st.markdown(f"**Bull Call Spread** (牛市價差)\n* Buy Call @ {last['Close']:.1f}\n* Sell Call @ {last['Close']*1.05:.1f}")
