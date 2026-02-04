# ==========================================
# 老陳 AI 交易系統 V15.3 - 防封鎖替身版
# 修復：當 Yahoo 封鎖 ^HSI 時，自動切換至 2800.HK (盈富基金)
# ==========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests # 新增：用於偽裝瀏覽器

st.set_page_config(page_title="老陳價量分析 V15.3", layout="wide", page_icon="💰")

# --- 0. 智能代號修正 ---
def smart_symbol(symbol):
    s = symbol.upper().strip()
    if s == "HSI" or s == "HSI.HK": return "^HSI"
    if s.isdigit(): return f"{s.zfill(4)}.HK"
    return s

# --- 1. 核心計算函數 (含替身機制) ---
@st.cache_data(ttl=60)
def get_data_v15(symbol):
    
    # 定義一個偽裝瀏覽器的 Header (騙過 Yahoo)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    })

    clean_sym = smart_symbol(symbol)
    
    try:
        # 第一次嘗試下載
        df = download_wrapper(clean_sym, session)
        
        # === 替身機制 (Failover) ===
        # 如果下載恆指失敗，自動嘗試 2800.HK
        if (df is None or df.empty) and clean_sym == "^HSI":
            st.toast("⚠️ Yahoo 封鎖了恆指數據，正在切換至盈富基金 (2800.HK)...", icon="🔄")
            clean_sym = "2800.HK" # 切換代號
            df = download_wrapper(clean_sym, session) # 再試一次

        if df is None or df.empty: return None, clean_sym

        # === 指標計算 (保持不變) ===
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        pos_mf_sum = pd.Series(positive_flow).rolling(window=14).sum()
        neg_mf_sum = pd.Series(negative_flow).rolling(window=14).sum()
        mfi_ratio = pos_mf_sum / neg_mf_sum
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        df['MFI'].index = df.index

        low_list = df['Low'].rolling(9, min_periods=9).min()
        high_list = df['High'].rolling(9, min_periods=9).max()
        rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        df = df.dropna()
        if df.empty: return None, clean_sym

        return df, clean_sym
    except Exception as e:
        print(f"Error: {e}")
        return None, symbol

# 輔助下載函數
def download_wrapper(sym, session):
    try:
        # 加入 session 參數來偽裝
        df = yf.download(sym, period='1y', interval='1d', progress=False, auto_adjust=False, session=session)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.apply(pd.to_numeric, errors='coerce')
        if df.empty: return None
        return df
    except:
        return None

def analyze_volume_money(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0; signals = []
    
    # 分析邏輯
    if last['MFI'] > 80: score -= 2; signals.append("💰 MFI 資金超買 (>80)")
    elif last['MFI'] < 20: score += 2; signals.append("💰 MFI 資金超賣 (<20)")
    if last['Close'] > prev['Close'] and last['MFI'] < prev['MFI'] and last['MFI'] > 60: score -= 1; signals.append("⚠️ 頂背馳")

    if last['Vol_Ratio'] > 2.0:
        if last['Close'] > last['Open']: score += 1; signals.append("🔥 爆量長陽")
        else: score -= 1; signals.append("💀 爆量長陰")
    
    if last['Close'] > prev['Close'] and last['Volume'] > prev['Volume']: score += 1; signals.append("📈 價量齊升")
    if last['J'] < 10 and last['J'] > prev['J']: score += 1; signals.append("⚡ J線低位勾頭")
    if last['Close'] > last['MA20']: score += 1

    return score, signals

# --- 2. 介面 ---
st.title("💰 老陳 AI - 智能防封鎖版 (V15.3)")

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("股票代號 (輸入 HSI 自動偵測)", value="HSI").upper()
with col2:
    if st.button("刷新"): st.rerun()

df, real_symbol = get_data_v15(user_input)

if df is not None and not df.empty and len(df) >= 2:
    try:
        last = df.iloc[-1]
        change = last['Close'] - df.iloc[-2]['Close']
        
        # 顯示當前使用的代號
        if real_symbol == "2800.HK" and "^HSI" in smart_symbol(user_input):
            st.warning("⚠️ 由於 Yahoo 數據源不穩，系統已自動切換至 **2800.HK (盈富基金)** 進行分析，走勢與恆指同步。")
        else:
            st.caption(f"當前分析代號: {real_symbol}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", f"{last['Close']:,.2f}", f"{change:+.2f}")
        c2.metric("成交量", f"{last['Volume']/1e6:.1f}M", f"x{last['Vol_Ratio']:.1f}倍")
        c3.metric("MFI 資金", f"{last['MFI']:.1f}")
        c4.metric("J線", f"{last['J']:.1f}")

        score, signals = analyze_volume_money(df)
        
        st.markdown("---")
        if score >= 4: st.success("🚀 強力買入 (Strong Buy)")
        elif score <= -3: st.error("💥 強力賣出 (Strong Sell)")
        elif score > 0: st.info("👀 偏好 (Weak Buy)")
        else: st.warning("👀 偏淡 (Weak Sell)")
            
        with st.expander("詳細訊號"):
            for s in signals: st.write(s)

        # 繪圖
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.2, 0.2, 0.2], subplot_titles=('價格', '成交量', 'MFI', 'J線'))
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
        colors_vol = ['#00cc96' if c >= o else '#ef553b' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors_vol, name='Vol'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], line=dict(color='#00bfff', width=2), name='MFI'), row=3, col=1)
        fig.add_hline(y=80, line_dash="dot", row=3, col=1, line_color="red"); fig.add_hline(y=20, line_dash="dot", row=3, col=1, line_color="green")
        fig.add_trace(go.Scatter(x=df.index, y=df['J'], line=dict(color='#ab63fa', width=2), name='J線'), row=4, col=1)
        fig.add_hline(y=100, line_dash="dot", row=4, col=1, line_color="red"); fig.add_hline(y=0, line_dash="dot", row=4, col=1, line_color="green")
        fig.update_layout(height=1000, xaxis_rangeslider_visible=False, showlegend=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"發生未知錯誤: {e}")
else:
    st.error(f"❌ 無法獲取 {user_input} 的數據。")
    st.info("💡 建議：Yahoo 可能暫時封鎖了該代號，請嘗試輸入 '2800' 或 '700'。")
