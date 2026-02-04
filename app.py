# ==========================================
# 老陳 AI 交易系統 V15.4 - 單兵突破版
# 修改：棄用 yf.download，改用 yf.Ticker().history() 避開封鎖
# ==========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="老陳 V15.4 (TSLA修復)", layout="wide", page_icon="💰")

# --- 0. 智能代號修正 ---
def smart_symbol(symbol):
    s = symbol.upper().strip()
    # 港股修正
    if s.isdigit(): return f"{s.zfill(4)}.HK"
    # 恆指修正
    if s in ["HSI", "HSI.HK"]: return "^HSI"
    # 美股修正 (TSLA -> TSLA)
    return s

# --- 1. 核心計算函數 (使用 Ticker.history) ---
@st.cache_data(ttl=60)
def get_data_v15(symbol):
    clean_sym = smart_symbol(symbol)
    
    try:
        # 🔥 重大修改：改用 Ticker 物件
        ticker = yf.Ticker(clean_sym)
        
        # 使用 history 抓取，這通常比 download 更難被封鎖
        df = ticker.history(period='1y', interval='1d')
        
        # 替身機制：如果恆指失敗，試試盈富基金
        if df.empty and clean_sym == "^HSI":
            clean_sym = "2800.HK"
            ticker = yf.Ticker(clean_sym)
            df = ticker.history(period='1y', interval='1d')

        # 再次檢查
        if df.empty: return None, clean_sym

        # === 數據清理 (History 格式略有不同) ===
        # history 出來的 index 已經是 datetime，且通常沒有 MultiIndex 問題
        # 移除時區資訊 (避免畫圖報錯)
        df.index = df.index.tz_localize(None)

        # === 指標計算 ===
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        # 防除以零
        df['Vol_Ratio'] = np.where(df['Vol_MA20'] > 0, df['Volume'] / df['Vol_MA20'], 0)

        # MFI
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        pos_mf_sum = pd.Series(positive_flow).rolling(window=14).sum()
        neg_mf_sum = pd.Series(negative_flow).rolling(window=14).sum()
        # 防除以零
        mfi_ratio = np.divide(pos_mf_sum, neg_mf_sum, out=np.zeros_like(pos_mf_sum), where=neg_mf_sum!=0)
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        df['MFI'].index = df.index

        # KDJ
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

def analyze_volume_money(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0; signals = []
    
    if last['MFI'] > 80: score -= 2; signals.append("💰 MFI 資金超買 (>80)")
    elif last['MFI'] < 20: score += 2; signals.append("💰 MFI 資金超賣 (<20)")
    
    if last['Vol_Ratio'] > 2.0:
        if last['Close'] > last['Open']: score += 1; signals.append("🔥 爆量長陽")
        else: score -= 1; signals.append("💀 爆量長陰")
        
    if last['J'] < 10 and last['J'] > prev['J']: score += 1; signals.append("⚡ J線低位勾頭")
    if last['Close'] > last['MA20']: score += 1

    return score, signals

# --- 2. 介面 ---
st.title("💰 老陳 AI - V15.4 (TSLA 修復版)")

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("股票代號 (TSLA, NVDA, 700)", value="TSLA").upper()
with col2:
    if st.button("刷新"): st.rerun()

df, real_symbol = get_data_v15(user_input)

if df is not None and not df.empty and len(df) >= 2:
    try:
        last = df.iloc[-1]
        change = last['Close'] - df.iloc[-2]['Close']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", f"{last['Close']:,.2f}", f"{change:+.2f}")
        c2.metric("成交量", f"{last['Volume']/1e6:.1f}M", f"x{last['Vol_Ratio']:.1f}倍")
        c3.metric("MFI 資金", f"{last['MFI']:.1f}")
        c4.metric("J線", f"{last['J']:.1f}")

        score, signals = analyze_volume_money(df)
        
        st.markdown("---")
        if score >= 4: st.success("🚀 強力買入")
        elif score <= -3: st.error("💥 強力賣出")
        elif score > 0: st.info("👀 偏好")
        else: st.warning("👀 偏淡")
            
        with st.expander("訊號詳情"):
            for s in signals: st.write(s)

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
        st.error(f"發生錯誤: {e}")
else:
    st.error(f"❌ 依然無法獲取 {user_input} 的數據。")
    st.info("💡 最後一招：Yahoo 正在封鎖雲端 IP。請嘗試在「你自己的電腦」上運行此程式 (Localhost)，保證 100% 成功。")
