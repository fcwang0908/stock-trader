# ==========================================
# 老陳 AI 交易系統 V15.2 - 智能修正版
# 修復：自動修正股票代號格式 (如自動把 700 改為 0700.HK)
# ==========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="老陳價量分析 V15.2", layout="wide", page_icon="💰")

# --- 0. 智能代號修正函數 (新增) ---
def smart_symbol(symbol):
    s = symbol.upper().strip()
    
    # 1. 修正恆指
    if s == "HSI" or s == "HSI.HK" or s == "^HSI.HK":
        return "^HSI"
    
    # 2. 修正港股 (輸入 700 -> 0700.HK)
    if s.isdigit(): 
        # 如果是純數字 (如 700, 5, 2800)
        return f"{s.zfill(4)}.HK" # 補足4位並加 .HK
    
    # 3. 修正美股 (如 tsla -> TSLA)
    # 不做額外處理，直接回傳
    return s

# --- 1. 核心計算函數 ---
@st.cache_data(ttl=60)
def get_data_v15(symbol):
    try:
        # 使用智能修正後的代號
        clean_sym = smart_symbol(symbol)
        
        # 下載 1年 數據
        df = yf.download(clean_sym, period='1y', interval='1d', progress=False, auto_adjust=False)
        
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        
        df = df.apply(pd.to_numeric, errors='coerce')

        if df.empty: return None, clean_sym

        # === 指標計算 ===
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # 成交量
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']

        # MFI 資金流
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        pos_mf_sum = pd.Series(positive_flow).rolling(window=14).sum()
        neg_mf_sum = pd.Series(negative_flow).rolling(window=14).sum()
        mfi_ratio = pos_mf_sum / neg_mf_sum
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
    
    # 資金流
    if last['MFI'] > 80: score -= 2; signals.append("💰 MFI 資金超買 (>80)")
    elif last['MFI'] < 20: score += 2; signals.append("💰 MFI 資金超賣 (<20)")
    if last['Close'] > prev['Close'] and last['MFI'] < prev['MFI'] and last['MFI'] > 60: score -= 1; signals.append("⚠️ 頂背馳")

    # 成交量
    if last['Vol_Ratio'] > 2.0:
        if last['Close'] > last['Open']: score += 1; signals.append("🔥 爆量長陽")
        else: score -= 1; signals.append("💀 爆量長陰")
    
    # 趨勢
    if last['Close'] > prev['Close'] and last['Volume'] > prev['Volume']: score += 1; signals.append("📈 價量齊升")
    if last['J'] < 10 and last['J'] > prev['J']: score += 1; signals.append("⚡ J線低位勾頭")
    if last['Close'] > last['MA20']: score += 1

    return score, signals

# --- 2. 介面 ---
st.title("💰 老陳 AI - 智能分析系統 (V15.2)")

col1, col2 = st.columns([3, 1])
with col1:
    # 預設值改為簡單的 HSI
    user_input = st.text_input("股票代號 (支援模糊輸入: 700, HSI, TSLA)", value="HSI").upper()
with col2:
    if st.button("刷新"): st.rerun()

# 獲取數據 (回傳 df 和 修正後的代號)
df, real_symbol = get_data_v15(user_input)

if df is not None and not df.empty and len(df) >= 2:
    try:
        last = df.iloc[-1]
        change = last['Close'] - df.iloc[-2]['Close']
        
        # 顯示修正後的代號
        st.caption(f"已自動修正代號為: {real_symbol}")
        
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
    # 更加詳細的錯誤提示
    st.error(f"❌ 無法獲取 {user_input} (修正後: {real_symbol}) 的數據。")
    st.info("💡 解決辦法：\n1. 如果你想查恆指，請直接輸入 'HSI' (程式會自動幫你加 ^)\n2. 如果你想查騰訊，請輸入 '700' (程式會自動幫你加 0 和 .HK)\n3. 確保你不是輸入了中文或其他符號")
