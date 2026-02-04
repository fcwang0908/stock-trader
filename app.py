# ==========================================
# 老陳 AI 交易系統 V15.5 - 終極離線/上傳版
# 解決方案：當 Yahoo 封鎖 IP 時，允許用戶「手動上傳 CSV」進行分析
# ==========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="老陳 V15.5 (終極版)", layout="wide", page_icon="📂")

# --- 0. 智能代號修正 ---
def smart_symbol(symbol):
    s = symbol.upper().strip()
    if s.isdigit(): return f"{s.zfill(4)}.HK"
    if s in ["HSI", "HSI.HK"]: return "^HSI"
    return s

# --- 1. 核心數據處理 (支援 CSV 上傳) ---
def process_data(df):
    # 確保索引是日期格式
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            # 嘗試把 'Date' 欄位變成索引 (針對 CSV)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            else:
                # 嘗試把 index 轉日期
                df.index = pd.to_datetime(df.index)
        except:
            return None

    # 移除時區
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # 確保數值正確
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # === 指標計算 ===
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = np.where(df['Vol_MA20']>0, df['Volume']/df['Vol_MA20'], 0)

    # MFI
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    pos_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    pos_mf = pd.Series(pos_flow).rolling(14).sum()
    neg_mf = pd.Series(neg_flow).rolling(14).sum()
    mfi_ratio = np.divide(pos_mf, neg_mf, out=np.zeros_like(pos_mf), where=neg_mf!=0)
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    df['MFI'].index = df.index

    # KDJ
    low_9 = df['Low'].rolling(9).min()
    high_9 = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    return df.dropna()

# 獲取數據入口
def get_data_v15(symbol, uploaded_file):
    # 優先處理上傳的檔案
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return process_data(df), "📄 上傳的檔案"
        except Exception as e:
            st.error(f"檔案讀取失敗: {e}")
            return None, symbol

    # 其次嘗試 Yahoo 下載
    clean_sym = smart_symbol(symbol)
    try:
        # 嘗試多種格式 (暴力測試)
        variants = [clean_sym]
        if ".HK" in clean_sym: variants.append(clean_sym.replace(".HK", "")) # 試試 700
        
        for sym in variants:
            ticker = yf.Ticker(sym)
            df = ticker.history(period='1y', interval='1d')
            if not df.empty:
                return process_data(df), sym
        
        return None, clean_sym
    except:
        return None, clean_sym

def analyze_logic(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0; signals = []
    
    if last['MFI'] > 80: score -= 2; signals.append("💰 MFI 超買 (>80)")
    elif last['MFI'] < 20: score += 2; signals.append("💰 MFI 超賣 (<20)")
    
    if last['Vol_Ratio'] > 2.0:
        if last['Close'] > last['Open']: score += 1; signals.append("🔥 爆量長陽")
        else: score -= 1; signals.append("💀 爆量長陰")
        
    if last['J'] < 10 and last['J'] > prev['J']: score += 1; signals.append("⚡ J線低位勾頭")
    if last['Close'] > last['MA20']: score += 1
    
    return score, signals

# --- 2. 介面 ---
st.title("💰 老陳 AI - V15.5 (終極數據版)")

# 側邊欄：手動上傳
with st.sidebar:
    st.header("📂 數據備用通道")
    st.info("如果 Yahoo 封鎖連線，請在此上傳 CSV。")
    uploaded_file = st.file_uploader("上傳歷史數據 (CSV)", type=['csv'])
    st.markdown("[👉 按此去 Yahoo 下載 CSV](https://finance.yahoo.com/quote/0700.HK/history)")

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("股票代號 (700, TSLA)", value="700").upper()
with col2:
    if st.button("刷新"): st.rerun()

# 獲取數據
df, real_sym = get_data_v15(user_input, uploaded_file)

if df is not None and not df.empty and len(df) >= 2:
    try:
        last = df.iloc[-1]
        change = last['Close'] - df.iloc[-2]['Close']
        
        st.success(f"✅ 成功獲取數據源: {real_sym}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", f"{last['Close']:,.2f}", f"{change:+.2f}")
        c2.metric("成交量", f"{last['Volume']/1e6:.1f}M", f"x{last['Vol_Ratio']:.1f}倍")
        c3.metric("MFI", f"{last['MFI']:.1f}")
        c4.metric("J線", f"{last['J']:.1f}")

        score, signals = analyze_logic(df)
        
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
        st.error(f"分析錯誤: {e}")
else:
    st.error(f"❌ 依然無法自動下載 {user_input}。Yahoo 封鎖了雲端 IP。")
    st.info("💡 **終極解決辦法：**")
    st.markdown("1. 點擊這裡下載數據：[Yahoo Finance 0700.HK](https://finance.yahoo.com/quote/0700.HK/history)")
    st.markdown("2. 點擊 Yahoo 頁面中間的 **'Download'** 下載 `.csv` 檔案。")
    st.markdown("3. 打開左側選單 ( > )，把檔案拖進 **'上傳歷史數據'** 框框。")
    st.markdown("4. 你的 AI 分析圖就會立刻出現！")
