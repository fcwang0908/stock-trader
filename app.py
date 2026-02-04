# ==========================================
# 老陳 AI 交易系統 V16.0 - 穩健圖表版
# 核心：使用 V7 的下載邏輯 (穩定) + V15 的視覺分析 (專業)
# 適用：日線級別趨勢分析 (Daily Trend Analysis)
# ==========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="老陳 V16 穩健版", layout="wide", page_icon="📈")

# --- 0. 智能代號修正 ---
def smart_symbol(symbol):
    s = symbol.upper().strip()
    if s.isdigit(): return f"{s.zfill(4)}.HK"
    if s in ["HSI", "HSI.HK"]: return "^HSI"
    return s

# --- 1. 核心數據下載 (回歸最簡單的 V7 邏輯) ---
@st.cache_data(ttl=3600) # 緩存 1小時，減少請求頻率，大幅降低被封鎖機率
def get_data_v16(symbol):
    clean_sym = smart_symbol(symbol)
    
    try:
        # 下載 1 年的日線數據 (最穩定的 API 請求方式)
        df = yf.download(clean_sym, period='1y', interval='1d', progress=False, auto_adjust=False)
        
        # 處理 MultiIndex (yfinance 新版特性)
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
            
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # 替身機制：如果恆指沒數據，試試盈富基金
        if df.empty and clean_sym == "^HSI":
            clean_sym = "2800.HK"
            df = yf.download(clean_sym, period='1y', interval='1d', progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        if df.empty: return None, clean_sym

        # === 指標計算 (保留 V15 的強大功能) ===
        # 1. 均線
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean() # 牛熊線
        
        # 2. 成交量
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_Ratio'] = np.where(df['Vol_MA20'] > 0, df['Volume'] / df['Vol_MA20'], 0)

        # 3. MFI 資金流
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        pos_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        neg_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        pos_mf = pd.Series(pos_flow).rolling(14).sum()
        neg_mf = pd.Series(neg_flow).rolling(14).sum()
        mfi_ratio = np.divide(pos_mf, neg_mf, out=np.zeros_like(pos_mf), where=neg_mf!=0)
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))

        # 4. KDJ (J線)
        low_9 = df['Low'].rolling(9).min()
        high_9 = df['High'].rolling(9).max()
        rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
        df['K'] = rsv.ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        return df.dropna(), clean_sym
    except Exception as e:
        return None, symbol

def analyze_logic(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0; signals = []
    
    # 資金流
    if last['MFI'] > 80: score -= 2; signals.append("💰 MFI 資金超買 (>80)")
    elif last['MFI'] < 20: score += 2; signals.append("💰 MFI 資金超賣 (<20)")
    
    # 爆量
    if last['Vol_Ratio'] > 2.0:
        if last['Close'] > last['Open']: score += 1; signals.append("🔥 爆量大陽燭")
        else: score -= 1; signals.append("💀 爆量大陰燭")
        
    # J線
    if last['J'] < 10 and last['J'] > prev['J']: score += 1; signals.append("⚡ J線觸底勾頭")
    
    # 趨勢
    if last['Close'] > last['MA60']: score += 1; signals.append("🐂 股價在牛熊線(MA60)之上")
    else: score -= 1; signals.append("🐻 股價在牛熊線(MA60)之下")
    
    return score, signals

# --- 2. 介面 ---
st.title("📈 老陳 V16 穩健圖表版")
st.caption("✅ 使用日線數據 (Daily Data) 以確保連線穩定性")

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("輸入股票代號 (如 700, 2800, TSLA)", value="700").upper()
with col2:
    if st.button("分析"): st.rerun()

df, real_sym = get_data_v16(user_input)

if df is not None and not df.empty and len(df) >= 2:
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        change = last['Close'] - prev['Close']
        pct = (change / prev['Close']) * 100
        
        # 顯示代號與日期
        st.markdown(f"### {real_sym} | 數據日期: {last.name.date()}")
        
        # 數據卡
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("收市價", f"{last['Close']:,.2f}", f"{change:+.2f} ({pct:.2f}%)")
        c2.metric("成交量倍數", f"x{last['Vol_Ratio']:.1f}", delta_color="off")
        c3.metric("MFI 資金流", f"{last['MFI']:.1f}")
        c4.metric("J 線數值", f"{last['J']:.1f}")

        # AI 分析
        score, signals = analyze_logic(df)
        st.markdown("---")
        if score >= 3: st.success("🚀 綜合評分：強勢 (Strong)")
        elif score <= -3: st.error("💥 綜合評分：弱勢 (Weak)")
        else: st.info("⚖️ 綜合評分：震盪 / 觀望")
        
        with st.expander("查看訊號邏輯"):
            for s in signals: st.write(s)

        # 專業四層圖表
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.4, 0.2, 0.2, 0.2], 
                            subplot_titles=('價格 & MA60(藍)', '成交量', 'MFI 資金流', 'KDJ (紫線=J)'))
        
        # 1. Price
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1.5), name='MA60'), row=1, col=1)
        
        # 2. Volume
        colors_vol = ['#00cc96' if c >= o else '#ef553b' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors_vol, name='Vol'), row=2, col=1)
        
        # 3. MFI
        fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], line=dict(color='#00bfff', width=2), name='MFI'), row=3, col=1)
        fig.add_hline(y=80, line_dash="dot", row=3, col=1, line_color="red")
        fig.add_hline(y=20, line_dash="dot", row=3, col=1, line_color="green")
        
        # 4. KDJ
        fig.add_trace(go.Scatter(x=df.index, y=df['J'], line=dict(color='#ab63fa', width=2), name='J線'), row=4, col=1)
        fig.add_hline(y=100, line_dash="dot", row=4, col=1, line_color="red")
        fig.add_hline(y=0, line_dash="dot", row=4, col=1, line_color="green")
        
        fig.update_layout(height=900, xaxis_rangeslider_visible=False, showlegend=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"圖表繪製錯誤: {e}")
else:
    st.error(f"找不到 {user_input} 的數據。")
    st.info("提示：此版本使用日線數據，請確認代號正確。如恆指請用 HSI，騰訊請用 700。")
