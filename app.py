# ==========================================
# 老陳 AI 交易系統 V15.1 - 修正防呆版 (Fix IndexError)
# 修復重點：
# 1. 數據下載期改為 1年 (確保 MA60 有足夠數據)
# 2. 加入 df.empty 嚴格檢查，防止崩潰
# ==========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="老陳價量分析 V15.1", layout="wide", page_icon="💰")

# --- 1. 核心計算函數 (含防呆機制) ---
@st.cache_data(ttl=60)
def get_data_v15(symbol):
    try:
        # [修復1] 改為下載 '1y' (1年)，確保有足夠數據計算 MA60
        df = yf.download(symbol, period='1y', interval='1d', progress=False, auto_adjust=False)
        
        # 處理 yfinance 新版 MultiIndex 問題
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        
        # 確保數據是數值
        df = df.apply(pd.to_numeric, errors='coerce')

        # [修復2] 下載後立刻檢查是否為空
        if df.empty: return None

        # === A. 基礎指標 ===
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # === B. 成交量分析 ===
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']

        # === C. 資金流指標 (MFI) ===
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        pos_mf_sum = pd.Series(positive_flow).rolling(window=14).sum()
        neg_mf_sum = pd.Series(negative_flow).rolling(window=14).sum()
        
        mfi_ratio = pos_mf_sum / neg_mf_sum
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        df['MFI'].index = df.index

        # === D. KDJ (J線) ===
        low_list = df['Low'].rolling(9, min_periods=9).min()
        high_list = df['High'].rolling(9, min_periods=9).max()
        rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
        
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        # 刪除 NaN 數據
        df = df.dropna()
        
        # [修復3] dropna 之後再次檢查是否被刪光了
        if df.empty: return None

        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def analyze_volume_money(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 0
    signals = []
    
    # 1. 資金流 (MFI)
    if last['MFI'] > 80:
        score -= 2; signals.append("💰 MFI 資金超買 (>80) - 大戶可能出貨")
    elif last['MFI'] < 20:
        score += 2; signals.append("💰 MFI 資金超賣 (<20) - 地板價")
    
    if last['Close'] > prev['Close'] and last['MFI'] < prev['MFI'] and last['MFI'] > 60:
        score -= 1; signals.append("⚠️ 頂背馳 (價升資金流出)")

    # 2. 成交量
    if last['Vol_Ratio'] > 2.0:
        if last['Close'] > last['Open']:
            score += 1; signals.append("🔥 爆量長陽 (Vol > 2x) - 大資金進場")
        else:
            score -= 1; signals.append("💀 爆量長陰 (Vol > 2x) - 大資金逃亡")
    elif last['Vol_Ratio'] < 0.5:
        signals.append("💤 縮量觀望")

    # 3. 價量與趨勢
    if last['Close'] > prev['Close'] and last['Volume'] > prev['Volume']:
        score += 1; signals.append("📈 價量齊升")
    
    if last['J'] < 10 and last['J'] > prev['J']:
        score += 1; signals.append("⚡ J線低位勾頭")
    if last['Close'] > last['MA20']:
        score += 1

    return score, signals

# --- 2. 介面主程式 ---
st.title("💰 老陳 AI - 價量籌碼分析系統 (V15.1)")

col1, col2 = st.columns([3, 1])
with col1:
    symbol = st.text_input("股票代號 (如 0700.HK, NVDA)", value="^HSI").upper()
with col2:
    if st.button("刷新分析"): st.rerun()

# 獲取數據
df = get_data_v15(symbol)

# [修復4] 嚴格檢查：必須不是 None 且不是 Empty 且至少有 2 行數據(計算漲跌用)
if df is not None and not df.empty and len(df) >= 2:
    try:
        last = df.iloc[-1]
        change = last['Close'] - df.iloc[-2]['Close']
        
        # 數據卡片
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現價", f"{last['Close']:,.2f}", f"{change:+.2f}")
        c2.metric("成交量", f"{last['Volume']/1e6:.1f}M", f"x{last['Vol_Ratio']:.1f}倍")
        c3.metric("MFI 資金", f"{last['MFI']:.1f}")
        c4.metric("J線", f"{last['J']:.1f}")

        # 分析
        score, signals = analyze_volume_money(df)
        
        st.markdown("---")
        st.subheader("🤖 AI 診斷結果")
        if score >= 4:
            st.success("🚀 強力買入 (Strong Buy)")
        elif score <= -3:
            st.error("💥 強力賣出 (Strong Sell)")
        elif score > 0:
            st.info("👀 偏好 (Weak Buy)")
        else:
            st.warning("👀 偏淡 (Weak Sell)")
            
        with st.expander("查看訊號詳情"):
            for s in signals: st.write(s)

        # 繪圖
        st.subheader("📊 綜合操盤圖")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.4, 0.2, 0.2, 0.2],
                            subplot_titles=('價格 & MA', '成交量', '資金流向 (MFI)', 'KDJ (J線)'))

        # K線
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)

        # 成交量
        colors_vol = ['#00cc96' if c >= o else '#ef553b' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors_vol, name='Volume'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Vol_MA20'], line=dict(color='white', width=1, dash='dot'), name='Vol MA'), row=2, col=1)

        # MFI
        fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], line=dict(color='#00bfff', width=2), name='MFI'), row=3, col=1)
        fig.add_hline(y=80, line_dash="dot", row=3, col=1, line_color="red")
        fig.add_hline(y=20, line_dash="dot", row=3, col=1, line_color="green")

        # J線
        fig.add_trace(go.Scatter(x=df.index, y=df['J'], line=dict(color='#ab63fa', width=2), name='J線'), row=4, col=1)
        fig.add_hline(y=100, line_dash="dot", row=4, col=1, line_color="red")
        fig.add_hline(y=0, line_dash="dot", row=4, col=1, line_color="green")

        fig.update_layout(height=1000, xaxis_rangeslider_visible=False, showlegend=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"發生未知錯誤: {e}")
else:
    # 這裡就是顯示給你看的錯誤提示，而不是直接崩潰
    st.error(f"❌ 無法獲取 {symbol} 的數據，或數據不足。")
    st.info("💡 可能原因：\n1. 股票代號錯誤 (港股記得加 .HK，如 0700.HK)\n2. 該股票停牌或剛上市數據不足\n3. Yahoo 暫時連線不穩 (請稍後再試)")
