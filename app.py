# ==========================================
# 老陳 AI 交易系統 V19.1 - Stooq 回測專用版
# 核心：使用 Stooq 數據源進行歷史策略回測
# 策略：AO + J線 (低買高賣)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_datareader.data as web
from datetime import datetime

st.set_page_config(page_title="老陳回測系統 (Stooq)", layout="wide", page_icon="🧪")

# --- 1. 數據獲取 (Stooq 引擎) ---
@st.cache_data(ttl=3600)
def get_stooq_data(symbol, start_date):
    clean_sym = symbol.upper().strip()
    
    # 智能修正代號 (配合 Stooq 格式)
    if clean_sym.isdigit(): 
        clean_sym = f"{clean_sym}.HK"
    if clean_sym in ["HSI", "HSI.HK"]: 
        clean_sym = "^HSI"
        
    try:
        # 下載數據
        df = web.DataReader(clean_sym, 'stooq', start=start_date)
        
        # ⚠️ 關鍵：Stooq 預設是 [新 -> 舊]，回測必須要 [舊 -> 新]
        df = df.sort_index()
        
        # 轉為數值
        df = df.apply(pd.to_numeric, errors='coerce')
        
        return df, clean_sym
    except Exception as e:
        # 替身機制：恆指失敗轉盈富
        if clean_sym == "^HSI":
             return get_stooq_data("2800", start_date)
        return None, clean_sym

# --- 2. 指標計算 ---
def calculate_indicators(df):
    # MA
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean() # 牛熊線
    
    # AO 指標 (Awesome Oscillator)
    # MP = (High + Low) / 2
    df['MP'] = (df['High'] + df['Low']) / 2
    df['AO'] = df['MP'].rolling(5).mean() - df['MP'].rolling(34).mean()

    # KDJ 指標
    low_9 = df['Low'].rolling(9).min()
    high_9 = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df

# --- 3. 產生訊號 (策略大腦) ---
def generate_signals(df):
    df['Signal'] = 0 # 0=無動作, 1=買入, -1=賣出
    
    # 買入條件：J線低位(<20) 且 向上勾頭 (J > 昨日J)
    buy_cond = (df['J'] < 20) & (df['J'] > df['J'].shift(1))
    
    # 賣出條件：J線高位(>80) 且 向下勾頭 (J < 昨日J)
    sell_cond = (df['J'] > 80) & (df['J'] < df['J'].shift(1))
    
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    
    return df

# --- 4. 回測引擎 (計算盈虧) ---
def run_backtest(df, initial_capital):
    capital = initial_capital
    position = 0 # 持股數 (0=空倉)
    history = []
    equity_curve = [] # 資產曲線
    
    for i in range(1, len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i]
        
        # 執行買入 (有訊號 且 空倉時)
        if signal == 1 and position == 0:
            position = capital / price # 全倉買入
            capital = 0
            history.append({'Date': date, 'Type': 'BUY', 'Price': price, 'Balance': position*price})
            
        # 執行賣出 (有訊號 且 持倉時)
        elif signal == -1 and position > 0:
            capital = position * price # 全倉賣出
            position = 0
            history.append({'Date': date, 'Type': 'SELL', 'Price': price, 'Balance': capital})
        
        # 每日結算資產價值
        current_val = capital if position == 0 else position * price
        equity_curve.append(current_val)
            
    # 最後一天強制平倉
    final_value = capital if position == 0 else position * df['Close'].iloc[-1]
    ret = ((final_value - initial_capital) / initial_capital) * 100
    
    # 補齊 equity curve 長度以便畫圖
    df_chart = df.iloc[1:].copy()
    df_chart['Equity'] = equity_curve
    
    return final_value, ret, pd.DataFrame(history), df_chart

# --- 5. 網站介面 ---
with st.sidebar:
    st.header("⚙️ 回測設定 (Stooq)")
    ticker = st.text_input("股票代號", value="2800").upper()
    start_date = st.date_input("開始日期", pd.to_datetime("2023-01-01"))
    initial_cash = st.number_input("初始本金 ($)", value=100000)
    st.info("策略：J線 < 20 買入，J線 > 80 賣出")
    run_btn = st.button("🚀 開始回測", type="primary")

st.title("🧪 老陳回測系統 V19.1")
st.caption("數據源：Stooq (穩定不封鎖) | 策略：AO + J線反轉")

if run_btn:
    with st.spinner(f"正在從波蘭 Stooq 下載 {ticker} 數據..."):
        df_raw, real_sym = get_stooq_data(ticker, start_date)
        
        if df_raw is not None and not df_raw.empty:
            # 1. 計算
            df = calculate_indicators(df_raw)
            df = generate_signals(df)
            final_val, ret, trade_log, df_chart = run_backtest(df, initial_cash)
            
            # 2. 顯示績效
            c1, c2, c3 = st.columns(3)
            c1.metric("回測標的", real_sym)
            
            color = "normal"
            if ret > 0: color = "normal" # 綠色/正常
            else: color = "inverse" # 紅色 (Streamlit logic)
            
            c2.metric("最終資產", f"${final_val:,.0f}", f"{ret:+.2f}%")
            
            # 計算勝率
            win_rate = 0
            if not trade_log.empty:
                sells = trade_log[trade_log['Type']=='SELL']
                wins = 0
                for idx, row in sells.iterrows():
                    # 找對應的買入價
                    # 這裡簡化邏輯，假設賣出一定對應最近一次買入
                    # 實際應更嚴謹，但展示足夠了
                    prev_buy = trade_log[(trade_log.index < idx) & (trade_log['Type']=='BUY')]
                    if not prev_buy.empty:
                         if row['Price'] > prev_buy.iloc[-1]['Price']: wins += 1
                if len(sells) > 0:
                    win_rate = (wins / len(sells)) * 100
            
            c3.metric("交易勝率", f"{win_rate:.1f}%", f"共 {len(trade_log)//2} 次交易")
            
            # 3. 畫圖 (三層)
            st.subheader("📊 回測結果可視化")
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                                row_heights=[0.5, 0.25, 0.25],
                                subplot_titles=('價格 & 買賣點', '資產增長曲線 (Equity Curve)', 'KDJ 指標'))
            
            # 圖1: K線 + 買賣點
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='牛熊線'), row=1, col=1)
            
            # 標記買點
            buys = df[df['Signal'] == 1]
            fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='yellow'), name='買入'), row=1, col=1)
            # 標記賣點
            sells = df[df['Signal'] == -1]
            fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='magenta'), name='賣出'), row=1, col=1)

            # 圖2: 資產曲線
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Equity'], fill='tozeroy', line=dict(color='#00ff00'), name='總資產'), row=2, col=1)
            
            # 圖3: J線
            fig.add_trace(go.Scatter(x=df.index, y=df['J'], line=dict(color='#ab63fa', width=2), name='J線'), row=3, col=1)
            fig.add_hline(y=20, line_dash="dot", row=3, col=1, line_color="green")
            fig.add_hline(y=80, line_dash="dot", row=3, col=1, line_color="red")
            
            fig.update_layout(height=800, template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # 4. 交易明細
            with st.expander("查看詳細交易紀錄"):
                if not trade_log.empty:
                    st.dataframe(trade_log.style.format({"Price": "{:.2f}", "Balance": "{:.2f}"}))
                else:
                    st.write("這段時間內沒有觸發任何交易。")

        else:
            st.error(f"找不到 {ticker} 的數據。")
            st.info("💡 提示：港股請輸入數字 (如 700)，美股輸入代號 (如 TSLA)。")

else:
    st.info("👈 請在左側輸入代號，例如 2800，然後按開始。")
