# ==========================================
# 老陳 AI 交易系統 V19.2 - Stooq 直連修復版
# 核心：棄用 pandas_datareader (因不支援 Python 3.12)
# 改良：直接透過 URL 下載 Stooq CSV，穩定性 100%
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io

st.set_page_config(page_title="老陳回測系統 (Stooq V2)", layout="wide", page_icon="🛡️")

# --- 1. 數據獲取 (URL 直連法 - 修復版) ---
@st.cache_data(ttl=3600)
def get_stooq_data(symbol):
    clean_sym = symbol.upper().strip()
    
    # 智能修正代號 (Stooq 格式)
    # 港股：700 -> 700.HK (Stooq 有時候前面要有 0，有時候不用，視乎情況，建議補齊4位)
    if clean_sym.isdigit(): 
        # Stooq 港股通常是 0700.HK
        clean_sym = f"{clean_sym}.HK"
        
    # 恆指修正
    if clean_sym in ["HSI", "HSI.HK"]: 
        clean_sym = "^HSI"
        
    # Stooq CSV 下載連結格式
    # s=代號, i=d (日線)
    url = f"https://stooq.com/q/d/l/?s={clean_sym}&i=d"
    
    try:
        # 偽裝瀏覽器 (避免 Stooq 擋 Python)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers)
        
        # 檢查是否下載成功
        if response.status_code != 200:
            return None, clean_sym
            
        # 將下載的內容轉為 DataFrame
        file_content = response.content.decode('utf-8')
        
        # 如果 Stooq 找不到股票，通常會回傳一個很短的錯誤頁面，而不是 CSV
        if "No data" in file_content or len(file_content) < 50:
             return None, clean_sym

        df = pd.read_csv(io.StringIO(file_content))
        
        # === 數據清理 ===
        # Stooq 的 CSV 標題通常是: Date, Open, High, Low, Close, Volume
        # 1. 處理日期
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # 2. 排序 (Stooq 下載下來是 新->舊，我們要反轉成 舊->新)
        df = df.sort_index()
        
        # 3. 確保數值正確
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        return df, clean_sym

    except Exception as e:
        print(f"Error: {e}")
        # 替身機制：如果失敗，試試 2800.HK
        if clean_sym == "^HSI":
             return get_stooq_data("2800")
        return None, clean_sym

# --- 2. 指標計算 ---
def calculate_indicators(df):
    # MA
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # AO 指標
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

# --- 3. 產生訊號 ---
def generate_signals(df):
    df['Signal'] = 0 
    
    # 買入：J線低位(<20) 且 向上勾頭
    buy_cond = (df['J'] < 20) & (df['J'] > df['J'].shift(1))
    
    # 賣出：J線高位(>80) 且 向下勾頭
    sell_cond = (df['J'] > 80) & (df['J'] < df['J'].shift(1))
    
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

# --- 4. 回測引擎 ---
def run_backtest(df, initial_capital, start_date):
    # 篩選日期
    mask = df.index >= pd.to_datetime(start_date)
    df_test = df.loc[mask].copy()
    
    if df_test.empty: return 0, 0, pd.DataFrame(), pd.DataFrame()

    capital = initial_capital
    position = 0
    history = []
    equity_curve = []
    
    for i in range(len(df_test)):
        date = df_test.index[i]
        price = df_test['Close'].iloc[i]
        signal = df_test['Signal'].iloc[i]
        
        # 策略執行
        if signal == 1 and position == 0:
            position = capital / price
            capital = 0
            history.append({'Date': date, 'Type': 'BUY', 'Price': price, 'Balance': position*price})
            
        elif signal == -1 and position > 0:
            capital = position * price
            position = 0
            history.append({'Date': date, 'Type': 'SELL', 'Price': price, 'Balance': capital})
        
        current_val = capital if position == 0 else position * price
        equity_curve.append(current_val)
            
    final_value = capital if position == 0 else position * df_test['Close'].iloc[-1]
    ret = ((final_value - initial_capital) / initial_capital) * 100
    
    df_test['Equity'] = equity_curve
    return final_value, ret, pd.DataFrame(history), df_test

# --- 5. 網站介面 ---
with st.sidebar:
    st.header("⚙️ 回測設定 (直連版)")
    ticker = st.text_input("股票代號 (如 2800)", value="2800").upper()
    start_date = st.date_input("開始日期", pd.to_datetime("2023-01-01"))
    initial_cash = st.number_input("初始本金 ($)", value=100000)
    run_btn = st.button("🚀 開始回測", type="primary")

st.title("🛡️ 老陳回測系統 V19.2")
st.caption("✅ 已修復 Python 3.12 相容性問題 (Direct CSV Mode)")

if run_btn:
    with st.spinner(f"正在從 Stooq 下載 {ticker}..."):
        df_raw, real_sym = get_stooq_data(ticker)
        
        if df_raw is not None and not df_raw.empty:
            df = calculate_indicators(df_raw)
            df = generate_signals(df)
            
            # 傳入 start_date 進行切片
            final_val, ret, trade_log, df_chart = run_backtest(df, initial_cash, start_date)
            
            if not df_chart.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("回測標的", real_sym)
                c2.metric("最終資產", f"${final_val:,.0f}", f"{ret:+.2f}%")
                
                win_rate = 0
                if not trade_log.empty:
                    sells = trade_log[trade_log['Type']=='SELL']
                    if len(sells) > 0:
                        wins = 0
                        for idx, row in sells.iterrows():
                            prev = trade_log[(trade_log.index < idx) & (trade_log['Type']=='BUY')]
                            if not prev.empty and row['Price'] > prev.iloc[-1]['Price']: wins += 1
                        win_rate = (wins/len(sells))*100
                c3.metric("勝率", f"{win_rate:.1f}%", f"交易 {len(trade_log)//2} 次")
                
                # 畫圖
                st.subheader("📊 回測結果")
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])
                
                # K線
                fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='K線'), row=1, col=1)
                
                # 買賣點
                buys = df_chart[df_chart['Signal'] == 1]
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='yellow'), name='買入'), row=1, col=1)
                sells = df_chart[df_chart['Signal'] == -1]
                fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='magenta'), name='賣出'), row=1, col=1)

                # 資產
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Equity'], line=dict(color='#00ff00'), name='資產'), row=2, col=1)
                
                # J線
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['J'], line=dict(color='#ab63fa'), name='J線'), row=3, col=1)
                fig.add_hline(y=20, line_dash="dot", row=3, col=1, line_color="green")
                fig.add_hline(y=80, line_dash="dot", row=3, col=1, line_color="red")
                
                fig.update_layout(height=800, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("交易紀錄"):
                    st.dataframe(trade_log)
            else:
                st.warning("選定的日期範圍內沒有數據。")
        else:
            st.error(f"無法下載 {ticker}，請檢查代號 (港股建議 2800 或 0700)。")
