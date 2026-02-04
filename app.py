# ==========================================
# 老陳 AI 交易系統 V19.7 - 期貨代號對應版
# 1. 新增期貨代號支援：輸入 MHI 自動轉 2800.HK
# 2. 輸入 HHI (國指) 自動轉 2828.HK
# 3. 避免誤判為美股
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io

st.set_page_config(page_title="老陳回測系統 V19.7", layout="wide", page_icon="🇭🇰")

# --- 1. 數據獲取 (V19.7 智能對應) ---
@st.cache_data(ttl=3600)
def get_stooq_data(symbol):
    # 轉大寫 + 去空白
    raw_sym = symbol.upper().strip()
    clean_sym = raw_sym # 預設值
    
    # === 智能導向系統 ===
    
    # 1. 恆指系列 (HSI, MHI 小期) -> 轉盈富 (2800.HK)
    # Stooq 沒有連續期貨數據，用 2800 是最佳替代品，走勢同步
    if raw_sym in ["HSI", "^HSI", "MHI", "HK50"]: 
        clean_sym = "2800.HK"
        
    # 2. 國指系列 (HHI, MCH 小國期) -> 轉恆生中國企業 (2828.HK)
    elif raw_sym in ["HHI", "^HHI", "MCH"]:
        clean_sym = "2828.HK"
        
    # 3. 科技指數 (HSTECH, ATMX) -> 轉南方恆生科技 (3033.HK)
    elif raw_sym in ["HSTECH", "ATMX"]:
        clean_sym = "3033.HK"

    # 4. 港股 (純數字) -> 去前導零 + 加 .HK
    elif raw_sym.isdigit(): 
        clean_sym = f"{int(raw_sym)}.HK"
        
    # 5. 美股 (純字母) -> 加 .US
    # 必須排除上面的 MHI, HHI 等關鍵字，否則會變 MHI.US
    elif raw_sym.isalpha() and "." not in raw_sym:
        clean_sym = f"{raw_sym}.US"
        
    # 下載連結
    url = f"https://stooq.com/q/d/l/?s={clean_sym}&i=d"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None, clean_sym
            
        file_content = response.content.decode('utf-8')
        
        if "No data" in file_content or len(file_content) < 50:
             return None, clean_sym

        df = pd.read_csv(io.StringIO(file_content))
        
        # 數據清理
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        return df, clean_sym

    except Exception as e:
        print(f"Error: {e}")
        return None, clean_sym

# --- 2. 指標計算 ---
def calculate_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    df['MP'] = (df['High'] + df['Low']) / 2
    df['AO'] = df['MP'].rolling(5).mean() - df['MP'].rolling(34).mean()

    low_9 = df['Low'].rolling(9).min()
    high_9 = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

# --- 3. 訊號生成 ---
def generate_signals(df):
    df['Signal'] = 0 
    buy_cond = (df['J'] < 20) & (df['J'] > df['J'].shift(1))
    sell_cond = (df['J'] > 80) & (df['J'] < df['J'].shift(1))
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

# --- 4. 回測引擎 ---
def run_backtest(df, initial_capital, start_date):
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
    st.header("⚙️ 回測設定 (V19.7)")
    
    if st.button("🗑️ 清除快取"):
        st.cache_data.clear()
    
    st.divider()
    
    # 預設改為 MHI 讓你試試
    ticker = st.text_input("代號 (MHI, HHI, 700)", value="MHI").upper()
    start_date = st.date_input("開始日期", pd.to_datetime("2023-01-01"))
    initial_cash = st.number_input("初始本金 ($)", value=100000)
    run_btn = st.button("🚀 開始回測", type="primary")

st.title("🇭🇰 老陳回測系統 V19.7")
st.caption("✅ 支援期貨代號 (MHI -> 2800, HHI -> 2828)")

if run_btn:
    with st.spinner(f"正在分析 {ticker} (已自動對應至相關 ETF)..."):
        df_raw, real_sym = get_stooq_data(ticker)
        
        if df_raw is not None and not df_raw.empty:
            df = calculate_indicators(df_raw)
            df = generate_signals(df)
            final_val, ret, trade_log, df_chart = run_backtest(df, initial_cash, start_date)
            
            if not df_chart.empty:
                c1, c2, c3 = st.columns(3)
                
                # 顯示這是「替身」數據
                if ticker == "MHI":
                    c1.metric("回測標的", "MHI (以2800計算)")
                elif ticker == "HHI":
                    c1.metric("回測標的", "HHI (以2828計算)")
                else:
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
                
                st.subheader("📊 回測結果")
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])
                
                fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='K線'), row=1, col=1)
                
                buys = df_chart[df_chart['Signal'] == 1]
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='yellow'), name='買入'), row=1, col=1)
                sells = df_chart[df_chart['Signal'] == -1]
                fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='magenta'), name='賣出'), row=1, col=1)

                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Equity'], line=dict(color='#00ff00'), name='資產'), row=2, col=1)
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
            st.error(f"❌ 無法下載 {ticker}。")
