# ==========================================
# 老陳 AI 交易系統 V19.8 - 策略大師版
# 1. 新增：每筆交易盈虧 (P&L) 計算
# 2. 新增：側邊欄參數調整 (MA週期, J線買賣門檻)
# 3. 優化：圖表標記更清晰
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io

st.set_page_config(page_title="老陳回測 V19.8 (大師版)", layout="wide", page_icon="🎛️")

# --- 1. 數據獲取 ---
@st.cache_data(ttl=3600)
def get_stooq_data(symbol):
    raw_sym = symbol.upper().strip()
    clean_sym = raw_sym 
    
    # 代號對應表
    if raw_sym in ["HSI", "^HSI", "MHI", "HK50"]: clean_sym = "2800.HK"
    elif raw_sym in ["HHI", "^HHI", "MCH"]: clean_sym = "2828.HK"
    elif raw_sym in ["HSTECH", "ATMX"]: clean_sym = "3033.HK"
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

# --- 2. 指標計算 (支援自訂參數) ---
def calculate_indicators(df, ma_fast, ma_slow):
    # 使用用戶設定的參數
    df['MA_Fast'] = df['Close'].rolling(window=ma_fast).mean()
    df['MA_Slow'] = df['Close'].rolling(window=ma_slow).mean()
    
    # KDJ (固定參數 9,3,3)
    low_9 = df['Low'].rolling(9).min()
    high_9 = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

# --- 3. 訊號生成 (支援自訂參數) ---
def generate_signals(df, buy_threshold, sell_threshold):
    df['Signal'] = 0 
    # 買入：J < 買入線 (例如20) 且 勾頭向上
    buy_cond = (df['J'] < buy_threshold) & (df['J'] > df['J'].shift(1))
    
    # 賣出：J > 賣出線 (例如80) 且 勾頭向下
    sell_cond = (df['J'] > sell_threshold) & (df['J'] < df['J'].shift(1))
    
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

# --- 4. 回測引擎 (計算每筆盈虧) ---
def run_backtest(df, initial_capital, start_date):
    mask = df.index >= pd.to_datetime(start_date)
    df_test = df.loc[mask].copy()
    
    if df_test.empty: return 0, 0, pd.DataFrame(), pd.DataFrame()

    capital = initial_capital
    position = 0
    
    # 交易紀錄表
    trade_log = []
    
    equity_curve = []
    
    # 記錄進場資訊
    entry_price = 0
    entry_date = None
    
    for i in range(len(df_test)):
        date = df_test.index[i]
        price = df_test['Close'].iloc[i]
        signal = df_test['Signal'].iloc[i]
        
        # 買入邏輯
        if signal == 1 and position == 0:
            position = capital / price
            capital = 0
            entry_price = price
            entry_date = date
            # 記錄動作
            trade_log.append({
                '進場日期': date, '動作': '買入', '價格': price, 
                '出場日期': None, '盈虧 ($)': None, '報酬率 (%)': None
            })
            
        # 賣出邏輯
        elif signal == -1 and position > 0:
            capital = position * price
            
            # 計算這筆交易賺多少
            profit = (price - entry_price) * position
            pct_return = ((price - entry_price) / entry_price) * 100
            
            # 更新上一筆買入紀錄，補上出場資訊
            if trade_log:
                trade_log[-1]['出場日期'] = date
                trade_log[-1]['動作'] = '已平倉' # 狀態更新
                trade_log[-1]['出場價格'] = price
                trade_log[-1]['盈虧 ($)'] = profit
                trade_log[-1]['報酬率 (%)'] = pct_return
            
            position = 0
            entry_price = 0
        
        # 計算每日資產
        current_val = capital if position == 0 else position * price
        equity_curve.append(current_val)
            
    final_value = capital if position == 0 else position * df_test['Close'].iloc[-1]
    ret = ((final_value - initial_capital) / initial_capital) * 100
    df_test['Equity'] = equity_curve
    
    # 轉成 DataFrame
    df_log = pd.DataFrame(trade_log)
    return final_value, ret, df_log, df_test

# --- 5. 網站介面 ---

# === 側邊欄：參數控制室 ===
with st.sidebar:
    st.header("🎛️ 參數控制室")
    
    if st.button("🗑️ 清除快取"): st.cache_data.clear()
    
    st.subheader("1. 基本設定")
    ticker = st.text_input("股票代號 (MHI, 700)", value="MHI").upper()
    start_date = st.date_input("開始回測", pd.to_datetime("2023-01-01"))
    initial_cash = st.number_input("本金 ($)", value=100000)

    st.subheader("2. 策略參數 (J線)")
    # 滑桿讓用戶調整
    buy_thresh = st.slider("買入門檻 (J < ?)", 0, 40, 20, help="數值越小越保守，交易次數越少")
    sell_thresh = st.slider("賣出門檻 (J > ?)", 60, 100, 80, help="數值越大越貪心，希望能吃到盡頭")
    
    st.subheader("3. 均線設定 (僅參考)")
    ma_fast_p = st.number_input("快線週期", value=20)
    ma_slow_p = st.number_input("慢線週期", value=60)
    
    run_btn = st.button("🚀 執行回測", type="primary")

st.title("🇭🇰 老陳 V19.8 - 策略大師版")

if run_btn:
    with st.spinner(f"正在分析 {ticker}..."):
        df_raw, real_sym = get_stooq_data(ticker)
        
        if df_raw is not None and not df_raw.empty:
            # 傳入用戶設定的參數
            df = calculate_indicators(df_raw, ma_fast_p, ma_slow_p)
            df = generate_signals(df, buy_thresh, sell_thresh)
            final_val, ret, df_log, df_chart = run_backtest(df, initial_cash, start_date)
            
            if not df_chart.empty:
                # 1. 績效總覽
                c1, c2, c3 = st.columns(3)
                if ticker in ["MHI", "HHI"]:
                    c1.metric("回測標的", f"{ticker} (代理數據)")
                else:
                    c1.metric("回測標的", real_sym)
                    
                color = "normal" if ret > 0 else "inverse"
                c2.metric("最終資產", f"${final_val:,.0f}", f"{ret:+.2f}%")
                
                # 計算勝率 (基於 df_log)
                win_rate = 0
                total_trades = 0
                if not df_log.empty:
                    closed_trades = df_log.dropna(subset=['盈虧 ($)']) # 只算已平倉
                    total_trades = len(closed_trades)
                    if total_trades > 0:
                        wins = len(closed_trades[closed_trades['盈虧 ($)'] > 0])
                        win_rate = (wins / total_trades) * 100
                
                c3.metric("勝率", f"{win_rate:.1f}%", f"共 {total_trades} 筆完整交易")
                
                # 2. 圖表分析
                st.subheader("📊 買賣點與資產走勢")
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
                                    subplot_titles=('K線 & 買賣點', '資產曲線', 'J線訊號區'))
                
                # K線
                fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='K線'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA_Slow'], line=dict(color='blue', width=1), name=f'MA{ma_slow_p}'), row=1, col=1)
                
                # 買賣標記
                buys = df_chart[df_chart['Signal'] == 1]
                sells = df_chart[df_chart['Signal'] == -1]
                fig.add_trace(go.Scatter(x=buys.index, y=buys['Low']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=15, color='yellow'), name='買入'), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells.index, y=sells['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=15, color='magenta'), name='賣出'), row=1, col=1)

                # 資產
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Equity'], fill='tozeroy', line=dict(color='#00ff00'), name='總資產'), row=2, col=1)
                
                # J線
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['J'], line=dict(color='#ab63fa'), name='J線'), row=3, col=1)
                fig.add_hline(y=buy_thresh, line_dash="dot", row=3, col=1, line_color="green", annotation_text="買入區")
                fig.add_hline(y=sell_thresh, line_dash="dot", row=3, col=1, line_color="red", annotation_text="賣出區")
                
                fig.update_layout(height=900, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. 詳細交易表 (重點更新)
                st.subheader("📝 詳細交易紀錄 (每筆賺蝕)")
                if not df_log.empty:
                    # 美化表格顯示
                    display_log = df_log.copy()
                    # 格式化日期
                    display_log['進場日期'] = display_log['進場日期'].dt.date
                    display_log['出場日期'] = pd.to_datetime(display_log['出場日期']).dt.date
                    
                    # 顏色標示
                    def highlight_profit(val):
                        if pd.isna(val): return ''
                        color = '#90ee90' if val > 0 else '#ffcccb' # 淺綠 / 淺紅
                        return f'background-color: {color}; color: black'

                    st.dataframe(
                        display_log.style.format({
                            "價格": "{:.2f}", 
                            "出場價格": "{:.2f}", 
                            "盈虧 ($)": "{:+.2f}", 
                            "報酬率 (%)": "{:+.2f}%"
                        }).map(highlight_profit, subset=['盈虧 ($)', '報酬率 (%)']),
                        use_container_width=True
                    )
                else:
                    st.info("這段期間沒有觸發任何交易。試試調整參數？")
            else:
                st.warning("無數據")
        else:
            st.error("下載失敗")
