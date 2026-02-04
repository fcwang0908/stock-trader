# ==========================================
# 老陳 AI 交易系統 V20.0 - 正股/期權雙核回測版
# 核心升級：
# 1. 引入 Black-Scholes 模型模擬期權價格
# 2. 支援「正股買賣」與「期權 (Call/Put)」兩種模式
# 3. 雙向交易：訊號 1 做多 (Call)，訊號 -1 做空 (Put)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
from scipy.stats import norm # 用來算期權價格

st.set_page_config(page_title="老陳 V20.0 (期權版)", layout="wide", page_icon="⚖️")

# --- 1. Black-Scholes 期權定價模型 ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    S: 股價, K: 行使價, T: 年化剩餘時間 (天數/365), r: 利率, sigma: IV, type: call/put
    """
    # 避免數學錯誤 (股價不能為0)
    if S <= 0 or K <= 0 or T <= 0: return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else: # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0.01) # 最小價值 0.01

# --- 2. 數據獲取 (Stooq) ---
@st.cache_data(ttl=3600)
def get_stooq_data(symbol):
    raw_sym = symbol.upper().strip()
    clean_sym = raw_sym 
    
    # 智能代號對應
    if raw_sym in ["HSI", "^HSI", "MHI", "HK50"]: clean_sym = "2800.HK"
    elif raw_sym in ["HHI", "^HHI", "MCH"]: clean_sym = "2828.HK"
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

# --- 3. 指標計算 ---
def calculate_indicators(df):
    # MA
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # KDJ
    low_9 = df['Low'].rolling(9).min()
    high_9 = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

# --- 4. 訊號生成 ---
def generate_signals(df, buy_thresh, sell_thresh):
    df['Signal'] = 0 
    buy_cond = (df['J'] < buy_thresh) & (df['J'] > df['J'].shift(1))
    sell_cond = (df['J'] > sell_thresh) & (df['J'] < df['J'].shift(1))
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

# --- 5. 回測引擎 (雙模式) ---
def run_backtest(df, initial_capital, start_date, mode, iv_param=0.3):
    mask = df.index >= pd.to_datetime(start_date)
    df_test = df.loc[mask].copy()
    if df_test.empty: return 0, 0, pd.DataFrame(), pd.DataFrame()

    capital = initial_capital
    position = 0 # 正股股數 或 期權張數
    
    trade_log = []
    equity_curve = []
    
    # 交易狀態記錄
    entry_price = 0       # 正股進場價
    entry_opt_price = 0   # 期權進場權利金
    entry_idx = 0         # 進場的時間點 (為了算持有天數)
    strike_price = 0      # 期權行使價
    holding_type = None   # 'stock', 'call', 'put'
    
    r_rate = 0.03 # 假設無風險利率 3%
    
    for i in range(len(df_test)):
        date = df_test.index[i]
        stock_price = df_test['Close'].iloc[i]
        signal = df_test['Signal'].iloc[i]
        
        # --- 每日市值計算 (Mark to Market) ---
        current_equity = capital
        
        if holding_type == 'stock':
            current_equity = position * stock_price
            
        elif holding_type in ['call', 'put']:
            # 計算期權當前理論價
            days_held = (i - entry_idx)
            days_left = 30 - days_held # 假設買入時剩30天
            if days_left <= 0: days_left = 0.01 # 快到期
            
            T_year = days_left / 365.0
            
            # 使用 BS 模型估值
            opt_price = black_scholes_price(stock_price, strike_price, T_year, r_rate, iv_param, holding_type)
            current_equity = capital + (opt_price - entry_opt_price) * position * 100 # 假設每張100股(美股)或自行調整
            # *注意：這裡簡化處理，假設 capital 是保證金或剩餘現金，這裡直接算 總權益 = 剩餘現金 + 期權市值
            # 為了簡單回測，我們假設全倉買入期權 (非常激進!) -> position = 總權利金 / 單價
            current_equity = position * opt_price 

        equity_curve.append(current_equity)

        # --- 交易邏輯 ---
        
        # 1. 買入訊號 (做多)
        if signal == 1:
            # 如果手上有 Put，先平倉
            if holding_type == 'put':
                profit = current_equity - capital_at_entry
                ret_pct = (profit / capital_at_entry) * 100
                trade_log[-1].update({'出場日期': date, '出場價(標的)': stock_price, '盈虧 ($)': profit, '報酬率 (%)': ret_pct})
                capital = current_equity # 資金滾動
                position = 0
                holding_type = None

            # 開倉做多 (如果空倉)
            if position == 0:
                capital_at_entry = capital
                
                if mode == 'Spot':
                    position = capital / stock_price
                    holding_type = 'stock'
                    entry_price = stock_price
                    log_action = "買入正股"
                else: # Options
                    # 買入 ATM Call, 30天到期
                    strike_price = stock_price # ATM
                    entry_opt_price = black_scholes_price(stock_price, strike_price, 30/365, r_rate, iv_param, 'call')
                    # 全倉買入 (High Risk)
                    position = capital / entry_opt_price
                    holding_type = 'call'
                    entry_price = stock_price # 記錄標的價格
                    entry_idx = i
                    log_action = f"Buy Call (K={strike_price:.1f})"

                trade_log.append({
                    '進場日期': date, '動作': log_action, '進場價(標的)': stock_price,
                    '出場日期': None, '出場價(標的)': None, '盈虧 ($)': None, '報酬率 (%)': None
                })

        # 2. 賣出訊號 (正股=平倉, 期權=反手做Put)
        elif signal == -1:
            # 如果手上有正股或 Call，先平倉
            if holding_type in ['stock', 'call']:
                profit = current_equity - capital_at_entry
                ret_pct = (profit / capital_at_entry) * 100
                trade_log[-1].update({'出場日期': date, '出場價(標的)': stock_price, '盈虧 ($)': profit, '報酬率 (%)': ret_pct})
                capital = current_equity
                position = 0
                holding_type = None
            
            # 期權模式下，賣出訊號 = 開倉做 Put (Long Put)
            if mode == 'Options' and position == 0:
                capital_at_entry = capital
                # 買入 ATM Put, 30天到期
                strike_price = stock_price
                entry_opt_price = black_scholes_price(stock_price, strike_price, 30/365, r_rate, iv_param, 'put')
                position = capital / entry_opt_price
                holding_type = 'put'
                entry_price = stock_price
                entry_idx = i
                
                trade_log.append({
                    '進場日期': date, '動作': f"Buy Put (K={strike_price:.1f})", '進場價(標的)': stock_price,
                    '出場日期': None, '出場價(標的)': None, '盈虧 ($)': None, '報酬率 (%)': None
                })

    final_val = equity_curve[-1]
    ret = ((final_val - initial_capital) / initial_capital) * 100
    df_test['Equity'] = equity_curve
    return final_val, ret, pd.DataFrame(trade_log), df_test

# --- 6. 介面 ---
with st.sidebar:
    st.header("🎛️ 交易模式設定")
    if st.button("🗑️ 清除快取"): st.cache_data.clear()
    
    # === 模式選擇 ===
    mode = st.radio("選擇回測模式", ["Spot (正股)", "Options (期權)"], index=1)
    
    st.divider()
    
    # 參數設定
    ticker = st.text_input("股票代號 (QQQ, 700)", value="QQQ").upper()
    initial_cash = st.number_input("本金 ($)", value=100000)
    start_date = st.date_input("開始日期", pd.to_datetime("2023-01-01"))
    
    # 期權專用參數
    iv_val = 0.3
    if mode == "Options":
        st.caption("📉 期權參數")
        iv_val = st.slider("IV (引伸波幅)", 0.1, 1.0, 0.25, help="指數約0.2，個股約0.3-0.5")
    
    st.divider()
    buy_thresh = st.slider("買入門檻 (J < ?)", 0, 40, 20)
    sell_thresh = st.slider("賣出門檻 (J > ?)", 60, 100, 80)
    
    run_btn = st.button("🚀 開始分析", type="primary")

st.title(f"⚖️ V20.0 - {mode} 回測系統")

if run_btn:
    with st.spinner(f"正在模擬 {mode} 交易策略..."):
        df_raw, real_sym = get_stooq_data(ticker)
        
        if df_raw is not None and not df_raw.empty:
            df = calculate_indicators(df_raw)
            df = generate_signals(df, buy_thresh, sell_thresh)
            
            final_val, ret, df_log, df_chart = run_backtest(df, initial_cash, start_date, mode.split()[0], iv_val)
            
            if not df_chart.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("標的", real_sym)
                c2.metric("最終資產", f"${final_val:,.0f}", f"{ret:+.2f}%")
                
                win_rate = 0
                if not df_log.empty:
                    closed = df_log.dropna(subset=['盈虧 ($)'])
                    if len(closed) > 0:
                        wins = len(closed[closed['盈虧 ($)'] > 0])
                        win_rate = (wins / len(closed)) * 100
                c3.metric("勝率", f"{win_rate:.1f}%", f"共 {len(df_log)} 筆")
                
                # 圖表
                st.subheader("資產走勢對比")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
                
                # 資產曲線 (如果是期權，波動會很大)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Equity'], fill='tozeroy', 
                                         line=dict(color='#00ff00' if mode=='Spot' else '#ffaa00'), name='總資產'), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['J'], line=dict(color='#ab63fa'), name='J線'), row=2, col=1)
                fig.add_hline(y=buy_thresh, line_dash="dot", row=2, col=1, line_color="green")
                fig.add_hline(y=sell_thresh, line_dash="dot", row=2, col=1, line_color="red")
                
                fig.update_layout(height=600, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # 交易紀錄
                st.subheader("交易紀錄")
                if not df_log.empty:
                    display = df_log.copy()
                    display['進場日期'] = display['進場日期'].dt.date
                    display['出場日期'] = pd.to_datetime(display['出場日期']).dt.date
                    
                    def color_row(val):
                        if pd.isna(val): return ''
                        return 'color: lightgreen' if val > 0 else 'color: #ff5555'

                    st.dataframe(display.style.format({
                        "進場價(標的)": "{:.2f}", "出場價(標的)": "{:.2f}",
                        "盈虧 ($)": "{:+.2f}", "報酬率 (%)": "{:+.2f}%"
                    }).map(color_row, subset=['盈虧 ($)', '報酬率 (%)']), use_container_width=True)
        else:
            st.error("無法取得數據")
