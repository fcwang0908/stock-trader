# ==========================================
# 老陳 AI 交易系統 V21.0 - 期權價差策略版
# 核心升級：
# 1. 新增 Vertical Spread (垂直價差) 回測
# 2. Bull Call Spread (看升) / Bear Put Spread (看跌)
# 3. 自動計算組合單 (Leg 1 - Leg 2) 的淨值變化
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
from scipy.stats import norm 
from datetime import datetime

st.set_page_config(page_title="老陳 V21.0 (價差策略)", layout="wide", page_icon="🦋")

# --- 1. Black-Scholes 模型 ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if S <= 0 or K <= 0 or T <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0.01)

# --- 2. 數據獲取 ---
@st.cache_data(ttl=3600)
def get_stooq_data(symbol):
    raw_sym = symbol.upper().strip()
    clean_sym = raw_sym 
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
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
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

# --- 5. 回測引擎 (支援價差組合) ---
def run_backtest(df, initial_capital, start_date, end_date, strategy_type, spread_width_pct, iv_param=0.3):
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    df_test = df.loc[mask].copy()
    if df_test.empty: return 0, 0, pd.DataFrame(), pd.DataFrame()

    capital = initial_capital
    position = 0
    trade_log = []
    equity_curve = []
    
    # 交易狀態變數
    entry_idx = 0
    
    # 單腿模式變數
    strike_price = 0
    
    # 價差模式變數 (Leg 1 = Long, Leg 2 = Short)
    strike_long = 0
    strike_short = 0
    
    holding_type = None # 'stock', 'long_call', 'long_put', 'bull_spread', 'bear_spread'
    capital_at_entry = 0
    
    r_rate = 0.03
    days_to_expiry = 30 # 假設都做近月

    for i in range(len(df_test)):
        date = df_test.index[i]
        stock_price = df_test['Close'].iloc[i]
        signal = df_test['Signal'].iloc[i]
        
        current_equity = capital
        
        # --- 1. 計算持倉市值 (Mark to Market) ---
        if holding_type == 'stock':
            current_equity = position * stock_price
            
        elif holding_type: # 期權相關 (單腿或價差)
            days_held = (i - entry_idx)
            days_left = days_to_expiry - days_held
            if days_left <= 0: days_left = 0.01
            T_year = days_left / 365.0
            
            # 計算目前價值
            if holding_type == 'long_call':
                val = black_scholes_price(stock_price, strike_price, T_year, r_rate, iv_param, 'call')
                current_equity = position * val
                
            elif holding_type == 'long_put':
                val = black_scholes_price(stock_price, strike_price, T_year, r_rate, iv_param, 'put')
                current_equity = position * val
                
            elif holding_type == 'bull_spread': 
                # Bull Call Spread = Long ATM Call - Short OTM Call
                val_long = black_scholes_price(stock_price, strike_long, T_year, r_rate, iv_param, 'call')
                val_short = black_scholes_price(stock_price, strike_short, T_year, r_rate, iv_param, 'call')
                spread_val = val_long - val_short # 淨值
                current_equity = position * spread_val
                
            elif holding_type == 'bear_spread':
                # Bear Put Spread = Long ATM Put - Short OTM Put
                val_long = black_scholes_price(stock_price, strike_long, T_year, r_rate, iv_param, 'put')
                val_short = black_scholes_price(stock_price, strike_short, T_year, r_rate, iv_param, 'put')
                spread_val = val_long - val_short
                current_equity = position * spread_val

        equity_curve.append(current_equity)

        # --- 2. 交易執行邏輯 ---
        
        # 定義平倉函數
        def close_position():
            nonlocal capital, position, holding_type
            profit = current_equity - capital_at_entry
            pct = (profit/capital_at_entry)*100 if capital_at_entry > 0 else 0
            trade_log[-1].update({'出場日期': date, '出場價(標的)': stock_price, '盈虧 ($)': profit, '報酬率 (%)': pct})
            capital = current_equity
            position = 0
            holding_type = None

        # 訊號 1: 買入 (Bull)
        if signal == 1:
            # 如果持有空頭部位 (Put / Bear Spread)，先平倉
            if holding_type in ['long_put', 'bear_spread']:
                close_position()

            # 開倉 Bull 部位
            if position == 0:
                capital_at_entry = capital
                entry_idx = i
                
                if strategy_type == 'Spot (正股)':
                    position = capital / stock_price
                    holding_type = 'stock'
                    trade_log.append({'進場日期': date, '動作': 'Buy Stock', '進場價(標的)': stock_price, '出場日期': None, '盈虧 ($)': None, '報酬率 (%)': None})
                
                elif strategy_type == 'Single Option (單腿)':
                    # Long ATM Call
                    strike_price = stock_price
                    cost = black_scholes_price(stock_price, strike_price, days_to_expiry/365, r_rate, iv_param, 'call')
                    position = capital / cost
                    holding_type = 'long_call'
                    trade_log.append({'進場日期': date, '動作': f'Long Call (K={strike_price:.0f})', '進場價(標的)': stock_price, '出場日期': None, '盈虧 ($)': None, '報酬率 (%)': None})
                
                elif strategy_type == 'Spread (價差組合)':
                    # Bull Call Spread: Long ATM, Short OTM
                    strike_long = stock_price
                    strike_short = stock_price * (1 + spread_width_pct/100)
                    
                    p_long = black_scholes_price(stock_price, strike_long, days_to_expiry/365, r_rate, iv_param, 'call')
                    p_short = black_scholes_price(stock_price, strike_short, days_to_expiry/365, r_rate, iv_param, 'call')
                    
                    net_debit = p_long - p_short
                    position = capital / net_debit
                    holding_type = 'bull_spread'
                    trade_log.append({'進場日期': date, '動作': f'Bull Spread (L:{strike_long:.0f}/S:{strike_short:.0f})', '進場價(標的)': stock_price, '出場日期': None, '盈虧 ($)': None, '報酬率 (%)': None})

        # 訊號 -1: 賣出 (Bear)
        elif signal == -1:
            # 如果持有 Bull 部位，先平倉
            if holding_type in ['stock', 'long_call', 'bull_spread']:
                close_position()

            # 開倉 Bear 部位 (正股模式不做空，只平倉)
            if position == 0 and strategy_type != 'Spot (正股)':
                capital_at_entry = capital
                entry_idx = i
                
                if strategy_type == 'Single Option (單腿)':
                    # Long ATM Put
                    strike_price = stock_price
                    cost = black_scholes_price(stock_price, strike_price, days_to_expiry/365, r_rate, iv_param, 'put')
                    position = capital / cost
                    holding_type = 'long_put'
                    trade_log.append({'進場日期': date, '動作': f'Long Put (K={strike_price:.0f})', '進場價(標的)': stock_price, '出場日期': None, '盈虧 ($)': None, '報酬率 (%)': None})
                
                elif strategy_type == 'Spread (價差組合)':
                    # Bear Put Spread: Long ATM, Short OTM (lower strike)
                    strike_long = stock_price
                    strike_short = stock_price * (1 - spread_width_pct/100)
                    
                    p_long = black_scholes_price(stock_price, strike_long, days_to_expiry/365, r_rate, iv_param, 'put')
                    p_short = black_scholes_price(stock_price, strike_short, days_to_expiry/365, r_rate, iv_param, 'put')
                    
                    net_debit = p_long - p_short
                    position = capital / net_debit
                    holding_type = 'bear_spread'
                    trade_log.append({'進場日期': date, '動作': f'Bear Spread (L:{strike_long:.0f}/S:{strike_short:.0f})', '進場價(標的)': stock_price, '出場日期': None, '盈虧 ($)': None, '報酬率 (%)': None})

    final_val = equity_curve[-1]
    ret = ((final_val - initial_capital) / initial_capital) * 100
    df_test['Equity'] = equity_curve
    return final_val, ret, pd.DataFrame(trade_log), df_test

# --- 6. 介面 ---
with st.sidebar:
    st.header("🎛️ 參數控制 (V21.0)")
    if st.button("🗑️ 清除快取"): st.cache_data.clear()
    
    # === 策略選擇器 ===
    strat = st.selectbox("交易策略", ["Spot (正股)", "Single Option (單腿)", "Spread (價差組合)"], index=2)
    
    # 期權參數
    iv_val = 0.3
    spread_width = 5.0
    
    if strat != "Spot (正股)":
        st.info(f"⚙️ {strat} 參數")
        iv_val = st.slider("IV (引伸波幅)", 0.1, 1.0, 0.25)
        
        if strat == "Spread (價差組合)":
            spread_width = st.slider("價差闊度 (%)", 1.0, 10.0, 3.0, help="Long Leg 與 Short Leg 的行使價距離")
    
    st.divider()
    
    ticker = st.text_input("代號", value="QQQ").upper()
    initial_cash = st.number_input("本金", value=100000)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1: start_date = st.date_input("開始", pd.to_datetime("2023-01-01"))
    with col_d2: end_date = st.date_input("結束", datetime.today())
    
    st.divider()
    buy_thresh = st.slider("買入 (J <)", 0, 40, 20)
    sell_thresh = st.slider("賣出 (J >)", 60, 100, 80)
    
    run_btn = st.button("🚀 執行回測", type="primary")

st.title(f"🦋 V21.0 - {strat} 回測")

if run_btn:
    if start_date > end_date:
        st.error("日期錯誤")
    else:
        with st.spinner("模擬價差策略中..."):
            df_raw, real_sym = get_stooq_data(ticker)
            
            if df_raw is not None and not df_raw.empty:
                df = calculate_indicators(df_raw)
                df = generate_signals(df, buy_thresh, sell_thresh)
                
                final_val, ret, df_log, df_chart = run_backtest(df, initial_cash, start_date, end_date, strat, spread_width, iv_val)
                
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
                    
                    st.subheader("資產走勢")
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Equity'], fill='tozeroy', line=dict(color='#00ff00' if strat=='Spot (正股)' else '#ffaa00'), name='資產'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['J'], line=dict(color='#ab63fa'), name='J線'), row=2, col=1)
                    fig.add_hline(y=buy_thresh, line_dash="dot", row=2, col=1, line_color="green")
                    fig.add_hline(y=sell_thresh, line_dash="dot", row=2, col=1, line_color="red")
                    fig.update_layout(height=600, template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("交易紀錄")
                    if not df_log.empty:
                        disp = df_log.copy()
                        disp['進場日期'] = disp['進場日期'].dt.date
                        disp['出場日期'] = pd.to_datetime(disp['出場日期']).dt.date
                        def color_row(val):
                            if pd.isna(val): return ''
                            return 'color: lightgreen' if val > 0 else 'color: #ff5555'
                        st.dataframe(disp.style.format({"進場價(標的)": "{:.2f}", "出場價(標的)": "{:.2f}", "盈虧 ($)": "{:+.2f}", "報酬率 (%)": "{:+.2f}%"}).map(color_row, subset=['盈虧 ($)', '報酬率 (%)']), use_container_width=True)
            else:
                st.warning("無數據")
