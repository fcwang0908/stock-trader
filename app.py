# ==========================================
# 老陳 AI 交易系統 V22.0 - 收租組合 + 注碼管理版
# 核心升級：
# 1. 新增「懶人組合選單」：一鍵選擇收租股 (823, 5, 941...)
# 2. 新增「注碼管理」：可選 All-in 或 定額投入 (如每次 $20,000)
# 3. 交易紀錄升級：顯示「每次投入金額」與「累積股數」
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

st.set_page_config(page_title="老陳 V22.0 (收租版)", layout="wide", page_icon="🏙️")

# --- 0. 懶人組合清單 ---
PRESETS = {
    "自行輸入": "MHI",
    "🏙️ 收租三寶 (港股)": {
        "823.HK (領展)": "823",
        "0005.HK (匯豐)": "5",
        "0941.HK (中移動)": "941",
        "0002.HK (中電)": "2",
        "0066.HK (港鐵)": "66"
    },
    "🚀 科技龍頭 (港股)": {
        "0700.HK (騰訊)": "700",
        "9988.HK (阿里)": "9988",
        "3690.HK (美團)": "3690",
        "3033.HK (科指ETF)": "3033"
    },
    "🇺🇸 美股 ETF": {
        "QQQ (納指)": "QQQ",
        "SPY (標普)": "SPY",
        "TLT (美債)": "TLT",
        "NVDA (輝達)": "NVDA"
    }
}

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

# --- 5. 回測引擎 (含注碼管理) ---
def run_backtest(df, initial_capital, start_date, end_date, 
                 mode_str, size_type, fixed_amt, 
                 iv_param=0.3):
    
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    df_test = df.loc[mask].copy()
    if df_test.empty: return 0, 0, pd.DataFrame(), pd.DataFrame()

    capital = initial_capital
    position = 0
    trade_log = []
    equity_curve = []
    
    entry_opt_price = 0
    entry_idx = 0
    strike_price = 0
    holding_type = None 
    
    # 記錄這一筆單「實際上」投入了多少錢 (Principal)
    invested_amount = 0 
    
    r_rate = 0.03
    is_option_mode = ("Options" in mode_str)

    for i in range(len(df_test)):
        date = df_test.index[i]
        stock_price = df_test['Close'].iloc[i]
        signal = df_test['Signal'].iloc[i]
        
        current_equity = capital
        
        # --- 市值計算 ---
        if holding_type == 'stock':
            current_equity = capital + (position * stock_price) - invested_amount # 現金 + 股票市值
            # 簡化算法：總資產 = 閒置現金 + 持倉市值
            # 因為我們買入時扣除了現金，所以 current_equity = current_cash + (pos * price)
        elif holding_type in ['call', 'put']:
            days_held = (i - entry_idx)
            days_left = 30 - days_held
            if days_left <= 0: days_left = 0.01
            T_year = days_left / 365.0
            opt_price = black_scholes_price(stock_price, strike_price, T_year, r_rate, iv_param, holding_type)
            # 權益 = 剩餘現金 + 期權市值
            # 這裡我們用簡單模型：Current Val = Capital (Reset at entry) + PnL
            # 為保持一致：
            current_equity = (capital - invested_amount) + (position * opt_price)

        equity_curve.append(current_equity if position > 0 else capital)

        # --- 交易執行 ---
        
        # 定義：計算該買多少
        def calc_position_size(price_per_unit):
            if size_type == "全倉 (All-in)":
                return capital # 投入所有資金
            else: # 定額
                return min(capital, fixed_amt) # 如果剩餘資金不足 fixed_amt，就梭哈剩餘的

        # 1. 買入訊號
        if signal == 1:
            # 平空倉
            if holding_type == 'put':
                # 賣出 Put 拿回現金
                cash_back = position * black_scholes_price(stock_price, strike_price, (30-(i-entry_idx))/365, r_rate, iv_param, 'put')
                capital = (capital - invested_amount) + cash_back
                profit = cash_back - invested_amount
                pct = (profit/invested_amount)*100
                trade_log[-1].update({'出場日期': date, '出場價': stock_price, '盈虧 ($)': profit, '報酬率 (%)': pct})
                position = 0
                holding_type = None
            
            # 開多倉
            if position == 0:
                if not is_option_mode: # 正股
                    invest_money = calc_position_size(stock_price)
                    if invest_money > 0:
                        position = invest_money / stock_price
                        invested_amount = invest_money
                        holding_type = 'stock'
                        trade_log.append({
                            '進場日期': date, '動作': '買入', 
                            '投入金額 ($)': invested_amount, 
                            '進場價': stock_price, '持有數量': position,
                            '出場日期': None, '盈虧 ($)': None, '報酬率 (%)': None
                        })
                else: # Call
                    strike_price = stock_price
                    opt_price = black_scholes_price(stock_price, strike_price, 30/365, r_rate, iv_param, 'call')
                    invest_money = calc_position_size(opt_price)
                    if invest_money > 0:
                        position = invest_money / opt_price
                        invested_amount = invest_money
                        holding_type = 'call'
                        entry_idx = i
                        entry_opt_price = opt_price
                        trade_log.append({
                            '進場日期': date, '動作': f'Long Call ({strike_price:.0f})', 
                            '投入金額 ($)': invested_amount,
                            '進場價': stock_price, '持有數量': position,
                            '出場日期': None, '盈虧 ($)': None, '報酬率 (%)': None
                        })

        # 2. 賣出訊號
        elif signal == -1:
            # 平多倉
            if holding_type in ['stock', 'call']:
                # 賣出拿回現金
                if holding_type == 'stock':
                    cash_back = position * stock_price
                else:
                    cash_back = position * black_scholes_price(stock_price, strike_price, (30-(i-entry_idx))/365, r_rate, iv_param, 'call')
                
                capital = (capital - invested_amount) + cash_back # 更新總本金
                profit = cash_back - invested_amount
                pct = (profit/invested_amount)*100
                trade_log[-1].update({'出場日期': date, '出場價': stock_price, '盈虧 ($)': profit, '報酬率 (%)': pct})
                position = 0
                holding_type = None
            
            # 開空倉 (僅期權)
            if is_option_mode and position == 0:
                strike_price = stock_price
                opt_price = black_scholes_price(stock_price, strike_price, 30/365, r_rate, iv_param, 'put')
                invest_money = calc_position_size(opt_price)
                if invest_money > 0:
                    position = invest_money / opt_price
                    invested_amount = invest_money
                    holding_type = 'put'
                    entry_idx = i
                    entry_opt_price = opt_price
                    trade_log.append({
                        '進場日期': date, '動作': f'Long Put ({strike_price:.0f})', 
                        '投入金額 ($)': invested_amount,
                        '進場價': stock_price, '持有數量': position,
                        '出場日期': None, '盈虧 ($)': None, '報酬率 (%)': None
                    })

    final_val = equity_curve[-1]
    ret = ((final_val - initial_capital) / initial_capital) * 100
    df_test['Equity'] = equity_curve
    return final_val, ret, pd.DataFrame(trade_log), df_test

# --- 6. 介面 ---
with st.sidebar:
    st.header("🎛️ V22.0 設定")
    if st.button("🗑️ 清除快取"): st.cache_data.clear()
    
    # === 1. 懶人組合選單 ===
    st.subheader("1. 選擇標的")
    category = st.selectbox("分類", list(PRESETS.keys()))
    
    if category == "自行輸入":
        ticker_input = st.text_input("輸入代號", value="MHI").upper()
    else:
        # 顯示該分類下的股票
        selected_stock = st.selectbox("股票", list(PRESETS[category].keys()))
        ticker_input = PRESETS[category][selected_stock]
    
    st.caption(f"當前分析代號: {ticker_input}")

    # === 2. 交易模式 ===
    st.divider()
    mode = st.radio("交易工具", ["Spot (正股)", "Options (期權)"], index=0)
    
    iv_val = 0.3 
    if "Options" in mode:
        iv_val = st.slider("IV (引伸波幅)", 0.1, 1.0, 0.25)

    # === 3. 注碼管理 (重點更新) ===
    st.subheader("2. 注碼管理")
    initial_cash = st.number_input("總本金 ($)", value=100000)
    
    size_type = st.radio("每次投入方式", ["全倉 (All-in)", "定額 (Fixed Amount)"])
    fixed_amt = 0
    if size_type == "定額 (Fixed Amount)":
        fixed_amt = st.number_input("每次投入金額 ($)", value=20000, step=5000)
        st.info(f"每次訊號出現時，將買入 ${fixed_amt:,} 的貨。")
    
    # 日期與參數
    st.divider()
    col_d1, col_d2 = st.columns(2)
    with col_d1: start_date = st.date_input("開始", pd.to_datetime("2023-01-01"))
    with col_d2: end_date = st.date_input("結束", datetime.today())
    
    buy_thresh = st.slider("買入 (J <)", 0, 40, 20)
    sell_thresh = st.slider("賣出 (J >)", 60, 100, 80)
    
    run_btn = st.button("🚀 執行回測", type="primary")

st.title(f"🏙️ V22.0 - 收租組合 & 注碼版")

if run_btn:
    if start_date > end_date:
        st.error("日期錯誤")
    else:
        with st.spinner(f"正在分析 {ticker_input} ({size_type})..."):
            df_raw, real_sym = get_stooq_data(ticker_input)
            
            if df_raw is not None and not df_raw.empty:
                df = calculate_indicators(df_raw)
                df = generate_signals(df, buy_thresh, sell_thresh)
                
                final_val, ret, df_log, df_chart = run_backtest(
                    df, initial_cash, start_date, end_date, 
                    mode, size_type, fixed_amt, iv_val
                )
                
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
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Equity'], fill='tozeroy', line=dict(color='#00ff00' if 'Spot' in mode else '#ffaa00'), name='資產'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['J'], line=dict(color='#ab63fa'), name='J線'), row=2, col=1)
                    fig.add_hline(y=buy_thresh, line_dash="dot", row=2, col=1, line_color="green")
                    fig.add_hline(y=sell_thresh, line_dash="dot", row=2, col=1, line_color="red")
                    fig.update_layout(height=600, template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("詳細交易紀錄 (含注碼)")
                    if not df_log.empty:
                        disp = df_log.copy()
                        disp['進場日期'] = disp['進場日期'].dt.date
                        disp['出場日期'] = pd.to_datetime(disp['出場日期']).dt.date
                        
                        def color_row(val):
                            if pd.isna(val): return ''
                            return 'color: lightgreen' if val > 0 else 'color: #ff5555'
                        
                        # 格式化顯示
                        st.dataframe(disp.style.format({
                            "投入金額 ($)": "{:,.0f}",
                            "進場價": "{:.2f}", 
                            "出場價": "{:.2f}", 
                            "盈虧 ($)": "{:+.0f}", 
                            "報酬率 (%)": "{:+.2f}%",
                            "持有數量": "{:.2f}"
                        }).map(color_row, subset=['盈虧 ($)', '報酬率 (%)']), use_container_width=True)
            else:
                st.warning("無數據")
