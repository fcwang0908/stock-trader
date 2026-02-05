# ==========================================
# 老陳 AI 交易系統 V25.0 - AI 預測旗艦版
# 新增功能：
# 1. 🔮 AI 未來預測：使用 Monte Carlo 模擬未來 5 天股價
# 2. 📉 J線教學圖層：圖表上清楚標示 80/20 區域
# 3. 修復所有顯示問題
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
from scipy.stats import norm 
from datetime import datetime, timedelta

st.set_page_config(page_title="老陳 V25.0 (AI 預測版)", layout="wide", page_icon="🔮")

# --- 0. 全局設定 ---
PRESETS = {
    "自行輸入": "MHI",
    "🏙️ 收租三寶": {"823 領展": "823", "5 匯豐": "5", "941 中移動": "941"},
    "🚀 科技龍頭": {"700 騰訊": "700", "9988 阿里": "9988", "3690 美團": "3690"},
    "🇺🇸 美股 ETF": {"QQQ 納指": "QQQ", "SPY 標普": "SPY", "TLT 美債": "TLT", "NVDA": "NVDA"}
}

# --- 1. 核心函數 ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if S <= 0 or K <= 0 or T <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(price, 0.01)

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

def calculate_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    low_9 = df['Low'].rolling(9).min()
    high_9 = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    pos_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    pos_mf = pd.Series(pos_flow).rolling(14).sum()
    neg_mf = pd.Series(neg_flow).rolling(14).sum()
    mfi_ratio = np.divide(pos_mf, neg_mf, out=np.zeros_like(pos_mf), where=neg_mf!=0)
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    return df

def generate_signals(df, buy_thresh, sell_thresh):
    df['Signal'] = 0 
    buy_cond = (df['J'] < buy_thresh) & (df['J'] > df['J'].shift(1))
    sell_cond = (df['J'] > sell_thresh) & (df['J'] < df['J'].shift(1))
    df.loc[buy_cond, 'Signal'] = 1
    df.loc[sell_cond, 'Signal'] = -1
    return df

# --- 2. 新增：AI 蒙地卡羅預測模組 ---
def run_monte_carlo(df, days=5, simulations=100):
    last_price = df['Close'].iloc[-1]
    # 計算日回報率的平均值與標準差 (Volatility)
    returns = df['Close'].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    
    simulation_df = pd.DataFrame()
    
    # 模擬 100 條路徑
    for x in range(simulations):
        price_series = []
        price = last_price
        for d in range(days):
            # 隨機漫步公式
            price = price * (1 + np.random.normal(mu, sigma))
            price_series.append(price)
        simulation_df[x] = price_series
        
    # 統計數據
    future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, days+1)]
    mean_path = simulation_df.mean(axis=1).values
    upper_bound = simulation_df.quantile(0.95, axis=1).values # 95% 信心上限
    lower_bound = simulation_df.quantile(0.05, axis=1).values # 5% 信心下限
    
    return future_dates, mean_path, upper_bound, lower_bound

# --- 3. 顯示模組 ---
def render_market_scan(df, real_sym):
    st.header(f"🔮 AI 戰情室: {real_sym}")
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    change = last['Close'] - prev['Close']
    pct = (change / prev['Close']) * 100
    
    # 頂部看板
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新價", f"{last['Close']:,.2f}", f"{change:+.2f} ({pct:.2f}%)")
    c2.metric("MFI 資金流", f"{last['MFI']:.1f}", help=">80為超買, <20為超賣")
    c3.metric("J 線 (動能)", f"{last['J']:.1f}", help="橡筋理論：>80太貴, <20太便宜")
    
    # AI 判讀
    sentiment = "中性"
    score = 50
    if last['J'] < 20: 
        sentiment = "🟢 強力看漲 (超賣)"
        score = 90
    elif last['J'] > 80: 
        sentiment = "🔴 強力看跌 (超買)"
        score = 10
    c4.metric("AI 訊號判讀", sentiment)

    # --- AI 預測圖表 ---
    st.subheader("📈 價格走勢 & AI 未來 5 日預測")
    
    # 執行預測
    f_dates, f_mean, f_upper, f_lower = run_monte_carlo(df)
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3],
                        vertical_spacing=0.05)
    
    # 1. 主圖：K線 + AI 預測
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='歷史K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='MA60'), row=1, col=1)
    
    # 繪製預測區間 (扇形圖)
    fig.add_trace(go.Scatter(x=f_dates, y=f_upper, mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_dates, y=f_lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', name='AI 預測區間 (95%)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_dates, y=f_mean, mode='lines', line=dict(color='yellow', dash='dot'), name='AI 預期中位數'), row=1, col=1)

    # 2. 成交量
    colors = ['green' if c>=o else 'red' for c,o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # 3. J線與超買超賣區 (重點教學)
    fig.add_trace(go.Scatter(x=df.index, y=df['J'], line=dict(color='#ab63fa', width=2), name='J線 (橡筋)'), row=3, col=1)
    
    # 畫出 80/20 界線
    fig.add_hline(y=80, line_dash="dot", row=3, col=1, line_color="red", annotation_text="超買區 (太貴)", annotation_position="top left")
    fig.add_hline(y=20, line_dash="dot", row=3, col=1, line_color="green", annotation_text="超賣區 (太便宜)", annotation_position="bottom left")
    
    # 標示背景顏色，讓用戶更直觀
    fig.add_hrect(y0=80, y1=120, row=3, col=1, fillcolor="red", opacity=0.1, layer="below", annotation_text="🔴 賣出風險")
    fig.add_hrect(y0=-20, y1=20, row=3, col=1, fillcolor="green", opacity=0.1, layer="below", annotation_text="🟢 買入機會")

    fig.update_layout(height=900, template="plotly_dark", showlegend=True, 
                      title_text=f"{real_sym} - AI 趨勢分析")
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"💡 AI 解讀：黃色虛線是 AI 模擬未來 5 天最可能的走勢。綠色陰影範圍代表 95% 機率會到達的區間。")

def render_strategy_lab(df, real_sym):
    st.header(f"🦅 策略工廠: {real_sym}")
    last = df.iloc[-1]
    st.info(f"價格: {last['Close']:.2f} | J線: {last['J']:.1f}")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("趨勢策略")
        if last['J'] < 20:
            st.success("🚀 看升 (Bullish)")
            st.markdown(f"**Bull Call Spread**\n* Buy Call @ {last['Close']:.1f}\n* Sell Call @ {last['Close']*1.05:.1f}")
        elif last['J'] > 80:
            st.error("📉 看跌 (Bearish)")
            st.markdown(f"**Bear Put Spread**\n* Buy Put @ {last['Close']:.1f}\n* Sell Put @ {last['Close']*0.95:.1f}")
        else:
            st.warning("觀望")
    with c2:
        st.subheader("盤整策略")
        st.write("Iron Condor (鐵兀鷹)")

# --- 4. 回測引擎 ---
def run_advanced_backtest(df, initial_capital, start_date, end_date, 
                          mode_str, opt_strat, spread_width_pct,
                          size_type, fixed_amt, iv_param=0.3):
    
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    df_test = df.loc[mask].copy()
    if df_test.empty: return 0, 0, pd.DataFrame(), pd.DataFrame()

    capital = initial_capital
    position = 0
    trade_log = []
    equity_curve = []
    
    entry_idx = 0
    invested_amount = 0
    holding_type = None 
    strike_long = 0
    strike_short = 0
    r_rate = 0.03
    is_option_mode = ("Options" in mode_str)

    def calc_position_size(unit_cost):
        if size_type == "全倉 (All-in)": return capital
        else: return min(capital, fixed_amt)

    for i in range(len(df_test)):
        date = df_test.index[i]
        stock_price = df_test['Close'].iloc[i]
        signal = df_test['Signal'].iloc[i]
        
        current_equity = capital
        
        if holding_type == 'stock':
            current_equity = (capital - invested_amount) + (position * stock_price)
        elif holding_type: 
            days_held = (i - entry_idx)
            days_left = max(0.01, 30 - days_held)
            T_yr = days_left / 365.0
            unit_val = 0
            if holding_type == 'long_call':
                unit_val = black_scholes_price(stock_price, strike_long, T_yr, r_rate, iv_param, 'call')
            elif holding_type == 'long_put':
                unit_val = black_scholes_price(stock_price, strike_long, T_yr, r_rate, iv_param, 'put')
            elif holding_type == 'bull_spread':
                val_L = black_scholes_price(stock_price, strike_long, T_yr, r_rate, iv_param, 'call')
                val_S = black_scholes_price(stock_price, strike_short, T_yr, r_rate, iv_param, 'call')
                unit_val = val_L - val_S
            elif holding_type == 'bear_spread':
                val_L = black_scholes_price(stock_price, strike_long, T_yr, r_rate, iv_param, 'put')
                val_S = black_scholes_price(stock_price, strike_short, T_yr, r_rate, iv_param, 'put')
                unit_val = val_L - val_S
            current_equity = (capital - invested_amount) + (position * unit_val)

        equity_curve.append(current_equity)

        def close_pos():
            nonlocal capital, position, holding_type
            cash_back = current_equity - (capital - invested_amount)
            profit = cash_back - invested_amount
            return profit

        if signal == 1: 
            if holding_type in ['long_put', 'bear_spread']: 
                profit = close_pos()
                trade_log[-1].update({'出場日期': date, '出場價': stock_price, '盈虧': profit, '回報%': (profit/invested_amount)*100})
                capital = current_equity; position = 0; holding_type = None

            if position == 0:
                if not is_option_mode:
                    amt = calc_position_size(stock_price)
                    if amt > 0:
                        position = amt / stock_price
                        invested_amount = amt
                        holding_type = 'stock'
                        trade_log.append({'進場日期': date, '動作': 'Buy Stock', '投入': amt, '進場價': stock_price, '出場日期': None, '出場價': None, '盈虧': None, '回報%': None})
                else:
                    entry_idx = i
                    if opt_strat == "Single (單腿)":
                        strike_long = stock_price
                        cost = black_scholes_price(stock_price, strike_long, 30/365, r_rate, iv_param, 'call')
                        amt = calc_position_size(cost)
                        if amt > 0:
                            position = amt / cost
                            invested_amount = amt
                            holding_type = 'long_call'
                            trade_log.append({'進場日期': date, '動作': f'Long Call ({strike_long:.0f})', '投入': amt, '進場價': stock_price, '出場日期': None, '出場價': None, '盈虧': None, '回報%': None})
                    else: 
                        strike_long = stock_price
                        strike_short = stock_price * (1 + spread_width_pct/100)
                        cost_L = black_scholes_price(stock_price, strike_long, 30/365, r_rate, iv_param, 'call')
                        cost_S = black_scholes_price(stock_price, strike_short, 30/365, r_rate, iv_param, 'call')
                        net_debit = cost_L - cost_S
                        amt = calc_position_size(net_debit)
                        if amt > 0:
                            position = amt / net_debit
                            invested_amount = amt
                            holding_type = 'bull_spread'
                            trade_log.append({'進場日期': date, '動作': f'Bull Spread ({strike_long:.0f}/{strike_short:.0f})', '投入': amt, '進場價': stock_price, '出場日期': None, '出場價': None, '盈虧': None, '回報%': None})

        elif signal == -1: 
            if holding_type in ['stock', 'long_call', 'bull_spread']: 
                profit = close_pos()
                trade_log[-1].update({'出場日期': date, '出場價': stock_price, '盈虧': profit, '回報%': (profit/invested_amount)*100})
                capital = current_equity; position = 0; holding_type = None
            
            if is_option_mode and position == 0:
                entry_idx = i
                if opt_strat == "Single (單腿)":
                    strike_long = stock_price
                    cost = black_scholes_price(stock_price, strike_long, 30/365, r_rate, iv_param, 'put')
                    amt = calc_position_size(cost)
                    if amt > 0:
                        position = amt / cost
                        invested_amount = amt
                        holding_type = 'long_put'
                        trade_log.append({'進場日期': date, '動作': f'Long Put ({strike_long:.0f})', '投入': amt, '進場價': stock_price, '出場日期': None, '出場價': None, '盈虧': None, '回報%': None})
                else:
                    strike_long = stock_price
                    strike_short = stock_price * (1 - spread_width_pct/100)
                    cost_L = black_scholes_price(stock_price, strike_long, 30/365, r_rate, iv_param, 'put')
                    cost_S = black_scholes_price(stock_price, strike_short, 30/365, r_rate, iv_param, 'put')
                    net_debit = cost_L - cost_S
                    amt = calc_position_size(net_debit)
                    if amt > 0:
                        position = amt / net_debit
                        invested_amount = amt
                        holding_type = 'bear_spread'
                        trade_log.append({'進場日期': date, '動作': f'Bear Spread ({strike_long:.0f}/{strike_short:.0f})', '投入': amt, '進場價': stock_price, '出場日期': None, '出場價': None, '盈虧': None, '回報%': None})

    df_test['Equity'] = equity_curve
    final_val = equity_curve[-1] if equity_curve else initial_capital
    ret = ((final_val - initial_capital) / initial_capital) * 100
    return final_val, ret, pd.DataFrame(trade_log), df_test

# --- 5. 控制台 ---
with st.sidebar:
    st.title("🎛️ AI 戰情室控制台")
    app_mode = st.radio("功能", ["🔮 市場掃描 (含AI預測)", "🦅 策略工廠", "⚙️ 回測實驗室"])
    st.divider()
    
    st.subheader("1. 標的選擇")
    cat = st.selectbox("分類", list(PRESETS.keys()))
    if cat == "自行輸入": ticker_input = st.text_input("代號", value="MHI").upper()
    else: ticker_input = PRESETS[cat][st.selectbox("股票", list(PRESETS[cat].keys()))]
    
    if st.button("🗑️ 清除快取"): st.cache_data.clear()
    
    st.divider()
    st.subheader("2. 訊號參數")
    buy_thresh = st.slider("買入 (J <)", 0, 40, 20)
    sell_thresh = st.slider("賣出 (J >)", 60, 100, 80)
    
    bp = {}
    if app_mode == "⚙️ 回測實驗室":
        st.divider()
        st.subheader("3. 回測設定")
        bp['mode'] = st.radio("工具", ["Spot (正股)", "Options (期權)"])
        
        bp['opt_strat'] = "Single (單腿)"
        bp['width'] = 5.0
        bp['iv'] = 0.3
        
        if "Options" in bp['mode']:
            bp['opt_strat'] = st.selectbox("期權策略", ["Single (單腿)", "Spread (價差組合)"])
            bp['iv'] = st.slider("IV (波動率)", 0.1, 1.0, 0.25)
            if bp['opt_strat'] == "Spread (價差組合)":
                bp['width'] = st.slider("價差闊度 (%)", 1.0, 10.0, 5.0)
        
        bp['size'] = st.radio("注碼", ["全倉 (All-in)", "定額"])
        bp['amt'] = st.number_input("每次金額", value=20000) if bp['size']=="定額" else 0
        
        c1, c2 = st.columns(2)
        with c1: bp['start'] = st.date_input("開始", pd.to_datetime("2023-01-01"))
        with c2: bp['end'] = st.date_input("結束", datetime.today())
        
    run_btn = st.button("🚀 執行分析", type="primary")

if run_btn:
    with st.spinner(f"AI 正在分析 {ticker_input}..."):
        df_raw, real_sym = get_stooq_data(ticker_input)
        if df_raw is not None and not df_raw.empty:
            df = calculate_indicators(df_raw)
            df = generate_signals(df, buy_thresh, sell_thresh)
            
            if app_mode == "🔮 市場掃描 (含AI預測)": render_market_scan(df, real_sym)
            elif app_mode == "🦅 策略工廠": render_strategy_lab(df, real_sym)
            elif app_mode == "⚙️ 回測實驗室":
                final, ret, logs, df_c = run_advanced_backtest(
                    df, 100000, bp['start'], bp['end'],
                    bp['mode'], bp['opt_strat'], bp.get('width', 5.0),
                    bp['size'], bp['amt'], bp['iv']
                )
                
                st.header(f"回測報告: {real_sym}")
                c1,c2,c3 = st.columns(3)
                c1.metric("最終資產", f"${final:,.0f}", f"{ret:+.2f}%")
                wr = 0
                if not logs.empty:
                    cls = logs.dropna(subset=['盈虧'])
                    if len(cls)>0: wr = (len(cls[cls['盈虧']>0])/len(cls))*100
                c3.metric("勝率", f"{wr:.1f}%", f"共 {len(logs)} 筆")
                
                st.subheader("資產走勢")
                fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.6,0.4])
                fig.add_trace(go.Scatter(x=df_c.index,y=df_c['Equity'],fill='tozeroy',line=dict(color='#00ff00'),name='資產'),row=1,col=1)
                fig.add_trace(go.Scatter(x=df_c.index,y=df_c['J'],line=dict(color='magenta'),name='J線'),row=2,col=1)
                fig.add_hline(y=20,line_dash="dot",row=2,col=1,line_color="green")
                fig.add_hline(y=80,line_dash="dot",row=2,col=1,line_color="red")
                fig.update_layout(height=600,template="plotly_dark",showlegend=False)
                st.plotly_chart(fig,use_container_width=True)
                
                def safe_fmt(val, pattern):
                    if pd.isna(val) or val is None: return "-"
                    try: return pattern.format(val)
                    except: return str(val)

                if not logs.empty:
                    st.dataframe(
                        logs.style.format({
                            "投入": lambda x: safe_fmt(x, "{:,.0f}"),
                            "進場價": lambda x: safe_fmt(x, "{:.2f}"),
                            "出場價": lambda x: safe_fmt(x, "{:.2f}"),
                            "盈虧": lambda x: safe_fmt(x, "{:+.0f}"),
                            "回報%": lambda x: safe_fmt(x, "{:+.2f}%")
                        }), 
                        use_container_width=True
                    )
        else:
            st.error("無法下載數據")
else:
    st.info("👈 請點擊「執行分析」")
