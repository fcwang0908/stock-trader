# ==========================================
# 老陳 AI 交易系統 V23.1 - 緊急修復版
# 修復內容：
# 1. 解決 KeyError: 'Signal' (補上訊號生成步驟)
# 2. 確保回測模式下可以讀取到 Signal 欄位
# 3. 優化側邊欄：買賣門檻 (J線) 現在對所有模式生效
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

# 頁面設定
st.set_page_config(page_title="老陳 V23.1 (修復版)", layout="wide", page_icon="🏦")

# --- 0. 全局設定與數據 ---
PRESETS = {
    "自行輸入": "MHI",
    "🏙️ 收租三寶": {"823 領展": "823", "5 匯豐": "5", "941 中移動": "941"},
    "🚀 科技龍頭": {"700 騰訊": "700", "9988 阿里": "9988", "3690 美團": "3690"},
    "🇺🇸 美股 ETF": {"QQQ 納指": "QQQ", "SPY 標普": "SPY", "TLT 美債": "TLT", "NVDA": "NVDA"}
}

# --- 1. 核心函數庫 ---

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

# --- 2. 各分頁功能模組 ---

def render_market_scan(df, real_sym):
    st.header(f"📊 報價與資金流: {real_sym}")
    last = df.iloc[-1]
    prev = df.iloc[-2]
    change = last['Close'] - prev['Close']
    pct = (change / prev['Close']) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新價", f"{last['Close']:,.2f}", f"{change:+.2f} ({pct:.2f}%)")
    c2.metric("MFI 資金流", f"{last['MFI']:.1f}", delta_color="off")
    c3.metric("J 線", f"{last['J']:.1f}", delta_color="off")
    vol_ratio = last['Volume'] / df['Volume'].rolling(20).mean().iloc[-1]
    c4.metric("量比", f"x{vol_ratio:.1f}")

    st.markdown("---")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.subheader("🧐 趨勢診斷")
        if last['Close'] > last['MA60']: st.success("🐂 牛市區域 (>MA60)")
        else: st.error("🐻 熊市區域 (<MA60)")
        if last['J'] < 20: st.warning("⚡ J線超賣")
        elif last['J'] > 80: st.warning("⚡ J線超買")
        else: st.info("⚖️ J線中性")
    with status_col2:
        st.subheader("💰 資金流向")
        if last['MFI'] > 80: st.error("🔥 資金超買 (出貨)")
        elif last['MFI'] < 20: st.success("🟢 資金超賣 (吸籌)")
        else: st.info("🌊 資金平穩")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='blue', width=1), name='MA60'), row=1, col=1)
    colors = ['green' if c>=o else 'red' for c,o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], line=dict(color='cyan'), name='MFI'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['J'], line=dict(color='magenta'), name='J線'), row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", row=3, col=1, line_color="green")
    fig.add_hline(y=80, line_dash="dot", row=3, col=1, line_color="red")
    fig.update_layout(height=800, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_strategy_lab(df, real_sym):
    st.header(f"🦅 期權策略工廠: {real_sym}")
    last = df.iloc[-1]
    price = last['Close']
    j_val = last['J']
    
    st.info(f"參考價: {price:.2f} | J線: {j_val:.1f}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("方向性建議")
        if j_val < 20:
            st.success("🚀 看升 (Bullish)")
            st.markdown(f"**Bull Put Spread**\n* Sell Put @ {price*0.98:.1f}\n* Buy Put @ {price*0.95:.1f}")
        elif j_val > 80:
            st.error("📉 看跌 (Bearish)")
            st.markdown(f"**Bear Call Spread**\n* Sell Call @ {price*1.02:.1f}\n* Buy Call @ {price*1.05:.1f}")
        else:
            st.warning("⚖️ 震盪")
            st.write("觀望 或 Iron Condor")
    with col2:
        st.subheader("收租建議")
        st.markdown(f"**Iron Condor**\n* Sell Call @ {price*1.05:.1f}\n* Sell Put @ {price*0.95:.1f}")

# --- 3. 回測引擎 (V22 核心) ---
def run_full_backtest(df, initial_capital, start_date, end_date, mode_str, size_type, fixed_amt, iv_param=0.3):
    # 這裡的 df 必須已經包含 'Signal' 欄位
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    df_test = df.loc[mask].copy()
    if df_test.empty: return 0, 0, pd.DataFrame(), pd.DataFrame()

    capital = initial_capital
    position = 0
    trade_log = []
    equity_curve = []
    entry_idx = 0
    strike_price = 0
    holding_type = None 
    invested_amount = 0 
    r_rate = 0.03
    is_option_mode = ("Options" in mode_str)

    def calc_position_size(price):
        if size_type == "全倉 (All-in)": return capital
        else: return min(capital, fixed_amt)

    for i in range(len(df_test)):
        date = df_test.index[i]
        stock_price = df_test['Close'].iloc[i]
        signal = df_test['Signal'].iloc[i] # 這裡需要 Signal 欄位
        
        current_equity = capital
        if holding_type == 'stock':
            current_equity = (capital - invested_amount) + (position * stock_price)
        elif holding_type in ['call', 'put']:
            days_held = (i - entry_idx)
            days_left = max(0.01, 30 - days_held)
            opt_price = black_scholes_price(stock_price, strike_price, days_left/365, r_rate, iv_param, holding_type)
            current_equity = (capital - invested_amount) + (position * opt_price)
        equity_curve.append(current_equity)

        if signal == 1: # Buy / Close Put
            if holding_type == 'put':
                days_left = max(0.01, 30-(i-entry_idx))
                cash_back = position * black_scholes_price(stock_price, strike_price, days_left/365, r_rate, iv_param, 'put')
                capital = (capital - invested_amount) + cash_back
                profit = cash_back - invested_amount
                trade_log[-1].update({'出場日期': date, '出場價': stock_price, '盈虧': profit, '回報%': (profit/invested_amount)*100})
                position = 0; holding_type = None
            
            if position == 0:
                if not is_option_mode:
                    amt = calc_position_size(stock_price)
                    if amt > 0:
                        position = amt / stock_price
                        invested_amount = amt
                        holding_type = 'stock'
                        trade_log.append({'進場日期': date, '動作': 'Buy Stock', '投入': amt, '進場價': stock_price, '出場日期': None, '盈虧': None})
                else:
                    strike_price = stock_price
                    opt_p = black_scholes_price(stock_price, strike_price, 30/365, r_rate, iv_param, 'call')
                    amt = calc_position_size(opt_p)
                    if amt > 0:
                        position = amt / opt_p
                        invested_amount = amt
                        holding_type = 'call'
                        entry_idx = i
                        trade_log.append({'進場日期': date, '動作': f'Long Call {strike_price:.0f}', '投入': amt, '進場價': stock_price, '出場日期': None, '盈虧': None})

        elif signal == -1: # Sell / Buy Put
            if holding_type in ['stock', 'call']:
                cash_back = 0
                if holding_type == 'stock': cash_back = position * stock_price
                else: 
                    days_left = max(0.01, 30-(i-entry_idx))
                    cash_back = position * black_scholes_price(stock_price, strike_price, days_left/365, r_rate, iv_param, 'call')
                
                capital = (capital - invested_amount) + cash_back
                profit = cash_back - invested_amount
                trade_log[-1].update({'出場日期': date, '出場價': stock_price, '盈虧': profit, '回報%': (profit/invested_amount)*100})
                position = 0; holding_type = None
            
            if is_option_mode and position == 0:
                strike_price = stock_price
                opt_p = black_scholes_price(stock_price, strike_price, 30/365, r_rate, iv_param, 'put')
                amt = calc_position_size(opt_p)
                if amt > 0:
                    position = amt / opt_p
                    invested_amount = amt
                    holding_type = 'put'
                    entry_idx = i
                    trade_log.append({'進場日期': date, '動作': f'Long Put {strike_price:.0f}', '投入': amt, '進場價': stock_price, '出場日期': None, '盈虧': None})

    df_test['Equity'] = equity_curve
    final_val = equity_curve[-1] if equity_curve else initial_capital
    ret = ((final_val - initial_capital) / initial_capital) * 100
    return final_val, ret, pd.DataFrame(trade_log), df_test

# --- 4. 主程式介面 ---

with st.sidebar:
    st.title("🎛️ 戰情室控制台")
    app_mode = st.radio("功能模組", ["📊 市場掃描", "🦅 策略工廠", "⚙️ 回測實驗室"])
    st.divider()
    
    st.subheader("1. 選擇標的")
    cat = st.selectbox("分類", list(PRESETS.keys()))
    if cat == "自行輸入":
        ticker_input = st.text_input("輸入代號", value="MHI").upper()
    else:
        sel = st.selectbox("股票", list(PRESETS[cat].keys()))
        ticker_input = PRESETS[cat][sel]
    
    if st.button("🗑️ 清除快取"): st.cache_data.clear()

    # === 全局參數 (關鍵修正：讓買賣門檻對所有模式生效) ===
    st.divider()
    st.subheader("2. 訊號參數 (J線)")
    buy_thresh = st.slider("買入 (J <)", 0, 40, 20)
    sell_thresh = st.slider("賣出 (J >)", 60, 100, 80)

    # 回測專用參數
    backtest_params = {}
    if app_mode == "⚙️ 回測實驗室":
        st.divider()
        st.subheader("3. 回測設定")
        backtest_params['mode'] = st.radio("交易工具", ["Spot (正股)", "Options (期權)"])
        backtest_params['size'] = st.radio("注碼", ["全倉 (All-in)", "定額"])
        if backtest_params['size'] == "定額":
            backtest_params['amt'] = st.number_input("每次金額", value=20000)
        else:
            backtest_params['amt'] = 0
            
        col_d1, col_d2 = st.columns(2)
        with col_d1: backtest_params['start'] = st.date_input("開始", pd.to_datetime("2023-01-01"))
        with col_d2: backtest_params['end'] = st.date_input("結束", datetime.today())
        
        backtest_params['iv'] = 0.3
        if "Options" in backtest_params['mode']:
            backtest_params['iv'] = st.slider("IV (引伸波幅)", 0.1, 1.0, 0.25)
            
    run_btn = st.button("🚀 執行分析", type="primary")

if run_btn:
    with st.spinner(f"正在分析 {ticker_input}..."):
        df_raw, real_sym = get_stooq_data(ticker_input)
        
        if df_raw is not None and not df_raw.empty:
            df = calculate_indicators(df_raw)
            
            # === ⚠️ 關鍵修復：在進入任何模式前，先產生訊號！ ===
            df = generate_signals(df, buy_thresh, sell_thresh)
            # =================================================
            
            if app_mode == "📊 市場掃描":
                render_market_scan(df, real_sym)
                
            elif app_mode == "🦅 策略工廠":
                render_strategy_lab(df, real_sym)
                
            elif app_mode == "⚙️ 回測實驗室":
                final, ret, logs, df_chart = run_full_backtest(
                    df, 100000, 
                    backtest_params['start'], backtest_params['end'],
                    backtest_params['mode'], backtest_params['size'], backtest_params['amt'],
                    backtest_params.get('iv', 0.3)
                )
                
                st.header(f"回測報告: {real_sym}")
                c1, c2, c3 = st.columns(3)
                c1.metric("最終資產", f"${final:,.0f}", f"{ret:+.2f}%")
                
                win_rate = 0
                if not logs.empty:
                    closed = logs.dropna(subset=['盈虧'])
                    if len(closed) > 0: win_rate = (len(closed[closed['盈虧']>0])/len(closed))*100
                c3.metric("勝率", f"{win_rate:.1f}%", f"共 {len(logs)} 筆")
                
                st.subheader("資產走勢")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Equity'], fill='tozeroy', line=dict(color='#00ff00'), name='資產'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['J'], line=dict(color='#ab63fa'), name='J線'), row=2, col=1)
                fig.add_hline(y=buy_thresh, line_dash="dot", row=2, col=1, line_color="green")
                fig.add_hline(y=sell_thresh, line_dash="dot", row=2, col=1, line_color="red")
                fig.update_layout(height=600, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                if not logs.empty:
                    st.dataframe(logs.style.format({"投入": "{:,.0f}", "進場價": "{:.2f}", "出場價": "{:.2f}", "盈虧": "{:+.0f}", "回報%": "{:+.2f}%"}), use_container_width=True)
        else:
            st.error("無法下載數據")
else:
    st.info("👈 請在左側按下「執行分析」")
