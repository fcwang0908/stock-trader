# --- 1. 數據獲取 (V19.3 HSI 強制修復版) ---
@st.cache_data(ttl=3600)
def get_stooq_data(symbol):
    clean_sym = symbol.upper().strip()
    
    # === 關鍵修改：強制恆指轉盈富 ===
    # 因為 Stooq 的 ^HSI 經常沒數據或沒成交量，導致策略失效
    # 所以只要用戶查恆指，我們一律抓 2800.HK (走勢一樣，但數據更全)
    if clean_sym in ["HSI", "HSI.HK", "^HSI"]: 
        clean_sym = "2800.HK"
    
    # 智能修正代號 (Stooq 格式)
    elif clean_sym.isdigit(): 
        # 港股補齊 .HK (例如 700 -> 0700.HK)
        # Stooq 對某些港股需要 4 位數還是 5 位數有時不統一
        # 通常 0700.HK 是最穩的
        clean_sym = f"{clean_sym.zfill(4)}.HK"
        
    # Stooq CSV 下載連結
    url = f"https://stooq.com/q/d/l/?s={clean_sym}&i=d"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10) # 加個 timeout 防止卡死
        
        if response.status_code != 200:
            return None, clean_sym
            
        file_content = response.content.decode('utf-8')
        
        # 檢查是不是無效數據
        if "No data" in file_content or len(file_content) < 50:
             return None, clean_sym

        df = pd.read_csv(io.StringIO(file_content))
        
        # === 數據清理 ===
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index() # 轉為舊 -> 新
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        return df, clean_sym

    except Exception as e:
        print(f"Error: {e}")
        return None, clean_sym
