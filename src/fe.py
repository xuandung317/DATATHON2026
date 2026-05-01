import pandas as pd
import numpy as np
from lunardate import LunarDate


# =============== 0. FULL TIMELINE ===============

def make_full_timeline(sales, TEST_START, TEST_END):

    test_dates = pd.date_range(TEST_START, TEST_END, freq='D')
    test_shell = pd.DataFrame({'Date': test_dates, 'Revenue': np.nan, 'COGS': np.nan})

    full = pd.concat([sales, test_shell], ignore_index=True)
    full = full.sort_values('Date').reset_index(drop=True)

    print(f'Full timeline: {full.Date.min().date()} -> {full.Date.max().date()}')
    print(f'Train rows: {full["Revenue"].notna().sum():,}')
    print(f'Test  rows: {full["Revenue"].isna().sum():,}')

    return full

# =============== 1. CALENDAR FEATURES ===============

def make_calendar_features(df):
    df = df.copy()
    d = df['Date']

    # Thời gian cơ bản
    df['day_of_week']  = d.dt.dayofweek
    df['day_of_month'] = d.dt.day
    df['day_of_year']  = d.dt.dayofyear
    df['week_of_year'] = d.dt.isocalendar().week.astype(int)
    df['month']        = d.dt.month
    df['year']         = d.dt.year
    df['quarter']      = d.dt.quarter

    # Các cờ theo lịch (cuối tuần, cuối tháng, payday,...)
    df['is_weekend']     = (d.dt.dayofweek >= 5).astype(int)
    df['is_month_end']   = d.dt.is_month_end.astype(int)
    df['is_quarter_end'] = d.dt.is_quarter_end.astype(int)
    df['is_payday']      = d.dt.day.isin([28,29,30,31,1]).astype(int)
    df['is_double_day']  = (d.dt.day == d.dt.month).astype(int)
    
    # Ngày lễ - EDA insight: Doanh thu cao trong khoảng cuối các tháng 3, 4, 5
    holidays = {(1,1),(4,30),(5,1),(9,2)}
    df['is_holiday'] = d.apply(lambda x: int((x.month, x.day) in holidays))

    # ==================== LỊCH ÂM (tính động) ====================
    def to_lunar(date):
        ld = LunarDate.fromSolarDate(date.year, date.month, date.day)
        return ld.month, ld.day

    lunar = d.apply(to_lunar)
    df['lunar_month'] = lunar.apply(lambda x: x[0])
    df['lunar_day']   = lunar.apply(lambda x: x[1])

    # Tết: ngày 1 tháng 1 âm lịch
    df['is_tet_day'] = ((df['lunar_month'] == 1) & (df['lunar_day'] == 1)).astype(int)

    # Khoảng cách tới Tết (tính từ ngày mùng 1 tháng 1 âm lịch của năm gần nhất)
    def get_tet_solar(year):
        """Lấy ngày dương lịch của mùng 1 Tết năm âm lịch tương ứng"""
        return LunarDate(year, 1, 1).toSolarDate()

    def days_to_nearest_tet(date):
        # Kiểm tra Tết năm hiện tại và năm sau
        for y in [date.year - 1, date.year, date.year + 1]:
            try:
                tet = pd.Timestamp(get_tet_solar(y))
                diff = (tet - date).days
                if -15 <= diff <= 30:
                    return int(diff)
            except Exception:
                continue
        return 999

    df['days_to_tet']   = d.apply(days_to_nearest_tet)
    df['is_tet_window'] = (df['days_to_tet'] != 999).astype(int)
    df['days_to_tet']   = df['days_to_tet'].clip(-15, 30).where(df['is_tet_window'] == 1, 0)

    # Fourier features
    for k in [1, 2, 3]:
        df[f'sin_year_{k}'] = np.sin(2*np.pi*k * df['day_of_year'] / 365.25)
        df[f'cos_year_{k}'] = np.cos(2*np.pi*k * df['day_of_year'] / 365.25)
        df[f'sin_week_{k}'] = np.sin(2*np.pi*k * df['day_of_week'] / 7)
        df[f'cos_week_{k}'] = np.cos(2*np.pi*k * df['day_of_week'] / 7)

    # Linear trend (xu hướng dài hạn)
    df['trend_days'] = (d - pd.Timestamp('2012-01-01')).dt.days

    print("- Calendar features: OK!")
    
    return df


# =============== 2. LAG FEATURES ===============

def make_lag_features(full, TEST_START):

    # Nguồn lag: chỉ dùng train data (Revenue != NaN)
    lag_source = full[full['Revenue'].notna()][['Date','Revenue','COGS']].copy()

    def add_lag(df, source, lag_days, col_map):
        """Merge lag an toàn bằng Date shift."""
        tmp = source[['Date'] + list(col_map.keys())].copy()
        tmp['Date'] = tmp['Date'] + pd.Timedelta(days=lag_days)
        return df.merge(tmp.rename(columns=col_map), on='Date', how='left')

    # Lag 728 ngày (2×364, đúng 2 chu kỳ 52 tuần)
    full = add_lag(full, lag_source, 728, {
        'Revenue': 'Revenue_lag728',
        'COGS':    'COGS_lag728'
    })

    # Lag 735 ngày (728+7, để tính momentum lag)
    full = add_lag(full, lag_source, 735, {
        'Revenue': 'Revenue_lag735',
        'COGS':    'COGS_lag735'
    })

    # Lag 2 năm calendar (DateOffset)
    tmp_2y = lag_source.copy()
    tmp_2y['Date'] = tmp_2y['Date'] + pd.DateOffset(years=2)
    tmp_2y = tmp_2y.rename(columns={'Revenue':'Revenue_lag2y','COGS':'COGS_lag2y'})
    full = full.merge(tmp_2y, on='Date', how='left')

    # Fallback nếu NaN (năm nhuận)
    full['Revenue_lag2y'] = full['Revenue_lag2y'].fillna(full['Revenue_lag728'])
    full['COGS_lag2y']    = full['COGS_lag2y'].fillna(full['COGS_lag728'])

    # Momentum (YoY change lag)
    full['Revenue_mom_lag'] = full['Revenue_lag728'] - full['Revenue_lag735']
    full['COGS_mom_lag']    = full['COGS_lag728']    - full['COGS_lag735']

    # YoY ratio
    full['Revenue_yoy_ratio'] = full['Revenue_lag728'] / (full['Revenue_lag2y'] + 1e-6)
    full['COGS_yoy_ratio']    = full['COGS_lag728']    / (full['COGS_lag2y']    + 1e-6)

    # Rolling trên lag728 (tính trên dãy đã shift -> safe)
    for target, lag_col in [('Revenue','Revenue_lag728'), ('COGS','COGS_lag728')]:
        full[f'{target}_roll7_lag2y']  = full[lag_col].rolling(7,  min_periods=1).mean()
        full[f'{target}_roll30_lag2y'] = full[lag_col].rolling(30, min_periods=1).mean()
        full[f'{target}_roll90_lag2y'] = full[lag_col].rolling(90, min_periods=1).mean()
        full[f'{target}_global_trend_lag'] = full[lag_col].rolling(364).mean()
        full[f'{target}_std30_lag2y']  = full[lag_col].rolling(30, min_periods=2).std()

    full['rolling_mean_7']  = full['Revenue'].shift(1).rolling(7).mean()
    full['rolling_mean_30'] = full['Revenue'].shift(1).rolling(30).mean()
    full['rolling_mean_90'] = full['Revenue'].shift(1).rolling(90).mean()
    full['rolling_trend']   = full['rolling_mean_30'].diff()
    full['rolling_std_30']  = full['Revenue'].shift(1).rolling(30).std()

    # Tỉ lệ thay đổi
    full['lag_ratio'] = full['Revenue_lag728'] / (full['rolling_mean_30'] + 1e-6)

    # Đánh dấu giai đoạn sau 2018
    full['post_2018'] = (full['Date'] >= '2019-01-01').astype(int)
    full['lag728_x_post2018'] = full['Revenue_lag728'] * full['post_2018']

    # test period không có NaN trong lag features
    test_mask = full['Date'] >= TEST_START
    for col in ['Revenue_lag728','Revenue_lag2y','Revenue_roll7_lag2y']:
        nan_pct = full.loc[test_mask, col].isna().mean()
        print(f'  {col}: {nan_pct:.1%} NaN trong test period')

    print("- Lag features: OK!")

    return full


# =============== 3. PROMOTION FEATURES ===============

def make_promo_features(full, promotions):

    # Expand promotions thành dữ liệu theo ngày
    promo_days = []
    for _, row in promotions.iterrows():
        for d in pd.date_range(row['start_date'], row['end_date']):
            promo_days.append({
                'Date': d,
                'discount_value': row['discount_value'],
                'is_pct_promo':   int(row['promo_type']=='percentage'),
                'is_fixed_promo': int(row['promo_type']=='fixed'),
                'stackable_flag': row['stackable_flag'],
                'has_min_order':  int(pd.notna(row.get('min_order_value', np.nan))),
            })

    promo_df = pd.DataFrame(promo_days)

    # Aggregate theo ngày
    pct_agg   = promo_df[promo_df['is_pct_promo']==1].groupby('Date')['discount_value'].max().rename('max_pct_discount')
    fixed_agg = promo_df[promo_df['is_fixed_promo']==1].groupby('Date')['discount_value'].sum().rename('total_fixed_disc')

    daily_promo = promo_df.groupby('Date').agg(
        n_promos      = ('discount_value','count'),
        has_stackable = ('stackable_flag','max'),
        has_min_order = ('has_min_order','max'),
        n_pct_promos  = ('is_pct_promo','sum'),
        n_fixed_promos= ('is_fixed_promo','sum'),
    ).join(pct_agg).join(fixed_agg).fillna(0).reset_index()

    daily_promo['is_promo_day'] = 1

    # Merge vào dataset chính
    full = full.merge(daily_promo, on='Date', how='left')

    # Fill NA cho ngày không có promotion
    for col in ['is_promo_day','n_promos','has_stackable','n_pct_promos','n_fixed_promos']:
        full[col] = full[col].fillna(0).astype(int)

    for col in ['max_pct_discount','total_fixed_disc']:
        full[col] = full[col].fillna(0)

    # Tháng cao điểm khuyến mãi
    full['is_peak_promo_month'] = full['month'].isin([7,9,11,12]).astype(int)

    # Promo lag
    full['n_promos_lag7']  = full['n_promos'].shift(7).fillna(0)    # promo tuần trước
    full['n_promos_lag14'] = full['n_promos'].shift(14).fillna(0)   # promo 2 tuần trước

    print("- Promotion features: OK!")
    
    return full


# =============== 4. WEB TRAFFIC FEATURES ===============

def make_web_features(full, web, TEST_START):

    # Aggregate web traffic theo ngày
    web_daily = web.groupby('date').agg(
        sessions        = ('sessions','sum'),
        page_views      = ('page_views','sum'),
        # weighted avg
        bounce_rate_avg = ('bounce_rate', lambda x: np.average(x, weights=web.loc[x.index, 'sessions'])),
        avg_session_dur = ('avg_session_duration_sec', lambda x: np.average(x, weights=web.loc[x.index, 'sessions'])),
    ).reset_index().rename(columns={'date':'Date'})

    # Hành vi người dùng
    web_daily['pages_per_session'] = (
        web_daily['page_views'] / (web_daily['sessions'] + 1))
    web_daily['engagement_score'] = (
        (1 - web_daily['bounce_rate_avg']) * web_daily['avg_session_dur'])

    # Lag
    web_cols = ['sessions', 'pages_per_session']
    tmp_web = web_daily[['Date'] + web_cols].copy()
    tmp_web['Date'] = tmp_web['Date'] + pd.Timedelta(days=728)

    tmp_web = tmp_web.rename(columns={
        'sessions': 'sessions_lag728',
        'pages_per_session': 'pps_lag728'
    })

    full = full.merge(tmp_web, on='Date', how='left')

    # Chất lượng traffic (volume + engagement)
    full['traffic_quality_lag'] = (
        np.log1p(full['sessions_lag728']) *
        np.log1p(full['pps_lag728'])
    )

    # Kiểm tra coverage
    cov = full.loc[full['Date']>=TEST_START, 'sessions_lag728'].notna().mean()

    print("- Web traffic features: OK!")
    print(f'  Web traffic coverage trong test period: {cov:.1%}')
    
    return full


# =============== 5. ORDERS FEATURES ===============

def make_order_features(full, orders):

    # Aggregate đơn hàng theo ngày
    daily_orders = orders.groupby('order_date').agg(
        total_orders  = ('order_id','count'),
        n_cancelled   = ('order_status', lambda x: (x=='cancelled').sum()),
        mobile_ratio  = ('device_type',  lambda x: (x=='mobile').mean()),
    ).reset_index().rename(columns={'order_date':'Date'})

    # Tỉ lệ hủy đơn
    daily_orders['cancel_rate'] = daily_orders['n_cancelled'] / (daily_orders['total_orders'] + 1)

    # Lag 728 ngày
    order_cols = ['total_orders','cancel_rate','mobile_ratio']
    tmp_ord = daily_orders[['Date'] + order_cols].copy()
    tmp_ord['Date'] = tmp_ord['Date'] + pd.Timedelta(days=728)

    tmp_ord = tmp_ord.rename(columns={c: f'orders_{c}_lag728' for c in order_cols})
    full = full.merge(tmp_ord, on='Date', how='left')

    # Rolling trend của số đơn (làm mượt + bắt trend)
    full['orders_roll30_lag2y'] = full['orders_total_orders_lag728'].rolling(30, min_periods=1).mean()

    print("- Orders features: OK!")
    
    return full


# =============== 6. STRUCTURAL PERIOD FEATURES ===============

def make_structural_period_features(full):
    """
    Căn cứ từ EDA offline (EDA.ipynb) — tất cả tính trên GROSS REVENUE:
    -----------------------------------------------------------------------
    [1] STL Decomposition (Revenue/gross, period=365, robust=True):
        - Đỉnh trend: 03/2016 (~5.4M/ngày)
        - Giảm dốc liên tục từ 2019, đáy: 04/2021 (~2.9M/ngày)
        - Baseline 2013-2018 avg ~4.5-4.8M/ngày
        - Giai đoạn 2019-2021 avg ~3.x M/ngày -> giảm ~30% so với baseline
        -> WEAKNESS_ADJ = 1/0.70

    [2] YoY Gross Revenue (sales_after_2020, chart xu hướng tăng trưởng):
        - 2021 vs 2020: -1.1% -> quá nhỏ, không adjust riêng
        - 2022 vs 2021: +12.1% -> REBOUND_ADJ = 1/1.121
        - CAGR 2020-2022: +5.32%/năm -> rebound nhẹ, không phải boom

    [3] Inventory analysis:
        - 2019-2021: STR + stockout đều thấp -> nhu cầu suy yếu toàn diện
        - Vòng quay hàng tồn kho đạt đỉnh tiêu cực năm 2020 (337 ngày)

    Lý do chọn adj factor:
    -----------------------------------------------------------------------
    - WEAKNESS_ADJ = 1/0.70:
        Trend gross revenue giảm từ ~5.4M (đỉnh 2016) xuống ~2.9M (đáy 2021).
        So với baseline 2013-2018, giai đoạn 2019-2021 thấp hơn ~30%.
        lag_728 trỏ về giai đoạn này sẽ underestimate target hiện tại
        -> nhân 1/0.70 để bù trừ.

    - REBOUND_ADJ = 1/1.121:
        YoY Gross Revenue 2022 vs 2021: +12.1% (EDA confirmed).
        lag_728 trỏ về 2022 sẽ overestimate target -> nhân 1/1.121 để bù trừ.
    -----------------------------------------------------------------------
    """

    d = full['Date']

    # Suy yếu cấu trúc: trend giảm dốc từ 2019, đáy 04/2021 [STL - ref 1]
    full['is_structural_decline'] = ((d >= '2019-01-01') & (d <= '2021-12-31')).astype(int)

    # Hồi phục nhẹ: YoY +12.1%, CAGR +5.32% [YoY chart - ref 2]
    full['is_mild_rebound']       = ((d >= '2022-01-01') & (d <= '2022-12-31')).astype(int)

    # lag_728 trỏ về giai đoạn bất thường -> giá trị lag bị lệch
    lag_d = d - pd.Timedelta(days=728)
    full['lag_weakness_period'] = ((lag_d >= '2019-01-01') & (lag_d <= '2021-12-31')).astype(int)
    full['lag_rebound_period']  = ((lag_d >= '2022-01-01') & (lag_d <= '2022-12-31')).astype(int)

    # WEAKNESS_ADJ = 1/0.70: gross revenue 2019-2021 thấp hơn baseline ~30% [ref 1, 3]
    WEAKNESS_ADJ = 1 / 0.70

    # REBOUND_ADJ = 1/1.121: YoY gross revenue 2022 = +12.1% [ref 2]
    REBOUND_ADJ  = 1 / 1.121

    for col in ['Revenue_lag728', 'COGS_lag728']:
        full[col] = np.where(full['lag_weakness_period'] == 1, full[col] * WEAKNESS_ADJ, full[col])
        full[col] = np.where(full['lag_rebound_period']  == 1, full[col] * REBOUND_ADJ,  full[col])

    print(f"- Structural period features: OK! ")

    return full

# =============== 7. COMBINED FEATURES ===============

def make_combined_features(full):
    # Độ lệch giữa trend ngắn hạn và cùng kỳ 2 năm trước
    # Chuẩn hoá theo độ biến động (std) để phát hiện spike/giảm bất thường
    full['trend_zscore'] = (
        (full['rolling_mean_7'] - full['Revenue_lag728']) /
        (full['rolling_std_30'] + 1e-6)
    )

    print("- Combined features: OK!")
    return full

# =============== 8. NAN FIX ===============

def fix_nan(X_train, X_test):

    # Fill has_min_order
    for df in [X_train, X_test]:
        if 'has_min_order' in df.columns:
            df['has_min_order'] = df['has_min_order'].fillna(0).astype(int)

    web_lag_cols = [
        c for c in X_train.columns
        if '_lag728' in c and 'Revenue' not in c and 'COGS' not in c and 'orders' not in c]

    web_roll_cols = [
        c for c in X_train.columns
        if ('sessions' in c or 'visitors' in c or 'page_views' in c
            or 'bounce' in c or 'engagement' in c or 'pages_per' in c)
        and 'roll' in c]

    # Fill missing cho web features
    for df in [X_train, X_test]:
        # Forward fill tối đa 7 ngày (điền missing cục bộ)
        df[web_lag_cols + web_roll_cols] = (
            df[web_lag_cols + web_roll_cols]
            .ffill(limit=7)
            .fillna(0)          # còn lại = 0 (không có data)
        )

    print("- NaN fix: OK!")

    return X_train, X_test

# =============== MAIN PIPELINE ===============

def build_features(full, promotions, web, orders, TEST_START, TEST_END):
    full = make_full_timeline(full, TEST_START, TEST_END)
    full = make_calendar_features(full)
    full = make_lag_features(full, TEST_START)
    full = make_promo_features(full, promotions)
    full = make_web_features(full, web, TEST_START)
    full = make_order_features(full, orders)
    full = make_structural_period_features(full)
    full = make_combined_features(full)
    return full