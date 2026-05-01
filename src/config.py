from pathlib import Path
import pandas as pd
 
# ─── Seed ───────────────────────────────────────────────────
SEED = 42
 
# ─── ROOT PATH ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ─── DATA & OUTPUT ──────────────────────────────────────────
DATA_DIR = ROOT / 'data'

OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Đường dẫn dữ liệu ──────────────────────────────────────
DATA_PATHS = {
    'products':   DATA_DIR / 'products.csv',
    'customers':  DATA_DIR / 'customers.csv',
    'geography':  DATA_DIR / 'geography.csv',
    'orders':     DATA_DIR / 'orders.csv',
    'order_items':DATA_DIR / 'order_items.csv',
    'promotions': DATA_DIR / 'promotions.csv',
    'payments':   DATA_DIR / 'payments.csv',
    'reviews':    DATA_DIR / 'reviews.csv',
    'sales':      DATA_DIR / 'sales.csv',
    'inventory':  DATA_DIR / 'inventory.csv',
    'shipments':  DATA_DIR / 'shipments.csv',
    'returns':    DATA_DIR / 'returns.csv',
    'web':        DATA_DIR / 'web_traffic.csv',
}

OUT_PATH = OUTPUT_DIR / 'submission.csv' 

# ─── Khoảng thời gian ───────────────────────────────────────
TEST_START  = pd.Timestamp('2023-01-01')
TEST_END    = pd.Timestamp('2024-07-01')
TRAIN_END   = pd.Timestamp('2022-12-31')
CUTOFF_DATE = '2015-07-04'          # bỏ 2 năm đầu cold-start
 
# ─── Cross-validation ────────────────────────────────────────
CV_N_SPLITS = 5
CV_GAP      = 30
CV_TEST_SIZE= 365

# ─── Các cột không phải feature ─────────────────────────────
NON_FEATURE_COLS = ['Date', 'Revenue', 'COGS', 
                    'Revenue_lag735', 'COGS_lag735']

# ─── Hyperparams LightGBM ────────────────────────────────────
LGB_PARAMS = dict(
    n_estimators     = 1000,
    learning_rate    = 0.05,
    num_leaves       = 31,
    min_child_samples= 50,
    subsample        = 0.7,
    subsample_freq   = 5,
    colsample_bytree = 0.7,
    reg_alpha        = 0.5,
    reg_lambda       = 5.0,
    random_state     = SEED,
    n_jobs           = -1,
    verbose          = -1,
)
 
# ─── Hyperparams XGBoost ─────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators     = 1000,
    learning_rate    = 0.05,
    max_depth        = 6,
    min_child_weight = 10,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    gamma            = 1,                   # Phạt các cây có cấu trúc phức tạp
    reg_alpha        = 0.5,
    reg_lambda       = 20,                  # Tăng mạnh để kéo dự báo về mức trung bình
    objective        = 'reg:absoluteerror',
    random_state     = SEED,
    n_jobs           = -1,
    tree_method      = 'hist',              # Tăng tốc độ huấn luyện
)