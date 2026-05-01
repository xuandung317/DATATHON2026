import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shap

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# ─── Internal ───────────────────────────────────────────────
import os as _os
OUTPUTS_DIR = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), '..', 'outputs')
)
_os.makedirs(OUTPUTS_DIR, exist_ok=True)

from config import (
    SEED, DATA_PATHS, OUT_PATH,
    TRAIN_END, TEST_START, TEST_END, CUTOFF_DATE,
    CV_N_SPLITS, CV_GAP, CV_TEST_SIZE,
    NON_FEATURE_COLS,
    LGB_PARAMS, XGB_PARAMS,
)
from fe import build_features, fix_nan

np.random.seed(SEED)


# ════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════
def load_data():
    sales      = pd.read_csv(DATA_PATHS['sales'],      parse_dates=['Date'])
    promotions = pd.read_csv(DATA_PATHS['promotions'], parse_dates=['start_date', 'end_date'])
    web        = pd.read_csv(DATA_PATHS['web'],        parse_dates=['date'])
    orders     = pd.read_csv(DATA_PATHS['orders'],     parse_dates=['order_date'])
    print('Data loaded!')
    print(f'  Train: {sales.Date.min().date()} -> {TRAIN_END.date()}')
    print(f'  Test : {TEST_START.date()} -> {TEST_END.date()}')
    return sales, promotions, web, orders


# ════════════════════════════════════════════════════════════
# 2. PREPARE TRAIN / TEST SPLITS
# ════════════════════════════════════════════════════════════
def prepare_splits(full):
    # Bỏ 2 năm đầu cold-start
    full = full[full['Date'] >= CUTOFF_DATE].reset_index(drop=True)

    feature_cols = [c for c in full.columns if c not in NON_FEATURE_COLS]

    train_df = full[full['Date'] <= TRAIN_END].copy()
    test_df  = full[full['Date'] >= TEST_START].copy()

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test  = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    y_train_rev  = np.log1p(train_df['Revenue'])       # log1p để ổn định variance
    y_train_cogs = np.log1p(train_df['COGS'])

    print(f'X_train: {X_train.shape}')
    print(f'X_test : {X_test.shape}')
    print(f'Features: {len(feature_cols)}')

    # NaN report
    nan_report = X_train.isna().mean()
    high_nan   = nan_report[nan_report > 0.05]
    if len(high_nan):
        print('\nFeatures có NaN > 5% trong train:')
        print(high_nan)
    else:
        print('\nKhông có feature nào có NaN > 5% trong train')

    return train_df, test_df, X_train, X_test, y_train_rev, y_train_cogs, feature_cols

def check_nan_after_fix(X_train, X_test):
    # Kiểm tra NaN > 5%
    nan_after = X_train.isna().mean()
    remaining = nan_after[nan_after > 0.05]

    print('NaN > 5% sau khi fix:')
    if len(remaining) == 0:
        print('Chỉ còn lag Revenue/COGS đầu chuỗi — Moddel tự xử lý')
    else:
        print(remaining)

    # Breakdown NaN theo nhóm loại
    rev_lag_cols = [
        c for c in X_train.columns
        if 'lag' in c and ('Revenue' in c or 'COGS' in c or 'orders' in c)
    ]

    nan_rev = X_train[rev_lag_cols].isna().mean().mean()
  

# ════════════════════════════════════════════════════════════
# 3. SAMPLE WEIGHTS
# ════════════════════════════════════════════════════════════
def get_sample_weights(dates):
    w = np.ones(len(dates))

    # Structural decline — khớp với make_structural_period_features
    structural_decline_mask = ((dates >= '2019-01-01') & (dates <= '2021-12-31')).values
    w[structural_decline_mask] = 0.4    # rev thấp hơn baseline ~30%

    # Recency weight
    days_to_end    = (dates.max() - dates).dt.days
    recency_weight = 1 / (1 + days_to_end / 365)

    return w * recency_weight


# ════════════════════════════════════════════════════════════
# 4. CROSS-VALIDATION
# ════════════════════════════════════════════════════════════
def run_cv(model, X, y_log, dates, tscv, name = ''):
    scores     = []
    fold_preds = np.zeros(len(X))

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
        dates_tr    = dates.iloc[tr_idx]
        
        w_tr        = get_sample_weights(dates_tr)

        model.fit(X_tr, y_tr, sample_weight=w_tr)

        y_pred_log  = model.predict(X_val)
        y_val_orig  = np.expm1(y_val)
        y_pred_orig = np.expm1(y_pred_log)
        fold_preds[val_idx] = y_pred_orig

        mae  = mean_absolute_error(y_val_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
        r2   = r2_score(y_val_orig, y_pred_orig)
        scores.append({'fold': fold + 1, 'MAE': mae, 'RMSE': rmse, 'R2': r2})

        d_val = dates.iloc[val_idx]
        print(f'  [{name}] Fold {fold+1} [{d_val.min().date()}->{d_val.max().date()}]: '
              f'MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.3f}')
        print(f'  Mean Pred: {y_pred_orig.mean():,.0f}, Mean True: {y_val_orig.mean():,.0f}')

    df = pd.DataFrame(scores)
    print(f'  {"─"*65}')
    print(f'  Mean: MAE={df.MAE.mean():,.0f}  '
          f'RMSE={df.RMSE.mean():,.0f}  R2={df.R2.mean():.3f}')
    return df, fold_preds


# ════════════════════════════════════════════════════════════
# 5. RETRAIN + PREDICT
# ════════════════════════════════════════════════════════════
def retrain_and_predict(model_rev, model_cogs, X_train, y_train_rev,
                        y_train_cogs, X_test, train_df):
    all_weights = get_sample_weights(train_df['Date'])

    model_rev.fit(X_train,  y_train_rev,  sample_weight=all_weights)
    model_cogs.fit(X_train, y_train_cogs, sample_weight=all_weights)
    print('Retrain on full train: OK!')

    # Predict
    revenue_pred = np.expm1(model_rev.predict(X_test))
    cogs_pred    = np.expm1(model_cogs.predict(X_test))
    revenue_pred = np.maximum(revenue_pred, 0)
    cogs_pred    = np.maximum(cogs_pred, 0)
    
    # Sanity check
    last_rev  = train_df['Revenue'].iloc[-60:].mean()
    last_cogs = train_df['COGS'].iloc[-60:].mean()
    pred_rev  = revenue_pred[:30].mean()
    pred_cogs = cogs_pred[:30].mean()
    print(f'\nSanity check (30d đầu prediction vs 60d cuối train):')
    print(f'  Revenue: train={last_rev:,.0f} | pred={pred_rev:,.0f} (ratio={pred_rev/last_rev:.2f})')
    print(f'  COGS   : train={last_cogs:,.0f} | pred={pred_cogs:,.0f} (ratio={pred_cogs/last_cogs:.2f})')

    return revenue_pred, cogs_pred


# ════════════════════════════════════════════════════════════
# 6. PLOT
# ════════════════════════════════════════════════════════════
def plot_forecast(train_df, test_df, revenue_pred, cogs_pred):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)

    # Revenue
    ax = axes[0]
    train_weekly = train_df.set_index('Date')['Revenue'].resample('W').sum()
    test_dates   = test_df['Date'].values
    test_weekly_dates = pd.date_range(TEST_START, TEST_END, freq='W')
    test_weekly_rev   = pd.Series(revenue_pred, index=test_df['Date']).resample('W').sum()

    ax.plot(train_weekly.index, train_weekly.values/1e6, color='steelblue', linewidth=1, label='Train (actual)')
    ax.plot(test_weekly_rev.index, test_weekly_rev.values/1e6, color='crimson', linewidth=2,
            linestyle='--', label='Prediction (2023–2024)')
    ax.axvline(TRAIN_END, color='gray', linestyle=':', linewidth=1.5)
    ax.set_title('Revenue Forecast: 2023-01-01 -> 2024-07-01 (Weekly sum)', fontweight='bold')
    ax.set_ylabel('Revenue (Triệu VND)'); ax.legend()
    ax.grid(alpha=0.3)

    # COGS
    ax = axes[1]
    train_cogs_weekly = train_df.set_index('Date')['COGS'].resample('W').sum()
    test_weekly_cogs  = pd.Series(cogs_pred, index=test_df['Date']).resample('W').sum()

    ax.plot(train_cogs_weekly.index, train_cogs_weekly.values/1e6, color='steelblue', linewidth=1, label='Train (actual)')
    ax.plot(test_weekly_cogs.index, test_weekly_cogs.values/1e6, color='darkorange', linewidth=2,
            linestyle='--', label='Prediction (2023–2024)')
    ax.axvline(TRAIN_END, color='gray', linestyle=':', linewidth=1.5)
    ax.set_title('COGS Forecast: 2023-01-01 -> 2024-07-01 (Weekly sum)', fontweight='bold')
    ax.set_ylabel('COGS (Triệu VND)'); ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Sales & COGS Forecasting', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/forecast_output.png', bbox_inches='tight', dpi=130)
    plt.show()


# ════════════════════════════════════════════════════════════
# 7. SHAP
# ════════════════════════════════════════════════════════════
def explain_model(model_rev, X_train):
    color_primary   = '#2a9d8f'
    color_secondary = '#264653'

    explainer   = shap.TreeExplainer(model_rev)
    sample_idx  = np.random.choice(len(X_train), min(1000, len(X_train)), replace=False)
    X_sample    = X_train.iloc[sample_idx]
    shap_values = explainer.shap_values(X_sample)

    # ---- Plot 1: Bar ----
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    plt.sca(ax1)
    shap.summary_plot(shap_values, X_sample, plot_type='bar', max_display=20,
                      show=False, color=color_primary)
    ax1.set_title('Feature Importance (Top 20)', fontweight='bold',
                  color=color_secondary, fontsize=13, pad=12)
    ax1.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    plt.close(fig1)

    # ---- Plot 2: Beeswarm ----
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    plt.sca(ax2)
    shap.summary_plot(shap_values, X_sample, max_display=15,
                      show=False, cmap='RdYlGn')
    ax2.set_title('Feature Impact Distribution (Top 15)', fontweight='bold',
                  color=color_secondary, fontsize=13, pad=12)
    ax2.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    plt.close(fig2)

    fig1.savefig(f'{OUTPUTS_DIR}/_shap_bar.png', bbox_inches='tight', dpi=200)
    fig2.savefig(f'{OUTPUTS_DIR}/_shap_beeswarm.png', bbox_inches='tight', dpi=200)

    # ---- Combine ----
    img1 = mpimg.imread(f'{OUTPUTS_DIR}/_shap_bar.png')
    img2 = mpimg.imread(f'{OUTPUTS_DIR}/_shap_beeswarm.png')

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle('SHAP Analysis — Revenue Model', fontsize=20, fontweight='bold',
                 color=color_secondary, y=1.02)

    axes[0].imshow(img1)
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/shap_revenue.png', bbox_inches='tight', dpi=200)
    plt.show()
    print('Saved: shap_revenue.png')

    # ---- Top features ----
    feature_importance = pd.DataFrame({
        'feature'        : X_train.columns,
        'shap_importance': np.abs(shap_values).mean(0)
    }).sort_values('shap_importance', ascending=False)

    print('\nTop 10 features quan trọng nhất (Revenue):')
    print(feature_importance.head(10).to_string(index=False))

    return feature_importance


# ════════════════════════════════════════════════════════════
# 8. SAVE SUBMISSION
# ════════════════════════════════════════════════════════════
def save_submission(test_df, revenue_pred, cogs_pred):
    submission = pd.DataFrame({
        'Date'   : test_df['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': np.round(revenue_pred, 2),
        'COGS'   : np.round(cogs_pred, 2),
    })
    print('\nSubmission preview:')
    print(submission.head(10).to_string(index=False))
    print(f'Total rows     : {len(submission)}')
    print(f'Revenue range  : [{submission.Revenue.min():.0f}, {submission.Revenue.max():.0f}]')
    print(f'COGS range     : [{submission.COGS.min():.0f}, {submission.COGS.max():.0f}]')
    print(f'Negative Rev   : {(submission.Revenue < 0).sum()}')
    print(f'Negative COGS  : {(submission.COGS < 0).sum()}')

    submission.to_csv(OUT_PATH, index=False)
    print('\nSaved: submission.csv')