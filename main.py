"""
main.py — Chạy toàn bộ pipeline dự đoán Revenue & COGS
Usage:
    python main.py
    python main.py --output my_submission.csv
    python main.py --no-plot --no-shap
"""

import sys, os, argparse, time
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# =============== Argument parser ===============
def parse_args():
    parser = argparse.ArgumentParser(
        description='Revenue & COGS Forecasting Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --output results/submission_v2.csv
  python main.py --no-plot --no-shap
        """
    )
    parser.add_argument('--output',   type=str, default=None,
                        help='Đường dẫn file output (mặc định: theo config)')
    parser.add_argument('--no-plot',  action='store_true',
                        help='Bỏ qua bước vẽ biểu đồ forecast')
    parser.add_argument('--no-shap',  action='store_true',
                        help='Bỏ qua bước SHAP / feature importance')
    parser.add_argument('--model',    type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'both'],
                        help='Model sử dụng để retrain & predict (default: xgboost)')
    return parser.parse_args()


# =============== Main pipeline ===============
def main():
    args = parse_args()
    t0 = time.time()

    print('=' * 65)
    print('  REVENUE & COGS FORECASTING PIPELINE')
    print('=' * 65)

    # =============== Imports ===============
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.base import clone
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    from config import (
        TEST_START, TEST_END,
        CV_N_SPLITS, CV_GAP, CV_TEST_SIZE,
        LGB_PARAMS, XGB_PARAMS
    )
    from fe import build_features, fix_nan
    from train import (
        load_data, prepare_splits, check_nan_after_fix,
        run_cv, retrain_and_predict, plot_forecast,
        explain_model, save_submission
    )

    # =============== Load data ===============
    print('\nLoad data...')
    sales, promotions, web, orders = load_data()

    # =============== Feature Engineering ===============
    print('\nFeature Engineering...')
    full = build_features(sales, promotions, web, orders, TEST_START, TEST_END)

    # =============== Split ===============
    print('\nPrepare Splits...')
    train_df, test_df, X_train, X_test, y_train_rev, y_train_cogs, feature_cols = \
        prepare_splits(full)

    X_train, X_test = fix_nan(X_train, X_test)
    check_nan_after_fix(X_train, X_test)

    # =============== CV ===============
    print('\nCross-Validation...')
    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS, gap=CV_GAP, test_size=CV_TEST_SIZE)

    print('Kiểm tra các fold:')
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        d_tr  = train_df['Date'].iloc[tr_idx]
        d_val = train_df['Date'].iloc[val_idx]
        print(f'  Fold {fold+1}: train {d_tr.min().date()}->{d_tr.max().date()} '
              f'| val {d_val.min().date()}->{d_val.max().date()} '
              f'({len(val_idx)} ngày)')

    # =============== XGBoost CV ===============
    print('\n' + '=' * 65)
    print('XGBoost — Revenue')
    print('=' * 65)
    xgb_rev  = XGBRegressor(**XGB_PARAMS)
    cv_rev, _ = run_cv(clone(xgb_rev), X_train, y_train_rev,
                       train_df['Date'], tscv, 'Revenue')

    print('\n' + '=' * 65)
    print('XGBoost — COGS')
    print('=' * 65)
    xgb_cogs = XGBRegressor(**XGB_PARAMS)
    cv_cogs, _ = run_cv(clone(xgb_cogs), X_train, y_train_cogs,
                        train_df['Date'], tscv, 'COGS')

    # =============== Retrain + Predict ===============
    print(f'\nRetrain & Predict (model={args.model})...')

    pred_rev_model, pred_cogs_model = xgb_rev, xgb_cogs

    revenue_pred, cogs_pred = retrain_and_predict(
        pred_rev_model, pred_cogs_model,
        X_train, y_train_rev, y_train_cogs,
        X_test, train_df
    )

    # =============== Plot ===============
    if not args.no_plot:
        print('\nPlot forecast...')
        plot_forecast(train_df, test_df, revenue_pred, cogs_pred)
    else:
        print('\nPlot forecast... (skipped)')

    # =============== SHAP ===============
    if not args.no_shap:
        print('\nSHAP / Feature Importance...')
        explain_model(pred_rev_model, X_train)
    else:
        print('\nSHAP... (skipped)')

    # =============== Save submission ===============
    print('\nSaving submission...')
    save_submission(test_df, revenue_pred, cogs_pred)

    elapsed = time.time() - t0
    print(f'\nDone! Total time: {elapsed:.1f}s')
    print('=' * 65)


if __name__ == '__main__':
    main()