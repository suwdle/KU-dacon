import numpy as np
import pandas as pd
import tqdm
from lightgbm import LGBMRegressor

def train(train_df, model=None):
    """
    log1p 변환을 적용한 LightGBM 회귀 학습
    """

    feature_cols = [
        # follower 시계열
        "b_t", "b_t_1", "b_t_2",

        # leader lagged series
        "a_lag", "a_lag_1", "a_lag_2",

        # follower rolling
        "b_mean3", "b_std3",

        # 공행성
        "max_corr", "best_lag",

        # leader rolling
        "a_mean3", "a_std3",

        # momentum
        "mom_b_1", "mom_b_2",
        "mom_a_1", "mom_a_2",

        # ratio features
        "ratio", "ratio_diff", "ratio_lag2",

        # interaction
        "interaction2",
    ]

    # 기본 모델 세팅
    if model is None:
        model = LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            boosting_type="gbdt",
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=700,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=-1,
            min_data_in_leaf=20,
            lambda_l2=1.0,
            random_state=42,
        )

    # feature matrix
    X = train_df[feature_cols].values

    # y를 log1p로 변환
    y = np.log1p(train_df["target"].values.astype(float))

    model.fit(X, y)
    return model

def predict(pivot, pairs, model):
    months = pivot.columns.to_list()
    n_months = len(months)

    t_last = n_months - 1
    t_prev = n_months - 2
    t_prev2 = n_months - 3

    preds = []

    for row in tqdm.tqdm(pairs.itertuples(index=False)):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot.index or follower not in pivot.index:
            continue

        a = pivot.loc[leader].values.astype(float)
        b = pivot.loc[follower].values.astype(float)

        if t_last - 2 < 0:
            continue
        if t_last - lag - 2 < 0:
            continue

        b_t   = b[t_last]
        b_t_1 = b[t_prev]
        b_t_2 = b[t_prev2]

        a_lag   = a[t_last - lag]
        a_lag_1 = a[t_last - lag - 1]
        a_lag_2 = a[t_last - lag - 2]

        # follower rolling
        start_b = max(0, t_last - 3)
        window_b = b[start_b:t_last]
        b_mean3 = float(np.mean(window_b)) if len(window_b) > 0 else b_t
        b_std3  = float(np.std(window_b)) if len(window_b) > 0 else 0.0

        # leader rolling
        if (t_last - lag) >= 3:
            window_a = a[t_last - lag - 3 : t_last - lag]
            a_mean3 = float(np.mean(window_a))
            a_std3 = float(np.std(window_a))
        else:
            a_mean3 = float(a_lag)
            a_std3 = 0.0

        # momentum
        mom_b_1 = b_t - b_t_1
        mom_b_2 = b_t - b_t_2
        mom_a_1 = a_lag - a_lag_1
        mom_a_2 = a_lag - a_lag_2

        # ratio
        ratio = b_t / (a_lag + 1e-6)
        ratio_prev = b_t_1 / (a_lag_1 + 1e-6)
        ratio_diff = ratio - ratio_prev
        ratio_lag2 = b_t_2 / (a_lag_2 + 1e-6)

        # interaction
        interaction2 = (b_t - b_t_1) * (a_lag - a_lag_1)

        X_test = np.array([[
            b_t, b_t_1, b_t_2,
            a_lag, a_lag_1, a_lag_2,
            b_mean3, b_std3,
            corr, float(lag),
            a_mean3, a_std3,
            mom_b_1, mom_b_2,
            mom_a_1, mom_a_2,
            ratio, ratio_diff, ratio_lag2,
            interaction2
        ]])

        # log 스케일 예측 → expm1 역변환
        log_pred = model.predict(X_test)[0]
        y_pred = np.expm1(log_pred)

        # post-process
        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
        })

    return pd.DataFrame(preds)