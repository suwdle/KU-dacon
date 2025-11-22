import numpy as np
import pandas as pd
import tqdm
from lightgbm import LGBMRegressor

def train(train_df, model=None):
    """
    log1p 변환을 적용한 LightGBM 회귀 학습
    """
    feature_cols = [c for c in train_df.columns if c not in ["target"]]
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

def predict(pivot, pairs, model, max_extra_lags=12):
    """
    build_training_data(max_extra_lags=12)와 완전 동일한 피처를 생성하는 predict 함수
    """

    months = pivot.columns.to_list()
    n_months = len(months)

    t_last = n_months - 1   # 가장 최근 달
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

        # 필요한 최소 인덱스 체크
        if t_last - max_extra_lags - lag < 0:
            # 시계열 길이가 부족하면 skip
            continue

        feature = {}

        # ─────────────────────────────────────────
        # 1) FOLLOWER lags (b_t, b_t-1, ..., b_t-k)
        # ─────────────────────────────────────────
        feature["b_t"] = b[t_last]
        for k in range(1, max_extra_lags + 1):
            feature[f"b_t_{k}"] = b[t_last - k]

        # ─────────────────────────────────────────
        # 2) LEADER lags (a_lag, a_lag-1, ..., a_lag-k)
        # ─────────────────────────────────────────
        feature["a_lag"] = a[t_last - lag]
        for k in range(1, max_extra_lags + 1):
            feature[f"a_lag_{k}"] = a[t_last - lag - k]

        # ─────────────────────────────────────────
        # 3) Rolling windows follower (3, 6, 12)
        # ─────────────────────────────────────────
        for w in [3, 6, 12]:
            if t_last >= w:
                segment = b[t_last - w : t_last]
                feature[f"b_mean_w{w}"] = np.mean(segment)
                feature[f"b_std_w{w}"]  = np.std(segment)
            else:
                feature[f"b_mean_w{w}"] = float(b[t_last])
                feature[f"b_std_w{w}"]  = 0.0

        # ─────────────────────────────────────────
        # 4) Rolling windows leader
        # ─────────────────────────────────────────
        for w in [3, 6, 12]:
            idx = t_last - lag
            if idx >= w:
                segment = a[idx - w : idx]
                feature[f"a_mean_w{w}"] = np.mean(segment)
                feature[f"a_std_w{w}"]  = np.std(segment)
            else:
                feature[f"a_mean_w{w}"] = float(a[idx])
                feature[f"a_std_w{w}"]  = 0.0

        # ─────────────────────────────────────────
        # 5) Multi-horizon momentum
        # follower
        # ─────────────────────────────────────────
        feature["mom_b_1"] = b[t_last] - b[t_last - 1]
        feature["mom_b_3"] = b[t_last] - b[t_last - 3]
        feature["mom_b_6"] = b[t_last] - b[t_last - 6]

        # leader momentum
        feature["mom_a_1"] = a[t_last - lag] - a[t_last - lag - 1]
        feature["mom_a_3"] = a[t_last - lag] - a[t_last - lag - 3]
        feature["mom_a_6"] = a[t_last - lag] - a[t_last - lag - 6]

        # ─────────────────────────────────────────
        # 6) Ratio features
        # ─────────────────────────────────────────
        feature["ratio"]   = b[t_last]   / (a[t_last - lag]       + 1e-6)
        feature["ratio_1"] = b[t_last-1] / (a[t_last - lag - 1]   + 1e-6)
        feature["ratio_3"] = b[t_last-3] / (a[t_last - lag - 3]   + 1e-6)
        feature["ratio_6"] = b[t_last-6] / (a[t_last - lag - 6]   + 1e-6)

        feature["ratio_diff"] = feature["ratio"] - feature["ratio_1"]

        # ─────────────────────────────────────────
        # 7) Interaction features
        # ─────────────────────────────────────────
        feature["interaction"]  = feature["mom_b_1"] * feature["mom_a_1"]
        feature["interaction6"] = feature["mom_b_6"] * feature["mom_a_6"]

        # ─────────────────────────────────────────
        # 8) 공행성 점수
        # ─────────────────────────────────────────
        feature["max_corr"] = corr
        feature["best_lag"] = float(lag)

        # ─────────────────────────────────────────
        # 9) DataFrame으로 변환 (model 입력)
        # ─────────────────────────────────────────
        X = pd.DataFrame([feature])

        # ─────────────────────────────────────────
        # 10) log1p 예측 → expm1 역변환
        # ─────────────────────────────────────────
        log_pred = model.predict(X)[0]
        y_pred = np.expm1(log_pred)

        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
        })

    return pd.DataFrame(preds)