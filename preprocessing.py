import pandas as pd
import numpy as np
import tqdm
from typing import Iterable, List, Tuple

def preprocessing(data_path):
    train = pd.read_csv(data_path)

    # year, month, item_id 기준으로 value 합산 (seq만 다르다면 value 합산)
    monthly = (
        train
        .groupby(["item_id", "year", "month"], as_index=False)["value"]
        .sum()
    )

    # year, month를 하나의 키(ym)로 묶기
    monthly["ym"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )

    # item_id × ym 피벗 (월별 총 무역량 매트릭스 생성)
    pivot = (
        monthly
        .pivot(index="item_id", columns="ym", values="value")
        .fillna(0.0)
    )
    return pivot

def build_training_data(pivot, pairs, max_extra_lags=12):
    """
    기존 피처 + 더 긴 시계열 라그 + 다중 윈도우 통계 추가
    max_extra_lags = 추가로 가져올 과거 시점 수 (예: 12 → b[t-3]~b[t-12])
    """
    months = pivot.columns.to_list()
    n_months = len(months)
    rows = []

    for row in pairs.itertuples(index=False):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot.index or follower not in pivot.index:
            continue

        a = pivot.loc[leader].values.astype(float)
        b = pivot.loc[follower].values.astype(float)

        # t는 target이 존재하는 범위에서 생성
        for t in range(max(lag + max_extra_lags + 1, 15), n_months - 1):

            feature = {}

            # ─────────────────────────────────────────
            # 1) FOLLOWER raw lags (b_t, b_t-1, ..., b_t-k)
            # ─────────────────────────────────────────
            feature["b_t"] = b[t]
            for k in range(1, max_extra_lags + 1):
                feature[f"b_t_{k}"] = b[t - k]

            # ─────────────────────────────────────────
            # 2) LEADER raw lags (a_lag, a_lag-1, ..., a_lag-k)
            # ─────────────────────────────────────────
            feature["a_lag"] = a[t - lag]
            for k in range(1, max_extra_lags + 1):
                feature[f"a_lag_{k}"] = a[t - lag - k]

            # ─────────────────────────────────────────
            # 3) Rolling windows
            # follower windows: 3, 6, 12
            # ─────────────────────────────────────────
            for w in [3, 6, 12]:
                feature[f"b_mean_w{w}"] = np.mean(b[t-w:t]) if t >= w else np.mean(b[:t])
                feature[f"b_std_w{w}"]  = np.std(b[t-w:t])  if t >= w else np.std(b[:t])

            # ─────────────────────────────────────────
            # 4) Leader rolling windows
            # ─────────────────────────────────────────
            for w in [3, 6, 12]:
                if (t - lag) >= w:
                    segment = a[t-lag-w : t-lag]
                    feature[f"a_mean_w{w}"] = np.mean(segment)
                    feature[f"a_std_w{w}"]  = np.std(segment)
                else:
                    feature[f"a_mean_w{w}"] = float(a[t-lag])
                    feature[f"a_std_w{w}"]  = 0.0

            # ─────────────────────────────────────────
            # 5) Multi-horizon momentum
            # follower
            # ─────────────────────────────────────────
            feature["mom_b_1"] = b[t] - b[t-1]
            feature["mom_b_3"] = b[t] - b[t-3]
            feature["mom_b_6"] = b[t] - b[t-6]

            # leader
            feature["mom_a_1"] = a[t-lag] - a[t-lag-1]
            feature["mom_a_3"] = a[t-lag] - a[t-lag-3]
            feature["mom_a_6"] = a[t-lag] - a[t-lag-6]

            # ─────────────────────────────────────────
            # 6) Ratio-based features
            # ─────────────────────────────────────────
            feature["ratio"] = b[t] / (a[t-lag] + 1e-6)
            feature["ratio_1"] = b[t-1] / (a[t-lag-1] + 1e-6)
            feature["ratio_3"] = b[t-3] / (a[t-lag-3] + 1e-6)
            feature["ratio_6"] = b[t-6] / (a[t-lag-6] + 1e-6)

            # change in ratio
            feature["ratio_diff"] = feature["ratio"] - feature["ratio_1"]

            # ─────────────────────────────────────────
            # 7) Interaction
            # ─────────────────────────────────────────
            feature["interaction"] = feature["mom_b_1"] * feature["mom_a_1"]
            feature["interaction6"] = feature["mom_b_6"] * feature["mom_a_6"]

            # ─────────────────────────────────────────
            # 공행성 정보 그대로 유지
            # ─────────────────────────────────────────
            feature["max_corr"] = corr
            feature["best_lag"] = float(lag)

            # ─────────────────────────────────────────
            # 타겟
            # ─────────────────────────────────────────
            target = b[t + 1]
            feature["target"] = target

            rows.append(feature)

    return pd.DataFrame(rows)
