import pandas as pd
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
import numpy as np
import pandas as pd

def build_training_data(pivot, pairs):
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

        for t in range(max(lag + 2, 3), n_months - 1):
            b_t   = b[t]
            b_t_1 = b[t-1]
            b_t_2 = b[t-2]

            a_lag   = a[t-lag]
            a_lag_1 = a[t-lag-1]
            a_lag_2 = a[t-lag-2]

            # Rolling stats (follower)
            b_mean3 = np.mean(b[t-3:t]) if t >= 3 else np.mean(b[:t])
            b_std3  = np.std(b[t-3:t])  if t >= 3 else np.std(b[:t])

            # Rolling stats (leader)
            if (t-lag) >= 3:
                a_mean3 = np.mean(a[t-lag-3:t-lag])
                a_std3  = np.std(a[t-lag-3:t-lag])
            else:
                a_mean3 = float(a_lag)
                a_std3  = 0.0

            # Momentum (follower)
            mom_b_1 = b_t - b_t_1
            mom_b_2 = b_t - b_t_2

            # Momentum (leader)
            mom_a_1 = a_lag - a_lag_1
            mom_a_2 = a_lag - a_lag_2

            # Ratio features
            ratio      = b_t / (a_lag + 1e-6)
            ratio_prev = b_t_1 / (a_lag_1 + 1e-6)
            ratio_diff = ratio - ratio_prev
            ratio_lag2 = b_t_2 / (a_lag_2 + 1e-6)

            # Interaction features
            interaction2 = (b_t - b_t_1) * (a_lag - a_lag_1)

            target = b[t+1]

            rows.append({
                # 기존 특징들
                "b_t": b_t, "b_t_1": b_t_1, "b_t_2": b_t_2,
                "a_lag": a_lag, "a_lag_1": a_lag_1, "a_lag_2": a_lag_2,
                "b_mean3": b_mean3, "b_std3": b_std3,
                "max_corr": corr, "best_lag": float(lag),

                # 새 파생 피처들
                "a_mean3": a_mean3, "a_std3": a_std3,
                "mom_b_1": mom_b_1, "mom_b_2": mom_b_2,
                "mom_a_1": mom_a_1, "mom_a_2": mom_a_2,
                "ratio": ratio, "ratio_diff": ratio_diff,
                "ratio_lag2": ratio_lag2,
                "interaction2": interaction2,

                # 타겟
                "target": target
            })

    return pd.DataFrame(rows)
