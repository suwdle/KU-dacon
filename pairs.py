import numpy as np
import pandas as pd
import tqdm
from scipy.stats import spearmanr


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0

    try:
        c = np.corrcoef(x, y)[0, 1]
        return 0.0 if np.isnan(c) else c
    except Exception:
        return 0.0


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 2 or y.size < 2:
        return 0.0

    try:
        c, _ = spearmanr(x, y)
        return 0.0 if np.isnan(c) else c
    except Exception:
        return 0.0


def diff_corr(x, y, lag):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)
    if len(dx) <= lag or len(dy) <= lag:
        return 0.0
    return safe_corr(dx[:-lag], dy[lag:])


def cross_corr(x, y, lag):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) <= lag:
        return 0.0
    return safe_spearman(x[:-lag], y[lag:])


def find_comovement_pairs(
    pivot,
    max_lag=6,
    min_nonzero=10,
    corr_threshold=0.35,
):
    items = pivot.index.tolist()
    n_months = pivot.shape[1]
    results = []

    for i, leader in tqdm.tqdm(enumerate(items), total=len(items)):
        x = pivot.loc[leader].values.astype(float)
        if np.count_nonzero(x) < min_nonzero:
            continue

        for follower in items:
            if follower == leader:
                continue

            y = pivot.loc[follower].values.astype(float)
            if np.count_nonzero(y) < min_nonzero:
                continue

            best_lag = None
            best_score = 0.0
            best_p = 0.0

            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue

                # 1) Pearson
                p = safe_corr(x[:-lag], y[lag:])

                # 2) cross-corr
                cc = cross_corr(x, y, lag)

                # 3) diff corr
                dc = diff_corr(x, y, lag)

                score = (
                    0.70 * abs(p)
                    + 0.20 * abs(cc)
                    + 0.10 * abs(dc)
                )

                if score > best_score:
                    best_score = score
                    best_lag = lag
                    best_p = p

            if best_lag is not None and best_score >= corr_threshold:
                results.append({
                    "leading_item_id": leader,
                    "following_item_id": follower,
                    "best_lag": float(best_lag),
                    "max_corr": best_score,  
                    "pearson": best_p,        
                })

    return pd.DataFrame(results)
