"""Produce everything the volatility demo page needs, as one JSON blob.

Usage:  python scripts/vol_demo.py

Runs the same pipeline as scripts/vol_model.py but also emits current
forecasts with readable symbols and the position-sizing they imply, which is
what the forecast is actually for.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from nsefactor import adjust, backtest, portfolio, universe, volatility as vol
from nsefactor.config import DATA_DIR, DEFAULT_CONFIG

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("vol_demo")

CFG = DEFAULT_CONFIG
HORIZON = 21
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def main() -> None:
    panel = pd.read_parquet(DATA_DIR / "bhavcopy.parquet")
    panel = adjust.apply_isin_links(panel)
    panel = adjust.adjusted_close(panel)
    days = pd.DatetimeIndex(sorted(panel["date"].unique()))

    print("resolving universe membership...")
    rebals = backtest.month_end_dates(days)
    rebals = rebals[rebals >= days[300]]
    members: set[str] = set()
    spans = []
    for i, dt in enumerate(rebals):
        sel = universe.select(panel, dt, cfg=CFG)
        if sel.empty:
            continue
        members.update(sel.index)
        end = rebals[i + 1] if i + 1 < len(rebals) else days[-1]
        spans.append(pd.DataFrame({"isin": sel.index, "start": dt, "end": end}))
    spans = pd.concat(spans, ignore_index=True)

    print("building volatility dataset...")
    full = vol.build_dataset(
        panel, horizon=HORIZON,
        universe_isins=pd.Index(sorted(members)),
        require_target=False,
    ).dropna(subset=vol.FEATURE_COLUMNS)

    trainable = full.dropna(subset=["target_log_ratio"])

    dates = np.array(sorted(trainable["date"].unique()))
    i_train = int(len(dates) * TRAIN_FRAC)
    i_val = int(len(dates) * (TRAIN_FRAC + VAL_FRAC))
    train_end, val_end = dates[i_train], dates[i_val]
    gap = pd.Timedelta(days=int(HORIZON * 1.6))

    train = trainable[trainable["date"] <= train_end - gap]
    val = trainable[(trainable["date"] > train_end) & (trainable["date"] <= val_end - gap)]
    test = trainable[trainable["date"] > val_end]

    X_tr = train[vol.FEATURE_COLUMNS].to_numpy(float)
    y_tr = train["target_log_ratio"].to_numpy(float)
    X_va = val[vol.FEATURE_COLUMNS].to_numpy(float)
    y_va = val["target_log_ratio"].to_numpy(float)

    def make(n):
        return HistGradientBoostingRegressor(
            max_iter=n, learning_rate=0.05, max_depth=6, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=False, random_state=CFG.seed)

    print("fitting model...")
    probe = make(600)
    probe.fit(X_tr, y_tr)
    rmses = [np.sqrt(((p - y_va) ** 2).mean()) for p in probe.staged_predict(X_va)]
    best_iter = int(np.argmin(rmses)) + 1
    model = make(best_iter)
    model.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))

    # ---- Test evaluation on non-overlapping windows ---------------------
    tdates = np.array(sorted(test["date"].unique()))
    ev = test[test["date"].isin(set(tdates[::HORIZON]))].copy()
    ev["pred"] = ev["anchor"].to_numpy() * np.exp(model.predict(ev[vol.FEATURE_COLUMNS].to_numpy(float)))

    def rmse(p, a):
        return float(np.sqrt(((p - a) ** 2).mean()))

    actual = ev["forward_vol"]
    metrics_rows = [
        {"model": "persistence", "rmse": rmse(ev["rv_21"], actual),
         "corr": float(ev["rv_21"].corr(actual))},
        {"model": "EWMA (RiskMetrics)", "rmse": rmse(ev["anchor"], actual),
         "corr": float(ev["anchor"].corr(actual))},
        {"model": "gradient boosting", "rmse": rmse(ev["pred"], actual),
         "corr": float(ev["pred"].corr(actual))},
    ]
    gb, ew, pe = metrics_rows[2]["rmse"], metrics_rows[1]["rmse"], metrics_rows[0]["rmse"]

    per_date = ev.groupby("date").apply(
        lambda g: pd.Series({
            "gb": rmse(g["pred"], g["forward_vol"]),
            "ewma": rmse(g["anchor"], g["forward_vol"]),
            "realized": g["forward_vol"].median(),
            "prior": g["anchor"].median(),
            "n": len(g),
        }), include_groups=False)
    per_date["impr"] = (1 - per_date["gb"] / per_date["ewma"]) * 100
    per_date["vol_change"] = per_date["realized"] / per_date["prior"] - 1

    wins = int((per_date["gb"] < per_date["ewma"]).sum())
    sign_p = float(stats.binomtest(wins, len(per_date), 0.5, alternative="greater").pvalue)
    tt = stats.ttest_rel(per_date["gb"], per_date["ewma"])
    r = stats.pearsonr(per_date["vol_change"], per_date["impr"])
    up = per_date[per_date["vol_change"] > 0]
    dn = per_date[per_date["vol_change"] <= 0]

    # ---- Current forecasts, with symbols --------------------------------
    latest_date = full["date"].max()
    cur = full[full["date"] == latest_date].copy()
    cur["pred"] = cur["anchor"].to_numpy() * np.exp(
        model.predict(cur[vol.FEATURE_COLUMNS].to_numpy(float)))

    sel = universe.select(panel, days[-1], cfg=CFG)
    sym = sel["symbol"]
    last_close = sel["last_close"]
    cur = cur[cur["isin"].isin(sel.index)].copy()
    cur["symbol"] = cur["isin"].map(sym)
    cur["price"] = cur["isin"].map(last_close)
    cur["change"] = cur["pred"] / cur["ewma"] - 1.0
    cur = cur.dropna(subset=["symbol"])

    # Position sizing: what the forecast is for.
    top = cur.nlargest(200, "rv_21").set_index("isin")
    chosen = pd.Index(cur.nsmallest(20, "pred")["isin"])
    risk = cur.set_index("isin")["pred"]
    w_iv = portfolio.inverse_vol_weights(chosen, risk, max_weight=0.10)

    forecasts = [
        {
            "symbol": row.symbol,
            "price": round(float(row.price), 2) if pd.notna(row.price) else None,
            "trailing": round(float(row.rv_21) * 100, 1),
            "ewma": round(float(row.ewma) * 100, 1),
            "forecast": round(float(row.pred) * 100, 1),
            "change": round(float(row.change) * 100, 1),
        }
        for row in cur.sort_values("pred").itertuples()
    ]

    # Sizing comparison on a sample book of 20 varied-risk names.
    sample = cur.sort_values("pred").iloc[::max(1, len(cur) // 20)].head(12)
    sizing = []
    eq = 100.0 / len(sample)
    inv = 1.0 / sample["pred"]
    inv = inv / inv.sum() * 100
    for row, w in zip(sample.itertuples(), inv):
        sizing.append({
            "symbol": row.symbol,
            "forecast": round(float(row.pred) * 100, 1),
            "equal": round(eq, 2),
            "risk_weighted": round(float(w), 2),
        })

    out = {
        "as_of": str(pd.Timestamp(latest_date).date()),
        "panel_rows": int(len(panel)),
        "sessions": int(panel["date"].nunique()),
        "stocks_modelled": int(cur["isin"].nunique()),
        "train_rows": int(len(train)),
        "best_iter": best_iter,
        "split": {
            "train_end": str(pd.Timestamp(train_end).date()),
            "val_end": str(pd.Timestamp(val_end).date()),
            "test_end": str(pd.Timestamp(trainable["date"].max()).date()),
        },
        "metrics": [
            {**m, "rmse": round(m["rmse"], 5), "corr": round(m["corr"], 3)}
            for m in metrics_rows
        ],
        "improvement": {
            "vs_persistence": round((1 - gb / pe) * 100, 1),
            "vs_ewma": round((1 - gb / ew) * 100, 1),
        },
        "significance": {
            "wins": wins,
            "months": int(len(per_date)),
            "sign_p": round(sign_p, 5),
            "t_stat": round(float(tt.statistic), 2),
            "t_p": round(float(tt.pvalue), 6),
            "mean_impr": round(float(per_date["impr"].mean()), 2),
        },
        "regime": {
            "corr": round(float(r.statistic), 3),
            "corr_p": round(float(r.pvalue), 4),
            "fell_n": int(len(dn)),
            "fell_impr": round(float(dn["impr"].mean()), 2),
            "rose_n": int(len(up)),
            "rose_impr": round(float(up["impr"].mean()), 2),
        },
        "per_date": [
            {"date": str(pd.Timestamp(d).date()),
             "impr": round(float(row["impr"]), 2),
             "vol_change": round(float(row["vol_change"]) * 100, 1),
             "realized": round(float(row["realized"]) * 100, 1)}
            for d, row in per_date.iterrows()
        ],
        "forecasts": forecasts,
        "sizing": sizing,
    }

    path = DATA_DIR.parent / "reports" / "vol_demo.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    print(f"as of {out['as_of']}, {out['stocks_modelled']} stocks")
    print(f"improvement vs EWMA {out['improvement']['vs_ewma']}%, "
          f"wins {wins}/{len(per_date)}")


if __name__ == "__main__":
    main()
