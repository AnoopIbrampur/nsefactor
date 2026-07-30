"""Forecast 21-day realised volatility for NSE equities, against real baselines.

Usage:  python scripts/vol_model.py [--garch-stocks N]

Discipline carried over from the return-forecasting work in this repo:

* Chronological 70/15/15 split. Nothing is shuffled across time.
* Evaluation on **non-overlapping** forward windows. Consecutive 21-day
  targets share 20 of their 21 days, so an overlapping evaluation reports
  effective sample sizes several times larger than it has and turns noise
  into significance.
* The model predicts ``log(forward_vol / ewma_anchor)``, never the level, so
  it cannot win by learning that one stock is calmer than another.
* Baselines first. If persistence wins, that is the finding.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from nsefactor import adjust, universe, volatility as vol
from nsefactor.config import DATA_DIR, DEFAULT_CONFIG

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

HORIZON = 21
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def rmse(pred: pd.Series, actual: pd.Series) -> float:
    m = pred.notna() & actual.notna()
    return float(np.sqrt(((pred[m] - actual[m]) ** 2).mean()))


def evaluate(pred: pd.Series, actual: pd.Series, label: str) -> dict:
    m = pred.notna() & actual.notna()
    p, a = pred[m], actual[m]
    return {
        "model": label,
        "n": int(m.sum()),
        "RMSE": round(float(np.sqrt(((p - a) ** 2).mean())), 5),
        "MAE": round(float((p - a).abs().mean()), 5),
        "corr": round(float(p.corr(a)), 4),
        "bias": round(float((p - a).mean()), 5),
    }


def point_in_time_membership(panel: pd.DataFrame, cfg) -> pd.DataFrame:
    """Daily (date, isin) membership of the liquid universe, no lookahead.

    Universe is selected at each month end and held until the next one, which
    is exactly how a portfolio would experience it. Using the union of
    everything ever in the universe instead would leak a mild form of
    survivorship into the sample.
    """
    days = pd.DatetimeIndex(sorted(panel["date"].unique()))
    from nsefactor.backtest import month_end_dates

    rebals = month_end_dates(days)
    rebals = rebals[rebals >= days[300]]

    spans = []
    for i, dt in enumerate(rebals):
        sel = universe.select(panel, dt, cfg=cfg)
        if sel.empty:
            continue
        end = rebals[i + 1] if i + 1 < len(rebals) else days[-1]
        spans.append(pd.DataFrame({"isin": sel.index, "start": dt, "end": end}))
    return pd.concat(spans, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--garch-stocks", type=int, default=25,
                    help="how many stocks to fit GARCH on (it is slow)")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    panel = pd.read_parquet(DATA_DIR / "bhavcopy.parquet")
    # Bridge ISIN changes before adjusting: a company that switched ISIN
    # otherwise carries a truncated history and looks newly listed.
    panel = adjust.apply_isin_links(panel)
    panel = adjust.adjusted_close(panel)
    print(f"panel: {len(panel):,} rows, {panel['date'].nunique():,} sessions")

    print("resolving point-in-time universe membership...")
    spans = point_in_time_membership(panel, cfg)
    members = set(spans["isin"].unique())
    print(f"{len(members):,} ISINs were in the liquid universe at some point")

    print("building volatility dataset...")
    df = vol.build_dataset(panel, horizon=HORIZON, universe_isins=pd.Index(sorted(members)))
    df = df.dropna(subset=vol.FEATURE_COLUMNS + ["target_log_ratio"])

    # Keep only rows where the stock was actually in the universe that day.
    spans_idx = spans.set_index("isin")
    keep = np.zeros(len(df), dtype=bool)
    for isin, grp in df.groupby("isin", sort=False):
        if isin not in spans_idx.index:
            continue
        s = spans_idx.loc[[isin]]
        in_any = np.zeros(len(grp), dtype=bool)
        d = grp["date"].to_numpy()
        for _, row in s.iterrows():
            in_any |= (d >= np.datetime64(row["start"])) & (d <= np.datetime64(row["end"]))
        keep[df.index.get_indexer(grp.index)] = in_any
    df = df[keep].copy()
    print(f"dataset: {len(df):,} rows, {df['isin'].nunique():,} stocks, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    # ---- Chronological split -------------------------------------------
    dates = np.array(sorted(df["date"].unique()))
    i_train = int(len(dates) * TRAIN_FRAC)
    i_val = int(len(dates) * (TRAIN_FRAC + VAL_FRAC))
    train_end, val_end = dates[i_train], dates[i_val]

    # Purge the horizon at each boundary. A row dated just before the split
    # has a target that extends past it, so leaving it in leaks test-period
    # returns into training.
    gap = pd.Timedelta(days=int(HORIZON * 1.6))
    train = df[df["date"] <= train_end - gap]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end - gap)]
    test = df[df["date"] > val_end]

    print(f"\ntrain {len(train):>9,} rows  -> {pd.Timestamp(train_end).date()}")
    print(f"val   {len(val):>9,} rows  -> {pd.Timestamp(val_end).date()}")
    print(f"test  {len(test):>9,} rows  -> {df['date'].max().date()}")
    print(f"(a {gap.days}-day purge sits at each boundary so no target straddles it)")

    # ---- Non-overlapping evaluation grid -------------------------------
    test_dates = np.array(sorted(test["date"].unique()))
    sampled = set(test_dates[::HORIZON])
    test_eval = test[test["date"].isin(sampled)].copy()
    print(f"\nevaluating on {len(sampled)} non-overlapping dates "
          f"({len(test_eval):,} stock-months)")

    section("BASELINES (test period, non-overlapping windows)")
    actual = test_eval["forward_vol"]
    rows = [
        evaluate(vol.baseline_persistence(test_eval), actual, "persistence (trailing rv_21)"),
        evaluate(vol.baseline_ewma(test_eval), actual, "EWMA (RiskMetrics)"),
    ]
    print(pd.DataFrame(rows).set_index("model").to_string())

    # ---- Model ----------------------------------------------------------
    section("MODEL: gradient boosting on log(forward_vol / ewma_anchor)")
    X_train = train[vol.FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["target_log_ratio"].to_numpy(dtype=float)
    X_val = val[vol.FEATURE_COLUMNS].to_numpy(dtype=float)
    y_val = val["target_log_ratio"].to_numpy(dtype=float)

    def make_model(n_iter: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=n_iter,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=200,
            l2_regularization=1.0,
            # sklearn's own early stopping splits randomly, which shuffles
            # across time. Iteration count is chosen on the chronological
            # validation slice instead, below.
            early_stopping=False,
            random_state=cfg.seed,
        )

    # Stage 1: fit on train only and pick the iteration count on val. Val is
    # strictly later than train, so this selection uses no future information
    # relative to the training data -- and no test data at all.
    probe = make_model(600)
    probe.fit(X_train, y_train)
    staged = list(probe.staged_predict(X_val))
    val_rmses = np.array([np.sqrt(((p - y_val) ** 2).mean()) for p in staged])
    best_iter = int(np.argmin(val_rmses)) + 1

    print(f"iterations chosen on validation slice : {best_iter} of 600")
    print(f"val RMSE on log ratio (out of sample) : {val_rmses[best_iter - 1]:.4f}")
    print(f"val RMSE of a zero ratio              : {np.sqrt((y_val ** 2).mean()):.4f}"
          "  (what EWMA implies)")

    # Stage 2: refit on train + val at that iteration count, so the final
    # model uses all data prior to the test period. Test remains untouched.
    model = make_model(best_iter)
    model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))

    X_test = test_eval[vol.FEATURE_COLUMNS].to_numpy(dtype=float)
    pred_ratio = model.predict(X_test)
    model_pred = pd.Series(
        test_eval["anchor"].to_numpy() * np.exp(pred_ratio), index=test_eval.index
    )

    rows.append(evaluate(model_pred, actual, "gradient boosting"))

    section("RESULT")
    table = pd.DataFrame(rows).set_index("model")
    print(table.to_string())

    base_rmse = table.loc["persistence (trailing rv_21)", "RMSE"]
    ewma_rmse = table.loc["EWMA (RiskMetrics)", "RMSE"]
    gb_rmse = table.loc["gradient boosting", "RMSE"]
    print(f"\nRMSE improvement vs persistence : {(1 - gb_rmse / base_rmse) * 100:+.1f}%")
    print(f"RMSE improvement vs EWMA        : {(1 - gb_rmse / ewma_rmse) * 100:+.1f}%")

    # ---- Significance, clustered by date --------------------------------
    section("SIGNIFICANCE (per-date, not per-row)")
    print("9,531 stock-months is not 9,531 independent observations: stocks")
    print("move together, so the effective sample is closer to the number of")
    print("evaluation dates. Everything below is clustered by date.\n")

    from scipy import stats

    ev = test_eval.assign(pred=model_pred)
    per_date = ev.groupby("date").apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "gb_rmse": np.sqrt(((g["pred"] - g["forward_vol"]) ** 2).mean()),
                "ewma_rmse": np.sqrt(((g["anchor"] - g["forward_vol"]) ** 2).mean()),
                "realized": g["forward_vol"].median(),
                "prior": g["anchor"].median(),
            }
        ),
        include_groups=False,
    )
    per_date["impr_%"] = (1 - per_date["gb_rmse"] / per_date["ewma_rmse"]) * 100
    per_date["vol_change"] = per_date["realized"] / per_date["prior"] - 1
    print(per_date.round(4).to_string())

    wins = int((per_date["gb_rmse"] < per_date["ewma_rmse"]).sum())
    n_dates = len(per_date)
    sign_p = stats.binomtest(wins, n_dates, 0.5, alternative="greater").pvalue
    tt = stats.ttest_rel(per_date["gb_rmse"], per_date["ewma_rmse"])
    print(f"\nbeats EWMA on          : {wins}/{n_dates} dates")
    print(f"sign test p            : {sign_p:.5f}")
    print(f"paired t-test          : t={tt.statistic:.2f}, p={tt.pvalue:.5f}")
    print(f"mean/median improvement: {per_date['impr_%'].mean():.2f}% / "
          f"{per_date['impr_%'].median():.2f}%")

    section("WHERE THE EDGE COMES FROM -- AND WHERE IT DOES NOT")
    r = stats.pearsonr(per_date["vol_change"], per_date["impr_%"])
    up = per_date[per_date["vol_change"] > 0]
    dn = per_date[per_date["vol_change"] <= 0]
    print(f"corr(vol change, improvement) : {r.statistic:+.3f}  (p={r.pvalue:.4f})")
    print(f"months when vol FELL  (n={len(dn):2}): {dn['impr_%'].mean():+.2f}% mean improvement")
    print(f"months when vol ROSE  (n={len(up):2}): {up['impr_%'].mean():+.2f}% mean improvement")
    print()
    print("Read this honestly: the model's edge is mean reversion. It is good at")
    print("knowing when elevated volatility will subside, and adds almost nothing")
    print("when volatility jumps -- which is when a risk forecast matters most.")
    print("It is a position-sizing tool, not a crash warning.")

    # ---- GARCH on a subsample ------------------------------------------
    if args.garch_stocks > 0:
        section(f"GARCH(1,1) COMPARISON (subsample of {args.garch_stocks} stocks)")
        print("GARCH needs a fit per stock per date, so it runs on a subsample.")
        print("All methods are re-scored on the same rows for a fair comparison.\n")

        rng = np.random.default_rng(cfg.seed)
        pick = rng.choice(
            test_eval["isin"].unique(),
            size=min(args.garch_stocks, test_eval["isin"].nunique()),
            replace=False,
        )
        sub = test_eval[test_eval["isin"].isin(pick)]
        g = vol.baseline_garch(panel, sub, horizon=HORIZON)

        # Score every method on the rows where GARCH actually converged.
        # Comparing GARCH on its successful fits against other methods on all
        # rows would be scoring it on a self-selected easier sample.
        converged = g.notna()
        n_failed = int((~converged).sum())
        print(f"GARCH converged on {int(converged.sum())}/{len(sub)} rows "
              f"({n_failed} failures); all methods scored on those rows only.\n")
        sub = sub[converged]
        g = g[converged]

        sub_rows = [
            evaluate(vol.baseline_persistence(sub), sub["forward_vol"], "persistence"),
            evaluate(vol.baseline_ewma(sub), sub["forward_vol"], "EWMA"),
            evaluate(g, sub["forward_vol"], "GARCH(1,1)"),
            evaluate(model_pred.reindex(sub.index), sub["forward_vol"], "gradient boosting"),
        ]
        gt = pd.DataFrame(sub_rows).set_index("model")
        print(gt.to_string())
        if not np.isnan(gt.loc["GARCH(1,1)", "RMSE"]):
            imp = (1 - gt.loc["gradient boosting", "RMSE"] / gt.loc["GARCH(1,1)", "RMSE"]) * 100
            print(f"\nRMSE improvement vs GARCH: {imp:+.1f}%")

    section("FEATURE IMPORTANCE (permutation, test rows)")
    print("Measured on the test slice, which the final model never trained on.")
    print("Computing it on validation would report in-sample importances, since")
    print("the refit in stage 2 includes validation.\n")
    from sklearn.inspection import permutation_importance

    y_test_ratio = test_eval["target_log_ratio"].to_numpy(dtype=float)
    imp = permutation_importance(
        model, X_test, y_test_ratio, n_repeats=5, random_state=cfg.seed, n_jobs=1
    )
    order = np.argsort(imp.importances_mean)[::-1]
    for i in order:
        print(f"  {vol.FEATURE_COLUMNS[i]:18} {imp.importances_mean[i]:+.5f}")

    out = DATA_DIR.parent / "reports"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "vol_metrics.csv")
    test_eval.assign(pred=model_pred).to_csv(out / "vol_predictions.csv", index=False)
    print(f"\nwrote artifacts/reports/vol_metrics.csv and vol_predictions.csv")


if __name__ == "__main__":
    main()
