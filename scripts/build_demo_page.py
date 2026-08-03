"""Render the volatility demo page from artifacts/reports/vol_demo.json."""

from __future__ import annotations

import json
from pathlib import Path

from nsefactor.config import DATA_DIR

DATA = json.loads((DATA_DIR.parent / "reports" / "vol_demo.json").read_text())
OUT = DATA_DIR.parent / "demo" / "index.html"

HTML = """<title>Volatility Forecasting for NSE Equities</title>
<style>
  :root {
    --ground: #F4F5F7;
    --panel: #FFFFFF;
    --ink: #171B21;
    --ink-soft: #59616E;
    --ink-faint: #8A94A6;
    --rule: #DDE1E8;
    --accent: #0B6E75;
    --accent-soft: #E2F0F1;
    --warn: #B45309;
    --warn-soft: #FBEEDC;
    --base: #8A94A6;
    --shadow: 0 1px 2px rgba(20,24,29,.06), 0 8px 24px rgba(20,24,29,.05);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ground: #14181D;
      --panel: #1B2027;
      --ink: #E7EBF0;
      --ink-soft: #A3ADBB;
      --ink-faint: #6F7987;
      --rule: #2A313A;
      --accent: #4FBFC6;
      --accent-soft: #173134;
      --warn: #E0913F;
      --warn-soft: #2B2318;
      --base: #6F7987;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.25);
    }
  }
  :root[data-theme="dark"] {
    --ground: #14181D; --panel: #1B2027; --ink: #E7EBF0; --ink-soft: #A3ADBB;
    --ink-faint: #6F7987; --rule: #2A313A; --accent: #4FBFC6;
    --accent-soft: #173134; --warn: #E0913F; --warn-soft: #2B2318; --base: #6F7987;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.25);
  }
  :root[data-theme="light"] {
    --ground: #F4F5F7; --panel: #FFFFFF; --ink: #171B21; --ink-soft: #59616E;
    --ink-faint: #8A94A6; --rule: #DDE1E8; --accent: #0B6E75;
    --accent-soft: #E2F0F1; --warn: #B45309; --warn-soft: #FBEEDC; --base: #8A94A6;
    --shadow: 0 1px 2px rgba(20,24,29,.06), 0 8px 24px rgba(20,24,29,.05);
  }

  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--ground); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 16px; line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 0 24px 96px; }
  h1, h2, h3 {
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    font-weight: 600; text-wrap: balance; margin: 0;
  }
  h1 { font-size: clamp(2rem, 4.4vw, 3.1rem); line-height: 1.12; letter-spacing: -.015em; }
  h2 { font-size: clamp(1.35rem, 2.4vw, 1.75rem); line-height: 1.25; }
  h3 { font-size: 1.05rem; }
  p { margin: 0; }
  .num { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
         font-variant-numeric: tabular-nums; }
  .eyebrow {
    font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
    color: var(--ink-faint); font-weight: 600;
  }

  header.masthead { padding: 72px 0 40px; display: flex; flex-direction: column; gap: 18px; }
  .lede { font-size: 1.12rem; color: var(--ink-soft); max-width: 60ch; }
  .meta { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: .82rem;
          color: var(--ink-faint); border-top: 1px solid var(--rule); padding-top: 16px; }

  section { padding-top: 56px; display: flex; flex-direction: column; gap: 20px; }
  .prose { max-width: 65ch; color: var(--ink-soft); display: flex;
           flex-direction: column; gap: 14px; }
  .prose strong { color: var(--ink); font-weight: 600; }

  .tiles { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
  .tile { background: var(--panel); border: 1px solid var(--rule); border-radius: 10px;
          padding: 20px; box-shadow: var(--shadow); display: flex;
          flex-direction: column; gap: 6px; }
  .tile .v { font-size: 2.1rem; line-height: 1; font-weight: 600; letter-spacing: -.02em; }
  .tile .k { font-size: .78rem; color: var(--ink-faint); }
  .tile.hero .v { color: var(--accent); }

  .panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 10px;
           box-shadow: var(--shadow); overflow: hidden; }
  .panel-head { padding: 18px 22px; border-bottom: 1px solid var(--rule);
                display: flex; justify-content: space-between; align-items: baseline;
                gap: 16px; flex-wrap: wrap; }
  .panel-body { padding: 22px; }

  .scroller { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: .9rem; }
  th, td { text-align: right; padding: 10px 14px; border-bottom: 1px solid var(--rule);
           white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  thead th { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
             color: var(--ink-faint); font-weight: 600; }
  tbody tr:last-child td { border-bottom: none; }
  tr.win td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
  .best { color: var(--accent); font-weight: 600; }

  .chart { width: 100%; height: auto; display: block; }
  .legend { display: flex; flex-wrap: wrap; gap: 18px; font-size: .8rem;
            color: var(--ink-soft); padding: 0 22px 20px; }
  .swatch { display: inline-block; width: 11px; height: 11px; border-radius: 2px;
            margin-right: 7px; vertical-align: -1px; }

  .caution { background: var(--warn-soft); border: 1px solid var(--warn);
             border-radius: 10px; padding: 24px; display: flex;
             flex-direction: column; gap: 14px; }
  .caution h2 { color: var(--warn); }
  .caution .prose { color: var(--ink); }
  .split { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }
  .stat-inline { background: var(--panel); border: 1px solid var(--rule);
                 border-radius: 8px; padding: 16px; }
  .stat-inline .v { font-size: 1.5rem; font-weight: 600; }

  .controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  input[type="search"], select {
    font: inherit; font-size: .88rem; padding: 8px 12px; color: var(--ink);
    background: var(--ground); border: 1px solid var(--rule); border-radius: 7px;
  }
  input[type="search"]:focus-visible, select:focus-visible, button:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 2px;
  }
  .bar-cell { display: flex; align-items: center; gap: 9px; justify-content: flex-end; }
  .bar { height: 7px; border-radius: 3px; background: var(--accent); opacity: .75; }
  .bar.alt { background: var(--base); }

  footer { margin-top: 72px; padding-top: 24px; border-top: 1px solid var(--rule);
           font-size: .84rem; color: var(--ink-faint); display: flex;
           flex-direction: column; gap: 10px; max-width: 70ch; }
  @media (prefers-reduced-motion: no-preference) {
    .bar { transition: width .4s ease; }
  }
</style>

<div class="wrap">
  <header class="masthead">
    <span class="eyebrow">NSE equities &middot; 21-day realised volatility</span>
    <h1>Predicting how bumpy a stock will be &mdash; not which way it goes</h1>
    <p class="lede">
      Monthly returns in this universe were close enough to noise that a
      stock-ranking model lost to an index fund. Volatility is different: it
      clusters, so it can be forecast. This is what that model does, how well it
      works, and exactly where it stops working.
    </p>
    <div class="meta">
      <span>As of <strong class="num">__AS_OF__</strong></span>
      <span><strong class="num">__STOCKS__</strong> stocks</span>
      <span><strong class="num">__SESSIONS__</strong> trading sessions</span>
      <span><strong class="num">__TRAINROWS__</strong> training rows</span>
    </div>
  </header>

  <section>
    <span class="eyebrow">Headline</span>
    <div class="tiles">
      <div class="tile hero">
        <span class="v num">+__IMPR_EWMA__%</span>
        <span class="k">Lower error than RiskMetrics EWMA, the industry standard</span>
      </div>
      <div class="tile hero">
        <span class="v num">+__IMPR_PERS__%</span>
        <span class="k">Lower error than assuming next month repeats last month</span>
      </div>
      <div class="tile">
        <span class="v num">__WINS__/__MONTHS__</span>
        <span class="k">Test months where it beat EWMA</span>
      </div>
      <div class="tile">
        <span class="v num">__TP__</span>
        <span class="k">Paired t-test p-value across months</span>
      </div>
    </div>
  </section>

  <section>
    <span class="eyebrow">Accuracy against real baselines</span>
    <h2>Every method scored on the same non-overlapping months</h2>
    <div class="prose">
      <p>
        Error is root-mean-square error against realised volatility, so lower is
        better. Consecutive 21-day windows share 20 of their 21 days, so the test
        months are spaced out rather than overlapping &mdash; overlapping windows
        would inflate the apparent sample several times over.
      </p>
    </div>
    <div class="panel">
      <div class="scroller">
        <table>
          <thead>
            <tr><th>Method</th><th>Error (RMSE)</th><th>Correlation with reality</th></tr>
          </thead>
          <tbody>__METRIC_ROWS__</tbody>
        </table>
      </div>
    </div>
  </section>

  <section>
    <span class="eyebrow">Month by month</span>
    <h2>Where the edge comes from &mdash; and where it doesn&rsquo;t</h2>
    <div class="prose">
      <p>
        Each bar is one test month: how much lower the model&rsquo;s error was than
        EWMA&rsquo;s. Bars are coloured by what volatility actually did that month.
        <strong>The pattern is the point.</strong> The model does well when
        volatility was falling and barely helps when it rose.
      </p>
    </div>
    <div class="panel">
      <div class="panel-head">
        <h3>Improvement over EWMA, by month</h3>
        <span class="num" style="color:var(--ink-faint);font-size:.82rem">
          mean +__MEAN_IMPR__%
        </span>
      </div>
      <div class="scroller">__CHART__</div>
      <div class="legend">
        <span><span class="swatch" style="background:var(--accent)"></span>Volatility fell that month</span>
        <span><span class="swatch" style="background:var(--warn)"></span>Volatility rose that month</span>
      </div>
    </div>
  </section>

  <section>
    <div class="caution">
      <span class="eyebrow" style="color:var(--warn)">The limitation, stated plainly</span>
      <h2>It is a position-sizing tool, not a crash warning</h2>
      <div class="prose">
        <p>
          The model&rsquo;s edge is mean reversion &mdash; it is good at knowing when
          elevated volatility will settle back down. It adds very little when
          volatility jumps, which is exactly when a risk forecast would matter most.
        </p>
        <p>
          The correlation between how volatility moved and how much the model helped
          is <strong class="num">__REGIME_CORR__</strong>
          (p = <strong class="num">__REGIME_P__</strong>). A risk tool that quietly
          stops working during turbulence is worse than one that never worked, so
          this sits here rather than in a footnote.
        </p>
      </div>
      <div class="split">
        <div class="stat-inline">
          <div class="v num" style="color:var(--accent)">+__FELL_IMPR__%</div>
          <div class="k eyebrow" style="letter-spacing:.06em">
            when volatility fell &middot; __FELL_N__ months
          </div>
        </div>
        <div class="stat-inline">
          <div class="v num" style="color:var(--warn)">+__ROSE_IMPR__%</div>
          <div class="k eyebrow" style="letter-spacing:.06em">
            when volatility rose &middot; __ROSE_N__ months
          </div>
        </div>
      </div>
    </div>
  </section>

  <section>
    <span class="eyebrow">Live output</span>
    <h2>What the model says today</h2>
    <div class="prose">
      <p>
        Forecast annualised volatility for the next month, for every stock in the
        liquid universe. <strong>Change</strong> is the forecast against the EWMA
        estimate &mdash; negative means the model expects things to calm down.
      </p>
    </div>
    <div class="panel">
      <div class="panel-head">
        <h3>Forecasts as of <span class="num">__AS_OF__</span></h3>
        <div class="controls">
          <input type="search" id="q" placeholder="Filter by symbol…" aria-label="Filter by symbol">
          <select id="sort" aria-label="Sort order">
            <option value="calm">Calmest first</option>
            <option value="wild">Most volatile first</option>
            <option value="drop">Biggest expected drop</option>
            <option value="rise">Biggest expected rise</option>
          </select>
        </div>
      </div>
      <div class="scroller">
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Price</th><th>Trailing 21d</th>
              <th>EWMA</th><th>Forecast</th><th>Change</th>
            </tr>
          </thead>
          <tbody id="fbody"></tbody>
        </table>
      </div>
      <div class="legend"><span id="fcount"></span></div>
    </div>
  </section>

  <section>
    <span class="eyebrow">Why it matters</span>
    <h2>The forecast turns into position sizes</h2>
    <div class="prose">
      <p>
        Equal weighting gives every holding the same rupees, which hands most of a
        portfolio&rsquo;s actual risk to its jumpiest few names. Sizing by
        <strong>1 / forecast volatility</strong> equalises risk contribution instead.
        A calm stock earns a bigger position; a jumpy one earns a smaller one.
      </p>
    </div>
    <div class="panel">
      <div class="scroller">
        <table>
          <thead>
            <tr><th>Symbol</th><th>Forecast vol</th><th>Equal weight</th><th>Risk-weighted</th></tr>
          </thead>
          <tbody>__SIZING_ROWS__</tbody>
        </table>
      </div>
    </div>
  </section>

  <footer>
    <p>
      <strong>Method.</strong> Gradient boosting on
      <span class="num">log(forward volatility / EWMA anchor)</span> rather than the
      volatility level &mdash; stocks here span an order of magnitude in baseline
      volatility and cannot share an intercept. Chronological
      70/15/15 split with a 33-day purge at each boundary so no target straddles it.
      Iteration count (__ITERS__) chosen on the validation slice, never on test.
    </p>
    <p>
      <strong>Data.</strong> __SESSIONS__ NSE sessions rebuilt from daily bhavcopy
      archives, survivorship-free universe reconstructed point-in-time, corporate
      actions recovered from NSE&rsquo;s own prevclose convention, 368 ISIN changes
      bridged.
    </p>
    <p>
      Research tool for a paper-trading account. Not investment advice, and not
      produced by a registered adviser.
    </p>
  </footer>
</div>

<script>
  const FORECASTS = __FORECASTS__;
  const tbody = document.getElementById('fbody');
  const q = document.getElementById('q');
  const sortSel = document.getElementById('sort');
  const count = document.getElementById('fcount');
  const maxVol = Math.max(...FORECASTS.map(f => f.forecast));

  function render() {
    const term = q.value.trim().toUpperCase();
    let rows = FORECASTS.filter(f => !term || f.symbol.includes(term));
    const mode = sortSel.value;
    if (mode === 'calm') rows.sort((a, b) => a.forecast - b.forecast);
    if (mode === 'wild') rows.sort((a, b) => b.forecast - a.forecast);
    if (mode === 'drop') rows.sort((a, b) => a.change - b.change);
    if (mode === 'rise') rows.sort((a, b) => b.change - a.change);

    const shown = rows.slice(0, 40);
    tbody.innerHTML = shown.map(f => {
      const w = Math.max(2, (f.forecast / maxVol) * 100);
      const col = f.change < 0 ? 'var(--accent)' : 'var(--warn)';
      const sign = f.change > 0 ? '+' : '';
      return `<tr>
        <td><strong>${f.symbol}</strong></td>
        <td class="num">${f.price === null ? '&mdash;' : f.price.toLocaleString('en-IN')}</td>
        <td class="num" style="color:var(--ink-faint)">${f.trailing.toFixed(1)}%</td>
        <td class="num" style="color:var(--ink-faint)">${f.ewma.toFixed(1)}%</td>
        <td>
          <span class="bar-cell">
            <span class="num">${f.forecast.toFixed(1)}%</span>
            <span class="bar" style="width:${w * 0.9}px"></span>
          </span>
        </td>
        <td class="num" style="color:${col}">${sign}${f.change.toFixed(1)}%</td>
      </tr>`;
    }).join('');
    count.textContent = `Showing ${shown.length} of ${rows.length} stocks`;
  }
  q.addEventListener('input', render);
  sortSel.addEventListener('change', render);
  render();
</script>
"""


def chart_svg(per_date: list[dict]) -> str:
    """Bar chart of monthly improvement, coloured by volatility regime."""
    w, h = 980, 320
    pad_l, pad_r, pad_t, pad_b = 52, 16, 20, 62
    iw = w - pad_l - pad_r
    ih = h - pad_t - pad_b

    vals = [d["impr"] for d in per_date]
    lo = min(min(vals), 0) - 3
    hi = max(max(vals), 0) + 3

    def y(v: float) -> float:
        return pad_t + (hi - v) / (hi - lo) * ih

    n = len(per_date)
    slot = iw / n
    bw = min(34, slot * 0.62)

    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" '
             f'aria-label="Monthly improvement over EWMA">']

    # gridlines
    step = 10
    g = int(lo // step) * step
    while g <= hi:
        yy = y(g)
        parts.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w - pad_r}" y2="{yy:.1f}" '
            f'stroke="var(--rule)" stroke-width="1"/>')
        parts.append(
            f'<text x="{pad_l - 10}" y="{yy + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="var(--ink-faint)" '
            f'font-family="ui-monospace, monospace">{g:+d}%</text>')
        g += step

    zero = y(0)
    parts.append(f'<line x1="{pad_l}" y1="{zero:.1f}" x2="{w - pad_r}" y2="{zero:.1f}" '
                 f'stroke="var(--ink-faint)" stroke-width="1.5"/>')

    for i, d in enumerate(per_date):
        cx = pad_l + slot * (i + 0.5)
        v = d["impr"]
        rose = d["vol_change"] > 0
        colour = "var(--warn)" if rose else "var(--accent)"
        top = y(max(v, 0))
        height = abs(y(v) - zero)
        parts.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
            f'height="{max(height, 1.2):.1f}" rx="2" fill="{colour}" opacity="0.88">'
            f'<title>{d["date"]}: {v:+.1f}% vs EWMA, volatility '
            f'{"rose" if rose else "fell"} {d["vol_change"]:+.1f}%</title></rect>')
        if i % 2 == 0:
            label = d["date"][:7]
            parts.append(
                f'<text x="{cx:.1f}" y="{h - pad_b + 20:.1f}" text-anchor="middle" '
                f'font-size="10" fill="var(--ink-faint)" '
                f'font-family="ui-monospace, monospace" '
                f'transform="rotate(-45 {cx:.1f} {h - pad_b + 20:.1f})">{label}</text>')

    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    d = DATA
    metric_rows = []
    best = min(m["rmse"] for m in d["metrics"])
    for m in d["metrics"]:
        is_best = m["rmse"] == best
        cls = ' class="win"' if is_best else ""
        strong = ' class="num best"' if is_best else ' class="num"'
        metric_rows.append(
            f'<tr{cls}><td><strong>{m["model"]}</strong></td>'
            f'<td{strong}>{m["rmse"]:.5f}</td>'
            f'<td class="num">{m["corr"]:.3f}</td></tr>')

    sizing_rows = []
    for s in d["sizing"]:
        delta = s["risk_weighted"] - s["equal"]
        col = "var(--accent)" if delta > 0 else "var(--base)"
        sizing_rows.append(
            f'<tr><td><strong>{s["symbol"]}</strong></td>'
            f'<td class="num">{s["forecast"]:.1f}%</td>'
            f'<td class="num" style="color:var(--ink-faint)">{s["equal"]:.2f}%</td>'
            f'<td class="num" style="color:{col}"><strong>{s["risk_weighted"]:.2f}%</strong></td></tr>')

    sig, reg = d["significance"], d["regime"]
    html = (HTML
            .replace("__AS_OF__", d["as_of"])
            .replace("__STOCKS__", f'{d["stocks_modelled"]:,}')
            .replace("__SESSIONS__", f'{d["sessions"]:,}')
            .replace("__TRAINROWS__", f'{d["train_rows"]:,}')
            .replace("__IMPR_EWMA__", str(d["improvement"]["vs_ewma"]))
            .replace("__IMPR_PERS__", str(d["improvement"]["vs_persistence"]))
            .replace("__WINS__", str(sig["wins"]))
            .replace("__MONTHS__", str(sig["months"]))
            .replace("__TP__", f'{sig["t_p"]:.0e}'.replace("e-0", "e−"))
            .replace("__MEAN_IMPR__", f'{sig["mean_impr"]:.1f}')
            .replace("__REGIME_CORR__", f'{reg["corr"]:.2f}')
            .replace("__REGIME_P__", f'{reg["corr_p"]:.4f}')
            .replace("__FELL_IMPR__", f'{reg["fell_impr"]:.1f}')
            .replace("__FELL_N__", str(reg["fell_n"]))
            .replace("__ROSE_IMPR__", f'{reg["rose_impr"]:.1f}')
            .replace("__ROSE_N__", str(reg["rose_n"]))
            .replace("__ITERS__", str(d["best_iter"]))
            .replace("__METRIC_ROWS__", "".join(metric_rows))
            .replace("__SIZING_ROWS__", "".join(sizing_rows))
            .replace("__CHART__", chart_svg(d["per_date"]))
            .replace("__FORECASTS__", json.dumps(d["forecasts"])))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
