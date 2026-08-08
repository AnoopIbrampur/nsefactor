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
    <span class="eyebrow">Indian stock market &middot; 500 companies</span>
    <h1>Guessing how bumpy a stock will be &mdash; not whether it goes up</h1>
    <p class="lede">
      Nobody can reliably say which stocks will rise. We tried, and lost to
      simply buying the whole market. So we changed the question to one that
      does have an answer: <strong>how much will this stock move around?</strong>
      Calm months tend to follow calm months. That pattern is real, and it is
      what this predicts.
    </p>
    <div class="meta">
      <span>As of <strong class="num">__AS_OF__</strong></span>
      <span><strong class="num">__STOCKS__</strong> stocks</span>
      <span><strong class="num">__SESSIONS__</strong> days of market history</span>
      <span>Rebuilt from official NSE records</span>
    </div>
  </header>

  <section>
    <span class="eyebrow">Step 1 &middot; The short version</span>
    <h2>How much better is it than the usual methods?</h2>
    <div class="prose">
      <p>
        There are standard ways professionals estimate this. We beat all of
        them. The numbers below are measured on months the model had never
        seen &mdash; not on the data it learned from, which is the easy way to
        look good and the reason most such claims are worthless.
      </p>
    </div>
    <div class="tiles">
      <div class="tile hero">
        <span class="v num">+__IMPR_EWMA__%</span>
        <span class="k">More accurate than the formula the industry actually uses</span>
      </div>
      <div class="tile hero">
        <span class="v num">+__IMPR_PERS__%</span>
        <span class="k">More accurate than assuming next month looks like last month</span>
      </div>
      <div class="tile">
        <span class="v num">__WINS__/__MONTHS__</span>
        <span class="k">Months out of 20 where it beat the standard method</span>
      </div>
      <div class="tile">
        <span class="v num">__TP__</span>
        <span class="k">Odds this was luck (smaller means less likely)</span>
      </div>
    </div>
  </section>

  <section>
    <span class="eyebrow">Step 2 &middot; The competition</span>
    <h2>Three standard methods, same test, same months</h2>
    <div class="prose">
      <p>
        Every method was asked the same question on the same months, then
        marked against what actually happened. <strong>&ldquo;Typical
        miss&rdquo; is how far off it was on average, so lower is
        better.</strong> &ldquo;Tracks reality&rdquo; is how closely its
        guesses rose and fell alongside the real thing &mdash; higher is better.
      </p>
      <p>
        The three it is up against are: assuming next month simply repeats last
        month, the formula banks and funds actually use day to day, and a
        classic model from the textbooks.
      </p>
    </div>
    <div class="panel">
      <div class="scroller">
        <table>
          <thead>
            <tr><th>Method</th><th>Typical miss</th><th>Tracks reality</th></tr>
          </thead>
          <tbody>__METRIC_ROWS__</tbody>
        </table>
      </div>
    </div>
  </section>

  <section>
    <span class="eyebrow">Step 3 &middot; Month by month</span>
    <h2>It wins most months &mdash; but look at which ones it loses</h2>
    <div class="prose">
      <p>
        Each bar is one month. Above the line means the model beat the standard
        method that month; below means it lost. It wins nearly every time.
      </p>
      <p>
        <strong>Now look at the colours.</strong> Teal months are ones where
        the market got calmer. Amber months are ones where it got choppier. The
        losses are amber. That is the whole story of what this tool is and
        isn&rsquo;t, in one picture.
      </p>
    </div>
    <div class="panel">
      <div class="panel-head">
        <h3>How much better than the standard method, each month</h3>
        <span class="num" style="color:var(--ink-faint);font-size:.82rem">
          mean +__MEAN_IMPR__%
        </span>
      </div>
      <div class="scroller">__CHART__</div>
      <div class="legend">
        <span><span class="swatch" style="background:var(--accent)"></span>Market got calmer this month</span>
        <span><span class="swatch" style="background:var(--warn)"></span>Market got choppier this month</span>
      </div>
    </div>
  </section>

  <section>
    <div class="caution">
      <span class="eyebrow" style="color:var(--warn)">Step 4 &middot; What it cannot do</span>
      <h2>It tells you when things will settle down. It will not warn you about a crash.</h2>
      <div class="prose">
        <p>
          This is the honest catch, and it belongs here rather than in small
          print at the bottom.
        </p>
        <p>
          The model is good at spotting when a jumpy stock is about to calm
          down. It is close to useless at spotting when a calm stock is about to
          blow up &mdash; which is, annoyingly, the thing you would most want
          warning about.
        </p>
        <p>
          <strong>So treat it as a &ldquo;how big a bet should I make on this
          one?&rdquo; tool, not a &ldquo;get out now&rdquo; tool.</strong> A risk
          tool that quietly stops working during a panic is worse than one that
          never worked at all, so it is better to know this up front.
        </p>
      </div>
      <div class="split">
        <div class="stat-inline">
          <div class="v num" style="color:var(--accent)">+__FELL_IMPR__%</div>
          <div class="k eyebrow" style="letter-spacing:.06em">
            better, in the __FELL_N__ months things calmed down
          </div>
        </div>
        <div class="stat-inline">
          <div class="v num" style="color:var(--warn)">+__ROSE_IMPR__%</div>
          <div class="k eyebrow" style="letter-spacing:.06em">
            better, in the __ROSE_N__ months things got worse
          </div>
        </div>
      </div>
    </div>
  </section>

  <section id="live-section" hidden>
    <span class="eyebrow" style="color:var(--warn)">Live &middot; refreshes every 15 minutes</span>
    <h2>Which stocks are moving more than they should today?</h2>
    <div class="prose">
      <p>
        For each stock we know roughly how much it moves on an ordinary day.
        This compares that against what it is actually doing.
        <strong>&ldquo;2.0x&rdquo; means it has moved twice as much as a normal
        day for that stock.</strong> Anything past about 2.5x is unusual.
      </p>
      <p>
        Note it is measured against each stock&rsquo;s own habits, not the
        market&rsquo;s. A quiet stock moving 1% can be a bigger deal than a wild
        one moving 3%.
      </p>
    </div>
    <div class="panel">
      <div class="panel-head">
        <h3>Moving more than usual</h3>
        <span id="live-meta" class="num" style="color:var(--ink-faint);font-size:.82rem"></span>
      </div>
      <div class="scroller">
        <table>
          <thead>
            <tr>
              <th>Stock</th><th>Price</th><th>Moved today</th>
              <th>Normal day</th><th>Expected swing</th><th>How unusual</th>
            </tr>
          </thead>
          <tbody id="livebody"></tbody>
        </table>
      </div>
      <div class="legend"><span id="live-note"></span></div>
    </div>
  </section>

  <section>
    <span class="eyebrow">Step 5 &middot; What it says right now</span>
    <h2>Every stock, ranked calmest to jumpiest</h2>
    <div class="prose">
      <p>
        <strong>Forecast</strong> is how much this stock is expected to swing
        over the next month, expressed as a yearly percentage &mdash; the way
        the industry quotes it. A stock at 15% is steady; one at 50% moves
        around a lot.
      </p>
      <p>
        <strong>Change</strong> compares that to where the stock has been.
        Negative means the model expects it to settle down from here; positive
        means it expects more movement than usual.
      </p>
    </div>
    <div class="panel">
      <div class="panel-head">
        <h3>Forecasts as of <span class="num">__AS_OF__</span></h3>
        <div class="controls">
          <input type="search" id="q" placeholder="Filter by symbol&hellip;" aria-label="Filter by symbol">
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
              <th>Stock</th><th>Price</th><th>Last month</th>
              <th>Recent average</th><th>Forecast</th><th>Change</th>
            </tr>
          </thead>
          <tbody id="fbody"></tbody>
        </table>
      </div>
      <div class="legend"><span id="fcount"></span></div>
    </div>
  </section>

  <section>
    <span class="eyebrow">Step 6 &middot; What you would actually do with it</span>
    <h2>Buy less of the jumpy ones</h2>
    <div class="prose">
      <p>
        Most people split money evenly &mdash; the same rupees into every stock
        they buy. The problem is that a wild stock and a steady stock then carry
        wildly different amounts of risk, and a handful of jumpy names quietly
        end up driving the whole portfolio.
      </p>
      <p>
        <strong>Putting less money into the jumpy ones evens that out.</strong>
        Below, the same twelve stocks sized both ways. The calm names get more,
        the jumpy names get less, and the portfolio stops depending on a few
        volatile bets.
      </p>
    </div>
    <div class="panel">
      <div class="scroller">
        <table>
          <thead>
            <tr><th>Stock</th><th>Expected swing</th><th>Splitting evenly</th><th>Adjusted for risk</th></tr>
          </thead>
          <tbody>__SIZING_ROWS__</tbody>
        </table>
      </div>
    </div>
  </section>

  <footer>
    <p>
      <strong>How the numbers were kept honest.</strong> The model was tested
      only on months it had never seen while learning. That sounds obvious and
      is the step most people skip &mdash; testing on data you trained on is how
      you get an impressive result that falls apart with real money.
    </p>
    <p>
      <strong>Where the data comes from.</strong> __SESSIONS__ days of official
      NSE closing prices, rebuilt from scratch. Three things were fixed that
      quietly ruin most such projects: companies that went bust are still
      included (using only today&rsquo;s surviving companies makes any strategy
      look better than it was); share splits are corrected, so a stock splitting
      1-for-10 is not recorded as a 90% crash; and 368 companies whose
      identifier changed had their history stitched back together.
    </p>
    <p>
      <strong>Not investment advice.</strong> This is a research tool built for
      a practice account, and the author is not a registered financial adviser.
      An earlier attempt at picking which stocks would rise lost to simply
      buying an index fund, and that result is published too rather than hidden.
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

  // ---- Live panel -------------------------------------------------------
  // Fetched from this page's own origin, so there is no CORS negotiation and
  // no external request. When the file is absent -- opened straight from disk,
  // or published somewhere without the intraday job -- the section simply
  // stays hidden rather than showing an empty table that could be mistaken
  // for a calm market.
  const liveSection = document.getElementById('live-section');
  const liveBody = document.getElementById('livebody');
  const liveMeta = document.getElementById('live-meta');
  const liveNote = document.getElementById('live-note');

  function renderLive(d) {
    if (!d || !d.available || !d.rows || !d.rows.length) return;
    liveSection.hidden = false;

    const stamp = (d.as_of || '').slice(0, 16).replace('T', ' ');
    // \\u00B7 rather than a literal middot: this file is also served as a plain
    // static asset, where a missing charset header makes the browser read UTF-8
    // bytes as latin-1 and render mojibake.
    const dot = '\\u00B7';
    liveMeta.textContent = `market ${d.market_open ? 'open' : 'closed'} ${dot} ${stamp} IST`;

    liveBody.innerHTML = d.rows.map(r => {
      const hot = r.surprise >= 2.5;
      const moveCol = r.move >= 0 ? 'var(--accent)' : 'var(--warn)';
      const sign = r.move > 0 ? '+' : '';
      return `<tr${hot ? ' class="win"' : ''}>
        <td><strong>${r.symbol}</strong></td>
        <td class="num">${r.price.toLocaleString('en-IN')}</td>
        <td class="num" style="color:${moveCol}">${sign}${r.move.toFixed(2)}%</td>
        <td class="num" style="color:var(--ink-faint)">${r.expected_move.toFixed(2)}%</td>
        <td class="num" style="color:var(--ink-faint)">${r.forecast.toFixed(1)}%</td>
        <td class="num" style="font-weight:600;color:${hot ? 'var(--warn)' : 'var(--ink)'}">
          ${r.surprise.toFixed(2)}&times;</td>
      </tr>`;
    }).join('');

    const hotCount = d.hot || 0;
    liveNote.textContent = hotCount
      ? `${hotCount} of ${d.scored} stocks running at 2.5x their typical move or more.`
      : `Nothing above 2.5x across ${d.scored} stocks \\u2014 an ordinary session so far.`;
  }

  fetch('live.json', {cache: 'no-store'})
    .then(r => r.ok ? r.json() : null)
    .then(renderLive)
    .catch(() => { /* no live feed here; the static page stands on its own */ });
</script>
"""


def _plain_odds(p: float) -> str:
    """A p-value as words, for a reader who has never met one.

    "1e-06" is precise and communicates nothing outside a stats course. The
    page is meant to be explained out loud, so the tile says how unlikely the
    result would be if the model had no skill at all.
    """
    if p <= 0 or p < 1e-6:
        return "&lt;1 in a million"
    odds = round(1 / p)
    if odds >= 1_000_000:
        return "&lt;1 in a million"
    if odds >= 1000:
        return f"1 in {odds // 1000:,}k"
    return f"1 in {odds:,}"


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
            .replace("__TP__", _plain_odds(sig["t_p"]))
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

    # The site copy is byte-identical. The only difference is that a live.json
    # sits next to it there, which the page picks up from its own origin; the
    # standalone copy simply never finds one and hides the live section.
    site = DATA_DIR.parent / "site" / "index.html"
    site.parent.mkdir(parents=True, exist_ok=True)
    site.write_text(html)
    print(f"wrote {site}")


if __name__ == "__main__":
    main()
