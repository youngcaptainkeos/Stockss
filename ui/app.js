// The UI consumes only immutable research artifacts. It has no broker endpoints.
const $ = (id) => document.getElementById(id);
const colors = {teal: '#34d6b0', red: '#f67789', muted: '#8192aa', grid: '#223043', violet: '#a59aff'};
const money = (v, digits = 0) => v == null ? '—' : new Intl.NumberFormat('en-IN', {style: 'currency', currency: 'INR', maximumFractionDigits: digits}).format(v);
const number = (v, digits = 2) => v == null ? '—' : Number(v).toLocaleString('en-IN', {maximumFractionDigits: digits, minimumFractionDigits: digits});
const pct = (v, signed = false) => v == null ? '—' : `${signed && v > 0 ? '+' : ''}${(v * 100).toFixed(2)}%`;
const label = (name) => name.split('_').map(x => ({ridge: 'Ridge', lightgbm: 'LightGBM', per: 'Per', stock: 'stock', pooled: 'Pooled', hybrid: 'Hybrid', price: 'Price', context: 'Market + sector', buy: 'Buy', hold: 'hold', cash: 'Cash', momentum: 'Momentum', ema: 'EMA trend', mean: 'Mean', reversion: 'reversion'}[x] || x)).join(' · ').replace('Per · stock', 'Per-stock').replace('Buy · hold', 'Buy & hold').replace('Mean · reversion', 'Mean reversion');
const state = {data: null, cursor: 0, strategy: '', symbol: 'NETWEB', timer: null, view: 'replay'};
let candleView = [];

function setView(view) {
  state.view = view;
  for (const name of ['replay', 'evaluation', 'method']) $(`${name}-view`).hidden = name !== view;
  document.querySelectorAll('.nav').forEach(el => el.classList.toggle('active', el.dataset.view === view));
  if (view !== 'replay') stop();
  requestAnimationFrame(renderCharts);
}

function stop() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  $('play').textContent = 'Play replay';
}

function advance() {
  if (state.cursor < dates().length - 1) state.cursor++;
  else stop();
  render();
}
function dates() { return state.data.strategies[state.strategy].curve.map(x => x.date); }
function currentDate() { return dates()[state.cursor]; }
function currentStrategy() { return state.data.strategies[state.strategy]; }
function colorValue(el, value) { el.classList.remove('positive', 'negative'); if (value !== 0) el.classList.add(value > 0 ? 'positive' : 'negative'); }
function frame(canvas) {
  const box = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(box.width * ratio); canvas.height = Math.round(box.height * ratio);
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
  ctx.font = '11px ui-monospace, monospace'; ctx.lineWidth = 1;
  return {ctx, w: box.width, h: box.height};
}

function drawCandles() {
  if (state.view !== 'replay') return;
  const {ctx, w, h} = frame($('candles'));
  if (!w) return;
  candleView = (state.data.candles[state.symbol] || []).filter(x => x.date <= currentDate()).slice(-80);
  if (!candleView.length) {ctx.fillStyle = colors.muted; ctx.fillText('No observations at this date', 30, 60); return;}
  const left = 16, right = 75, top = 16, bottom = 24, volHeight = h * .16;
  const chartH = h - top - bottom - volHeight - 16;
  let lo = Math.min(...candleView.map(x => x.low)), hi = Math.max(...candleView.map(x => x.high));
  const pad = (hi - lo) * .08 || 1; lo -= pad; hi += pad;
  const y = p => top + (hi - p) / (hi - lo) * chartH;
  const step = (w - left - right) / Math.max(20, candleView.length);
  const x = i => left + (i + .5) * step;
  for (let j = 0; j <= 4; j++) {
    const py = top + j * chartH / 4;
    ctx.strokeStyle = colors.grid; ctx.globalAlpha = .6; ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(w - right, py); ctx.stroke(); ctx.globalAlpha = 1;
    ctx.fillStyle = colors.muted; ctx.fillText(number(hi - j * (hi - lo) / 4), w - right + 10, py + 4);
  }
  const maxVolume = Math.max(...candleView.map(x => x.volume), 1);
  const trades = currentStrategy().trades.filter(x => x.symbol === state.symbol && x.date <= currentDate());
  candleView.forEach((c, i) => {
    const col = c.close >= c.open ? colors.teal : colors.red;
    ctx.strokeStyle = col; ctx.fillStyle = col;
    ctx.beginPath(); ctx.moveTo(x(i), y(c.high)); ctx.lineTo(x(i), y(c.low)); ctx.stroke();
    ctx.fillRect(x(i) - step * .31, Math.min(y(c.open), y(c.close)), Math.max(2, step * .62), Math.max(1.5, Math.abs(y(c.open) - y(c.close))));
    ctx.globalAlpha = .24; const vh = c.volume / maxVolume * volHeight;
    ctx.fillRect(x(i) - step * .31, h - bottom - vh, Math.max(2, step * .62), vh); ctx.globalAlpha = 1;
    trades.filter(t => t.date === c.date).forEach(t => {
      const buy = t.side === 'BUY', yy = buy ? y(c.low) + 10 : y(c.high) - 10;
      ctx.fillStyle = buy ? colors.teal : colors.red; ctx.beginPath();
      ctx.moveTo(x(i), yy + (buy ? -4 : 4)); ctx.lineTo(x(i) - 4, yy + (buy ? 4 : -4)); ctx.lineTo(x(i) + 4, yy + (buy ? 4 : -4)); ctx.closePath(); ctx.fill();
    });
    if (i % Math.max(1, Math.floor(candleView.length / 5)) === 0) {
      ctx.fillStyle = colors.muted; ctx.fillText(c.date.slice(5), x(i) - 15, h - 6);
    }
  });
  const last = candleView.at(-1); ctx.fillStyle = colors.teal; ctx.fillRect(w - right + 2, y(last.close) - 9, 70, 18);
  ctx.fillStyle = '#08221c'; ctx.fillText(number(last.close), w - right + 6, y(last.close) + 4);
}

function drawEquity() {
  if (state.view !== 'replay') return;
  const {ctx, w, h} = frame($('equityChart')); if (!w) return;
  const asof = currentDate(), initial = state.data.config.initial_cash;
  const lines = [
    {data: currentStrategy().curve, color: colors.teal},
    {data: state.data.strategies.buy_hold.curve, color: colors.violet},
    {data: state.data.market_curve, color: colors.muted}
  ].map(line => ({...line, data: [{equity: initial, date: '2024-12-31'}, ...line.data.filter(x => x.date <= asof)]}));
  const all = lines.flatMap(l => l.data.map(x => x.equity));
  let lo = Math.min(...all) * .995, hi = Math.max(...all) * 1.005;
  if (hi === lo) hi += 100;
  const left = 20, right = 75, top = 16, bottom = 28;
  const xx = i => left + i / Math.max(20, lines[0].data.length - 1) * (w - left - right);
  const yy = v => top + (hi - v) / (hi - lo) * (h - top - bottom);
  for (let i = 0; i <= 3; i++) {
    const v = hi - i * (hi - lo) / 3, y = yy(v);
    ctx.strokeStyle = colors.grid; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w - right, y); ctx.stroke();
    ctx.fillStyle = colors.muted; ctx.fillText(`${(v / 100000).toFixed(2)}L`, w - right + 10, y + 4);
  }
  for (const line of lines.reverse()) {
    ctx.strokeStyle = line.color; ctx.lineWidth = line.color === colors.teal ? 2 : 1.3;
    ctx.beginPath(); line.data.forEach((p, i) => i ? ctx.lineTo(xx(i), yy(p.equity)) : ctx.moveTo(xx(i), yy(p.equity))); ctx.stroke();
  }
  ctx.fillStyle = colors.muted; ctx.fillText('JAN 2025', left, h - 8); ctx.fillText(asof, w - right - 70, h - 8);
}
function renderCharts() { if (state.data) {drawCandles(); drawEquity();} }

function render() {
  const date = currentDate(), s = currentStrategy(), row = s.curve[state.cursor], initial = state.data.config.initial_cash;
  $('asof').textContent = new Date(date + 'T12:00:00').toLocaleDateString('en-IN', {day: '2-digit', month: 'short', year: 'numeric'}).toUpperCase();
  $('dateSeek').value = date; $('timeline').value = state.cursor;
  $('sessionCount').textContent = `SESSION ${state.cursor + 1} / ${dates().length}`;
  $('equity').textContent = money(row.equity); $('return').textContent = pct(row.equity / initial - 1, true);
  colorValue($('return'), row.equity - initial); $('netPnl').textContent = `${money(row.equity - initial)} net P&L`;
  $('costs').textContent = money(row.costs); $('slippage').textContent = `+ ${money(row.slippage)} adverse slippage`;
  $('drawdown').textContent = pct(row.drawdown); colorValue($('drawdown'), row.drawdown);
  $('exposure').textContent = pct(row.exposure); $('positions').textContent = `${row.positions} positions · ${row.halted ? 'Risk halt active' : 'Long / cash'}`;
  $('selectedBadge').textContent = state.strategy === state.data.summary.selected_model ? 'Selected before final test' : 'Comparison strategy';
  const candles = state.data.candles[state.symbol].filter(x => x.date <= date), bar = candles.at(-1);
  $('tickerName').textContent = state.symbol; $('lastPrice').textContent = bar ? money(bar.close, 2) : '—';
  $('ohlc').textContent = bar ? `O ${number(bar.open)}   H ${number(bar.high)}   L ${number(bar.low)}   C ${number(bar.close)}   VOL ${number(bar.volume, 0)}` : 'No historical bar';
  const watch = [...new Set([state.symbol, 'NETWEB', 'E2E', 'TATACOMM', 'RELIANCE'])].slice(0, 4);
  $('watchlist').innerHTML = watch.map(symbol => {
    const c = state.data.candles[symbol].filter(x => x.date <= date), last = c.at(-1), prev = c.at(-2), ret = prev && last ? last.close / prev.close - 1 : 0;
    return `<button class="watch-row ${state.symbol === symbol ? 'active' : ''}" data-symbol="${symbol}"><span>${symbol}<small>${last ? money(last.close, 2) : '—'}</small></span><span class="${ret >= 0 ? 'positive' : 'negative'}">${pct(ret, true)}</span></button>`;
  }).join('');
  const signal = s.signals.find(x => x.date === date && x.symbol === state.symbol);
  const available = signal && signal.score != null;
  const enter = available && signal.eligible !== false && signal.score > signal.threshold;
  $('signalTitle').textContent = row.halted ? 'Portfolio risk halt' : enter ? 'Entry candidate' : available ? 'No entry signal' : 'Model unavailable';
  $('signalTitle').style.color = enter && !row.halted ? colors.teal : colors.muted;
  $('signalText').textContent = row.halted ? 'New entries are disabled for the rest of this run.' : signal?.eligible === false ? 'This instrument does not meet the history/liquidity rules.' : enter ? 'Above the selected threshold. Portfolio ranking and available cash still apply.' : 'The current rule does not qualify this instrument for entry.';
  $('forecast').textContent = available && signal.probability != null ? pct(signal.score, true) : 'Rule-based / unavailable';
  $('probability').textContent = signal?.probability != null ? pct(signal.probability) : '—';
  const trades = s.trades.filter(x => x.date <= date); $('tradeCount').textContent = `${trades.length} fills`;
  $('ledger').innerHTML = trades.length ? trades.slice(-60).reverse().map(t => `<tr><td>${t.date.slice(5)}</td><td>${t.symbol}<span class="trade-side ${t.side === 'SELL' ? 'sell' : ''}">${t.side}</span></td><td>${number(t.quantity, 0)}</td><td>${money(t.price, 2)}</td><td>${money(t.fees, 2)}</td><td class="${t.pnl > 0 ? 'positive' : t.pnl < 0 ? 'negative' : ''}">${t.pnl == null ? '—' : money(t.pnl)}</td></tr>`).join('') : '<tr><td colspan="6" class="empty">No fills yet. Advance the historical replay.</td></tr>';
  renderCharts();
}

function evaluation() {
  const summary = state.data.summary, chosen = summary.strategies[summary.selected_model];
  $('selectedName').textContent = label(summary.selected_model);
  const titles = {positive_net_return: 'Positive net return', beats_buy_hold: 'Beats buy & hold', beats_nifty_price: 'Beats NIFTY price index', sharpe_above_0_5: 'Sharpe above 0.5', positive_at_double_cost: 'Positive at double costs', at_least_30_trades: 'At least 30 completed trades', positive_95_percent_return_lower_bound: 'Positive 95% lower bound', all_positions_liquidated: 'All positions liquidated'};
  $('criteria').innerHTML = Object.entries(summary.criteria).map(([k, pass]) => `<div class="criterion ${pass ? 'pass' : ''}"><b>${pass ? '✓' : '×'}</b>${titles[k] || k}</div>`).join('');
  $('stress').innerHTML = Object.entries(chosen.cost_stress).map(([mult, v]) => `<div class="stress-row"><span>${mult}× fees & slippage</span><strong class="${v.net_return > 0 ? 'positive' : 'negative'}">${pct(v.net_return, true)}</strong></div>`).join('') + `<p>95% block-bootstrap return interval: <strong>${chosen.uncertainty.net_return_95_interval.map(v => pct(v)).join(' to ')}</strong>.<br>Conditional on this strategy and year; not adjusted for model-search selection.</p>`;
  const rows = Object.entries(summary.strategies);
  $('comparison').innerHTML = rows.map(([name, v]) => `<tr><td>${label(name)}${name === summary.selected_model ? '<span class="chosen">SELECTED</span>' : ''}</td><td class="${v.net_return > 0 ? 'positive' : v.net_return < 0 ? 'negative' : ''}">${pct(v.net_return, true)}</td><td>${number(v.sharpe)}</td><td class="negative">${pct(v.max_drawdown)}</td><td>${money(v.costs)}</td><td>${v.round_trips}</td><td>${pct(v.mean_exposure)}</td><td>${v.classification ? number(v.classification.brier, 3) : '—'}</td></tr>`).join('');
  $('ablation').innerHTML = `<p>Two feature groups and three architectures were compared for each family. Use the table to inspect whether context adds value consistently; a single test comparison is not a new selection decision.</p><p>NETWEB, E2E and TATACOMM closed-trade P&L in the selected strategy: <strong>${['NETWEB', 'E2E', 'TATACOMM'].map(s => `${s}: ${money(chosen.per_stock_pnl[s])}`).join(' · ')}</strong>.</p><p>News, fundamentals and intraday strategies remain untested because suitable timestamped source history was unavailable.</p>`;
  const c = chosen.classification;
  $('calibration').innerHTML = c ? `<p>Direction accuracy <strong>${pct(c.accuracy)}</strong> · balanced accuracy <strong>${pct(c.balanced_accuracy)}</strong><br>ROC AUC <strong>${number(c.auc, 3)}</strong> · Brier score <strong>${number(c.brier, 3)}</strong><br>Calibration error <strong>${number(c.expected_calibration_error, 3)}</strong> · five-session return MAE <strong>${pct(c.return_mae)}</strong></p><p>${c.n.toLocaleString()} eligible labelled observations. Labels overlap across time and stocks; accuracy alone does not establish profitability.</p>` : '<p>No classification model for this strategy.</p>';
  $('sourceSummary').innerHTML = `<p>Actual Yahoo Finance daily NSE OHLCV and Indian index histories, cached with checksums. No Groww environment credential was found. Development uses 2020–2024; the final test uses 2025. No US data is used.</p><p>NETWEB begins on 27 July 2023. E2E is excluded before its mainboard migration on 12 April 2022. Its final local model abstains: 118 eligible core observations versus the fixed 126 minimum. The hybrid uses pooled predictions for E2E. Some instruments have zero-volume sessions; missing or locked bars do not receive assumed fills.</p>`;
  $('verdict-title').textContent = 'Research-only replay.';
  $('verdict-text').textContent = summary.statistical_gate_passed ? 'Statistical gates passed in this sample; data and execution limits still block deployment.' : `The selected model ${chosen.net_return >= 0 ? 'gained' : 'lost'} ${pct(Math.abs(chosen.net_return))} after modelled costs and did not pass the profitability gate.`;
}

async function init() {
  try {
    const response = await fetch('replay.json'); if (!response.ok) throw new Error('Evaluation data is not ready. Run the pipeline first.');
    state.data = await response.json(); state.strategy = state.data.summary.selected_model;
    $('strategy').innerHTML = Object.keys(state.data.strategies).map(s => `<option value="${s}">${label(s)}</option>`).join('');
    $('strategy').value = state.strategy;
    $('symbol').innerHTML = Object.keys(state.data.config.universe).map(s => `<option value="${s}">${s} · NSE</option>`).join(''); $('symbol').value = state.symbol;
    $('timeline').max = dates().length - 1;
    $('timelineStart').textContent = dates()[0];
    $('timelineEnd').textContent = dates().at(-1);
    // Start one quarter into the sample, with all future observations still hidden.
    state.cursor = Math.min(62, dates().length - 1);
    $('loading').hidden = true; $('app').hidden = false;
    evaluation(); render();
  } catch (error) { $('loading').textContent = error.message; }
}

document.querySelectorAll('.nav').forEach(el => el.addEventListener('click', () => setView(el.dataset.view)));
$('viewEvidence').addEventListener('click', () => setView('evaluation'));
$('strategy').addEventListener('change', e => {state.strategy = e.target.value; render();});
$('symbol').addEventListener('change', e => {state.symbol = e.target.value; render();});
$('watchlist').addEventListener('click', e => {const row = e.target.closest('[data-symbol]'); if (row) {state.symbol = row.dataset.symbol; $('symbol').value = state.symbol; render();}});
$('timeline').addEventListener('input', e => {stop(); state.cursor = Number(e.target.value); render();});
$('dateSeek').addEventListener('change', e => {stop(); const i = dates().findIndex(d => d >= e.target.value); state.cursor = i < 0 ? dates().length - 1 : i; render();});
$('step').addEventListener('click', () => {stop(); advance();});
$('reset').addEventListener('click', () => {stop(); state.cursor = 0; render();});
$('play').addEventListener('click', () => {
  if (state.timer) return stop();
  if (state.cursor === dates().length - 1) state.cursor = 0;
  $('play').textContent = 'Pause replay'; state.timer = setInterval(advance, Number($('speed').value));
});
$('speed').addEventListener('change', () => {if (state.timer) {stop(); $('play').click();}});
$('candles').addEventListener('mousemove', e => {
  if (!candleView.length) return;
  const rect = e.target.getBoundingClientRect(), step = (rect.width - 91) / Math.max(20, candleView.length);
  const i = Math.floor((e.clientX - rect.left - 16) / step), c = candleView[i];
  if (!c) {$('chartTip').hidden = true; return;}
  $('chartTip').textContent = `${c.date} · O ${number(c.open)} · H ${number(c.high)} · L ${number(c.low)} · C ${number(c.close)}`;
  $('chartTip').hidden = false;
});
$('candles').addEventListener('mouseleave', () => {$('chartTip').hidden = true;});
$('export').addEventListener('click', () => {
  const rows = [['strategy', 'net_return', 'sharpe', 'max_drawdown', 'costs', 'slippage', 'round_trips'], ...Object.entries(state.data.summary.strategies).map(([k, v]) => [k, v.net_return, v.sharpe, v.max_drawdown, v.costs, v.slippage, v.round_trips])];
  const url = URL.createObjectURL(new Blob([rows.map(r => r.join(',')).join('\n')], {type: 'text/csv'}));
  const a = document.createElement('a'); a.href = url; a.download = 'nse-2025-evaluation.csv'; a.click(); URL.revokeObjectURL(url);
});
window.addEventListener('resize', renderCharts);
init();
