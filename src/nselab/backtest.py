"""Daily, event-driven, long/cash simulator. Broker order APIs do not exist here."""

import math
import numpy as np
import pandas as pd
from .common import config
from .costs import IndiaDeliveryCosts


def simulate(
    features, signals, start, end, cfg=None, cost_multiplier=1.0, buy_hold=False
):
    cfg = cfg or config()
    fees = IndiaDeliveryCosts()
    f = features.copy()
    all_dates = sorted(f.date.unique())
    dates = [
        pd.Timestamp(d)
        for d in all_dates
        if pd.Timestamp(start) <= d <= pd.Timestamp(end)
    ]
    if len(dates) < 2:
        raise ValueError("Insufficient backtest sessions")
    lookup = {(r.date, r.symbol): r for r in f.itertuples()}
    sig = {(r.date, r.symbol): r for r in signals.itertuples()}
    calendar = [pd.Timestamp(d) for d in all_dates]
    index = {d: i for i, d in enumerate(calendar)}
    cash = float(cfg["initial_cash"])
    receivable = 0.0
    positions = {}
    unsettled = []
    peak = cash
    halted = False
    rows = []
    trades = []
    closed = []
    rejects = []
    initial_bought = False
    first = index[dates[0]]
    last_price = {}
    pending_exits = set()
    costs = 0.0
    slippage = 0.0
    turnover = 0.0

    def slip(symbol):
        return (
            (
                cfg["special_slippage_bps"]
                if symbol in ["NETWEB", "E2E"]
                else cfg["slippage_bps"]
            )
            * 1e-4
            * cost_multiplier
        )

    def executable(bar):
        return (
            bar is not None
            and np.isfinite(bar.raw_open)
            and bar.raw_open > 0
            and bar.raw_volume > 0
            and bar.raw_high > bar.raw_low
        )

    def sell(symbol, date, reason):
        nonlocal cash, costs, slippage, turnover
        p = positions[symbol]
        b = lookup.get((date, symbol))
        if not executable(b):
            pending_exits.add(symbol)
            rejects.append(
                {
                    "date": str(date.date()),
                    "symbol": symbol,
                    "reason": "exit deferred: missing or locked bar",
                }
            )
            return
        px = float(b.raw_open) * (1 - slip(symbol))
        value = p["qty"] * px
        component = fees.components(value, "SELL", cost_multiplier)
        c = component["total"]
        costs += c
        slippage += p["qty"] * float(b.raw_open) - value
        turnover += value
        # Conservative fixed T+2 settlement throughout history, including T+1 era.
        unsettled.append((index[date] + 2, value - c))
        pnl = value - c - p["entry_cost"] + p["dividends"]
        trades.append(
            {
                "date": str(date.date()),
                "signal_date": p["signal_date"],
                "symbol": symbol,
                "side": "SELL",
                "quantity": p["qty"],
                "price": px,
                "notional": value,
                "fees": c,
                "fee_components": component,
                "pnl": pnl,
                "reason": reason,
            }
        )
        closed.append(
            {
                "symbol": symbol,
                "entry": p["entry"],
                "exit": str(date.date()),
                "pnl": pnl,
                "return": pnl / p["entry_cost"],
                "fees": p["entry_fees"] + c,
                "dividends": p["dividends"],
            }
        )
        del positions[symbol]
        pending_exits.discard(symbol)

    for date in dates:
        i = index[date]
        # Only settled cash can fund purchases. Sale receivables remain in equity.
        ready = [v for release, v in unsettled if release <= i]
        cash += sum(ready)
        unsettled = [(release, v) for release, v in unsettled if release > i]
        for symbol, p in list(positions.items()):
            b = lookup.get((date, symbol))
            if b is None:
                continue
            if np.isfinite(b.split) and b.split != 1:
                p["qty"] *= float(b.split)
                if symbol in last_price:
                    last_price[symbol] /= float(b.split)
            if np.isfinite(b.dividend) and b.dividend > 0:
                amount = p["qty"] * float(b.dividend)
                p["dividends"] += amount
                receivable += amount
            if np.isfinite(b.raw_open):
                last_price[symbol] = float(b.raw_open)
        # A signal at close t fills at the following session's open.
        prev = calendar[i - 1] if i > 0 else None
        rebal = (i - first) % cfg["horizon"] == 0
        final = date == dates[-1]
        if final or halted or (rebal and not buy_hold):
            for symbol in list(positions):
                sell(
                    symbol,
                    date,
                    (
                        "end of evaluation"
                        if final
                        else "risk halt" if halted else "five-session exit"
                    ),
                )
        for symbol in list(pending_exits):
            if symbol in positions:
                sell(symbol, date, "deferred exit")
        equity_open = (
            cash
            + sum(v for _, v in unsettled)
            + receivable
            + sum(p["qty"] * last_price.get(s, 0) for s, p in positions.items())
        )
        can_buy = (
            (rebal or (buy_hold and not initial_bought)) and not halted and not final
        )
        # Stop entries too close to the end to realize the prespecified horizon.
        can_buy = can_buy and i + cfg["horizon"] <= index[dates[-1]]
        if can_buy:
            candidates = []
            for symbol in cfg["universe"]:
                s = sig.get((prev, symbol))
                b = lookup.get((date, symbol))
                before = lookup.get((prev, symbol))
                if (
                    s is None
                    or before is None
                    or not before.eligible
                    or symbol in positions
                ):
                    continue
                if not np.isfinite(s.score) or s.score <= s.threshold:
                    continue
                if not executable(b):
                    rejects.append(
                        {
                            "date": str(date.date()),
                            "symbol": symbol,
                            "reason": "entry rejected: missing or locked bar",
                        }
                    )
                    continue
                candidates.append((float(s.score), symbol, b, before))
            candidates.sort(key=lambda x: (-x[0], x[1]))
            sector_values = {}
            for symbol, p in positions.items():
                sec = cfg["universe"][symbol]
                sector_values[sec] = sector_values.get(sec, 0) + p[
                    "qty"
                ] * last_price.get(symbol, 0)
            gross = sum(sector_values.values())
            for score, symbol, b, before in candidates:
                if len(positions) >= cfg["max_positions"]:
                    break
                sec = cfg["universe"][symbol]
                budget = max(
                    0,
                    min(
                        equity_open * cfg["max_weight"],
                        equity_open * cfg["max_gross"] - gross,
                        equity_open * cfg["max_sector_weight"]
                        - sector_values.get(sec, 0),
                        float(before.adv20) * cfg["max_adv_participation"],
                        cash,
                    ),
                )
                px = float(b.raw_open) * (1 + slip(symbol))
                qty = int(budget / px)
                while (
                    qty > 0
                    and qty * px
                    + fees.components(qty * px, "BUY", cost_multiplier)["total"]
                    > budget
                ):
                    qty -= 1
                if qty <= 0:
                    continue
                value = qty * px
                component = fees.components(value, "BUY", cost_multiplier)
                c = component["total"]
                cash -= value + c
                costs += c
                slippage += value - qty * float(b.raw_open)
                turnover += value
                positions[symbol] = {
                    "qty": qty,
                    "entry_cost": value + c,
                    "entry_fees": c,
                    "entry": str(date.date()),
                    "dividends": 0.0,
                    "signal_date": str(prev.date()),
                }
                last_price[symbol] = float(b.raw_open)
                sector_values[sec] = sector_values.get(sec, 0) + qty * float(b.raw_open)
                gross += qty * float(b.raw_open)
                trades.append(
                    {
                        "date": str(date.date()),
                        "signal_date": str(prev.date()),
                        "symbol": symbol,
                        "side": "BUY",
                        "quantity": qty,
                        "price": px,
                        "notional": value,
                        "fees": c,
                        "fee_components": component,
                        "pnl": None,
                        "reason": "next-open signal",
                        "score": score,
                    }
                )
            if buy_hold:
                initial_bought = True
        # Mark positions to contemporaneous close, never to a future fill.
        for symbol in positions:
            b = lookup.get((date, symbol))
            if b is not None and np.isfinite(b.raw_close):
                last_price[symbol] = float(b.raw_close)
        holdings = sum(p["qty"] * last_price.get(s, 0) for s, p in positions.items())
        equity = cash + sum(v for _, v in unsettled) + receivable + holdings
        peak = max(peak, equity)
        dd = equity / peak - 1
        if dd <= -cfg["max_drawdown_halt"] and not buy_hold:
            halted = True
        if cash < -0.001:
            raise AssertionError("Negative cash / leverage")
        rows.append(
            {
                "date": str(date.date()),
                "equity": equity,
                "cash": cash,
                "unsettled_cash": sum(v for _, v in unsettled),
                "dividend_receivable": receivable,
                "exposure": holdings / equity if equity else 0.0,
                "drawdown": dd,
                "costs": costs,
                "slippage": slippage,
                "positions": len(positions),
                "halted": halted,
            }
        )
    curve = pd.DataFrame(rows)
    return {
        "curve": curve,
        "trades": trades,
        "closed_trades": closed,
        "rejects": rejects,
        "open_positions": positions,
        "costs": costs,
        "slippage": slippage,
        "turnover": turnover,
        "unliquidated": bool(positions),
    }


def baseline_signals(frame, name):
    s = frame[["date", "symbol"]].copy()
    s["threshold"] = 0.0
    if name == "cash":
        s["score"] = -1.0
    elif name == "buy_hold":
        # Stable alphabetical rank (no future-return selection), broad capped basket.
        s["score"] = 1.0
    elif name == "momentum":
        s["score"] = frame.ret60.where(frame.ret60 > 0, -1)
    elif name == "ema":
        s["score"] = (frame.ema50_gap - frame.ema20_gap).where(
            (frame.ema20_gap > 0) & (frame.ema50_gap > frame.ema20_gap), -1
        )
    elif name == "mean_reversion":
        s["score"] = (-frame.ret5).where((frame.rsi14 < 30) & (frame.ret5 < 0), -1)
    else:
        raise ValueError(name)
    return s
