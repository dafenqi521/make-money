"""聚宽：场内 ETF 多资产趋势轮动策略（V1）。

适用范围
--------
1. 仅通过沪深交易所二级市场买卖 ETF 份额；
2. 不做一级市场申购赎回、LOF 折溢价套利或融资杠杆；
3. 即使标的支持 T+0，首版也不做当日回转；
4. 所有日线信号只使用 ``context.previous_date`` 及以前的完整数据。

使用方法：将本文件完整复制到聚宽策略编辑器，选择分钟级回测。
"""

from __future__ import annotations

from collections import defaultdict

import jqdata
from jqdata import *
import numpy as np
import pandas as pd


# =============================================================================
# 初始化与调度
# =============================================================================


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("avoid_future_data", True)
    set_option("use_real_price", True)
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0003,
            close_commission=0.0003,
            close_today_commission=0,
            min_commission=5,
        ),
        type="fund",
    )
    set_slippage(PriceRelatedSlippage(0.001))

    # 股票型 ETF 通常 T+1。为统一执行，所有 ETF 均不做日内回转。
    g.max_positions = 4
    g.cash_reserve = 0.10
    g.max_position_weight = 0.30
    g.min_listed_days = 120
    g.history_days = 121
    g.min_avg_money = 30_000_000.0
    g.min_daily_money = 5_000_000.0
    g.correlation_threshold = 0.90
    g.rank_exit = 8
    g.min_hold_days = 5
    g.time_stop_days = 10

    g.category_position_limits = {
        "domestic_broad": 2,
        "domestic_sector": 1,
        "overseas_equity": 1,
        "commodity": 1,
        "bond": 1,
        "other": 1,
    }
    g.category_weight_caps = {
        "domestic_broad": 0.55,
        "domestic_sector": 0.25,
        "overseas_equity": 0.30,
        "commodity": 0.25,
        "bond": 0.40,
        "other": 0.25,
    }
    g.excluded_keywords = (
        "货币",
        "现金",
        "理财",
        "分级",
        "杠杆",
        "反向",
        "REIT",
    )

    # 策略运行状态。
    g.features = {}
    g.ranks = {}
    g.eligible_count = 0
    g.target_weights = {}
    g.target_order = []
    g.entry_dates = {}
    g.high_water = {}
    g.rank_weak_days = defaultdict(int)
    g.trend_weak_days = defaultdict(int)
    g.bought_today = set()

    run_daily(
        prepare_signals,
        time="before_open",
        reference_security="000300.XSHG",
    )
    run_daily(
        execute_sells,
        time="10:00",
        reference_security="000300.XSHG",
    )
    run_daily(
        execute_buys,
        time="10:05",
        reference_security="000300.XSHG",
    )
    run_daily(
        intraday_risk_check,
        time="14:45",
        reference_security="000300.XSHG",
    )
    run_daily(
        after_market_close,
        time="after_close",
        reference_security="000300.XSHG",
    )


# =============================================================================
# ETF 分类与历史数据
# =============================================================================


def classify_etf(name):
    name = str(name or "").upper()
    if any(k in name for k in ("国债", "债券", "政金债", "信用债", "可转债", "公司债")):
        return "bond"
    if any(k in name for k in ("黄金", "白银", "原油", "豆粕", "商品", "有色", "能源化工")):
        return "commodity"
    if any(
        k in name
        for k in (
            "纳指",
            "纳斯达克",
            "标普",
            "恒生",
            "港股",
            "H股",
            "中概",
            "日经",
            "德国",
            "法国",
            "印度",
            "沙特",
            "东南亚",
        )
    ):
        return "overseas_equity"
    if any(
        k in name
        for k in (
            "沪深300",
            "上证50",
            "上证180",
            "中证500",
            "中证800",
            "中证1000",
            "中证2000",
            "A50",
            "A500",
            "深证100",
            "创业板",
            "科创50",
            "科创100",
            "红利",
        )
    ):
        return "domestic_broad"
    if "ETF" in name:
        return "domestic_sector"
    return "other"


def get_etf_universe(previous_date):
    securities = get_all_securities(types=["etf"], date=previous_date)
    result = []
    for code, row in securities.iterrows():
        name = str(row.get("display_name", row.get("name", code)))
        start_date = row.get("start_date")
        if start_date is None or pd.isna(start_date):
            info = get_security_info(code)
            start_date = info.start_date if info else None
        if start_date is None:
            continue
        listed_days = (previous_date - pd.Timestamp(start_date).date()).days
        if listed_days < g.min_listed_days:
            continue
        if any(keyword.upper() in name.upper() for keyword in g.excluded_keywords):
            continue
        result.append(
            {
                "code": code,
                "name": name,
                "category": classify_etf(name),
                "listed_days": listed_days,
            }
        )
    return result


def _normalise_batch_price(raw, chunk):
    """Convert JoinQuant multi-security get_price output to {code: frame}."""
    if raw is None or len(raw) == 0:
        return {}
    frame = raw.reset_index()
    code_col = None
    for candidate in ("code", "security"):
        if candidate in frame.columns:
            code_col = candidate
            break
    if code_col is None and len(chunk) == 1:
        frame["code"] = chunk[0]
        code_col = "code"
    if code_col is None:
        return {}
    result = {}
    for code, part in frame.groupby(code_col):
        keep = [c for c in ("close", "high", "low", "money", "volume") if c in part.columns]
        if {"close", "high", "low", "money"}.issubset(keep):
            result[str(code)] = part[keep].reset_index(drop=True)
    return result


def fetch_histories(codes, previous_date):
    """Batch-fetch completed daily bars; fall back to single-code calls."""
    histories = {}
    chunk_size = 100
    for start in range(0, len(codes), chunk_size):
        chunk = codes[start : start + chunk_size]
        try:
            raw = get_price(
                chunk,
                end_date=previous_date,
                count=g.history_days,
                frequency="daily",
                fields=["close", "high", "low", "money", "volume"],
                skip_paused=False,
                fq="pre",
                panel=False,
            )
            histories.update(_normalise_batch_price(raw, chunk))
        except Exception as error:
            log.warn("批量读取ETF历史数据失败，切换逐只读取: %s" % error)
            for code in chunk:
                try:
                    one = get_price(
                        code,
                        end_date=previous_date,
                        count=g.history_days,
                        frequency="daily",
                        fields=["close", "high", "low", "money", "volume"],
                        skip_paused=False,
                        fq="pre",
                    )
                    if one is not None and len(one) > 0:
                        histories[code] = one.reset_index(drop=True)
                except Exception:
                    continue
    return histories


# =============================================================================
# 特征、排序与组合构建
# =============================================================================


def _max_drawdown(values):
    values = np.asarray(values, dtype=float)
    running_max = np.maximum.accumulate(values)
    drawdown = values / np.where(running_max == 0, np.nan, running_max) - 1.0
    return float(abs(np.nanmin(drawdown)))


def _atr_pct(frame, days=20):
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(true_range.tail(days).mean() / close.iloc[-1])


def build_feature(meta, frame):
    if frame is None or len(frame) < g.history_days:
        return None
    frame = frame.copy()
    for col in ("close", "high", "low", "money"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["close", "high", "low", "money"])
    if len(frame) < g.history_days:
        return None

    close = frame["close"].astype(float)
    money20 = frame["money"].tail(20).astype(float)
    ma5 = float(close.tail(5).mean())
    ma10 = float(close.tail(10).mean())
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    last = float(close.iloc[-1])
    ret20 = float(last / close.iloc[-21] - 1.0)
    ret60 = float(last / close.iloc[-61] - 1.0)
    ret120 = float(last / close.iloc[-121] - 1.0)
    volatility20 = float(close.pct_change().tail(20).std(ddof=0) * np.sqrt(252))
    max_drawdown60 = _max_drawdown(close.tail(60).values)
    avg_money20 = float(money20.mean())
    min_money20 = float(money20.min())

    eligible = (
        avg_money20 >= g.min_avg_money
        and min_money20 >= g.min_daily_money
        and last > ma20 > ma60
        and ma5 > ma10
        and ret20 > 0
    )
    return {
        "code": meta["code"],
        "name": meta["name"],
        "category": meta["category"],
        "eligible": bool(eligible),
        "close": last,
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma60": ma60,
        "return20": ret20,
        "return60": ret60,
        "return120": ret120,
        "volatility20": volatility20,
        "max_drawdown60": max_drawdown60,
        "atr20_pct": _atr_pct(frame),
        "avg_money20": avg_money20,
        "returns60": close.pct_change().dropna().tail(60).values,
    }


def rank_universe(features):
    rows = [f for f in features.values() if f is not None and f["eligible"]]
    if not rows:
        return pd.DataFrame()
    ranked = pd.DataFrame(rows).set_index("code", drop=False)
    ranked["score"] = 100.0 * (
        0.35 * ranked["return20"].rank(pct=True)
        + 0.30 * ranked["return60"].rank(pct=True)
        + 0.15 * ranked["return120"].rank(pct=True)
        + 0.10 * ranked["volatility20"].rank(ascending=False, pct=True)
        + 0.10 * ranked["max_drawdown60"].rank(ascending=False, pct=True)
    )
    return ranked.sort_values(["score", "avg_money20"], ascending=False)


def _correlation(left, right):
    count = min(len(left), len(right), 60)
    if count < 30:
        return np.nan
    return float(np.corrcoef(left[-count:], right[-count:])[0, 1])


def select_targets(ranked):
    if ranked.empty:
        return [], {}
    selected = []
    category_counts = defaultdict(int)
    for code, row in ranked.iterrows():
        category = row["category"]
        if category_counts[category] >= g.category_position_limits.get(category, 1):
            continue
        too_correlated = False
        for old_code in selected:
            corr = _correlation(
                row["returns60"],
                ranked.loc[old_code, "returns60"],
            )
            if np.isfinite(corr) and corr >= g.correlation_threshold:
                too_correlated = True
                break
        if too_correlated:
            continue
        selected.append(code)
        category_counts[category] += 1
        if len(selected) >= g.max_positions:
            break

    if not selected:
        return [], {}
    selected_frame = ranked.loc[selected].copy()
    inverse_vol = 1.0 / selected_frame["volatility20"].clip(lower=0.01)
    weights = (1.0 - g.cash_reserve) * inverse_vol / inverse_vol.sum()
    weights = weights.clip(upper=g.max_position_weight)

    for category, cap in g.category_weight_caps.items():
        category_codes = [c for c in selected if ranked.loc[c, "category"] == category]
        current = float(weights.loc[category_codes].sum()) if category_codes else 0.0
        if current > cap:
            weights.loc[category_codes] *= cap / current
    return selected, {code: float(weights.loc[code]) for code in selected}


def prepare_signals(context):
    previous_date = context.previous_date
    universe = get_etf_universe(previous_date)
    codes = [item["code"] for item in universe]

    # 继续为已有持仓读取数据，即使它刚刚被移出ETF列表或筛选池。
    for code in context.portfolio.positions.keys():
        if code not in codes:
            info = get_security_info(code)
            universe.append(
                {
                    "code": code,
                    "name": info.display_name if info else code,
                    "category": classify_etf(info.display_name if info else code),
                    "listed_days": 999999,
                }
            )
            codes.append(code)

    histories = fetch_histories(codes, previous_date)
    features = {}
    for meta in universe:
        code = meta["code"]
        try:
            features[code] = build_feature(meta, histories.get(code))
        except Exception as error:
            log.debug("ETF特征计算失败 %s: %s" % (code, error))
            features[code] = None

    ranked = rank_universe(features)
    selected, target_weights = select_targets(ranked)
    g.features = features
    g.ranks = {code: idx + 1 for idx, code in enumerate(ranked.index.tolist())}
    g.eligible_count = len(ranked)
    g.target_order = selected
    g.target_weights = target_weights

    for code in context.portfolio.positions.keys():
        feature = features.get(code)
        rank = g.ranks.get(code)
        if rank is None or rank > g.rank_exit:
            g.rank_weak_days[code] += 1
        else:
            g.rank_weak_days[code] = 0
        if feature is None or feature["close"] < feature["ma20"]:
            g.trend_weak_days[code] += 1
        else:
            g.trend_weak_days[code] = 0

    log.info(
        "ETF池%d只，趋势合格%d只，目标: %s"
        % (len(universe), len(ranked), ", ".join(selected) if selected else "现金")
    )


# =============================================================================
# 卖出、买入与盘中风控
# =============================================================================


def _holding_days(context, code):
    entry_date = g.entry_dates.get(code)
    if entry_date is None:
        return 0
    try:
        return max(0, len(get_trade_days(start_date=entry_date, end_date=context.previous_date)) - 1)
    except Exception:
        return 0


def _position_closeable(position):
    return int(getattr(position, "closeable_amount", position.total_amount)) > 0


def _exit_reason(context, code, position, price, intraday_only=False):
    if not _position_closeable(position):
        return None
    cost = float(position.avg_cost)
    if cost <= 0 or price <= 0:
        return None
    feature = g.features.get(code)
    if feature is None:
        return None

    high = max(float(g.high_water.get(code, price)), price, cost)
    g.high_water[code] = high
    atr_pct = float(feature.get("atr20_pct", 0.02))
    if not np.isfinite(atr_pct):
        atr_pct = 0.02

    pnl = price / cost - 1.0
    hard_stop = float(np.clip(2.5 * atr_pct, 0.03, 0.08))
    if pnl <= -hard_stop:
        return "动态止损：收益%.1f%% <= -%.1f%%" % (pnl * 100, hard_stop * 100)

    peak_return = high / cost - 1.0
    giveback = price / high - 1.0
    trailing_stop = float(np.clip(2.0 * atr_pct, 0.025, 0.10))
    if peak_return >= 0.02 and giveback <= -trailing_stop:
        return "移动止盈：最高盈利%.1f%%后回撤%.1f%%" % (peak_return * 100, giveback * 100)

    if intraday_only:
        return None

    holding_days = _holding_days(context, code)
    if g.trend_weak_days.get(code, 0) >= 2 or feature["ma5"] < feature["ma10"]:
        return "趋势退出：连续跌破MA20或MA5下穿MA10"
    if holding_days >= g.min_hold_days and g.rank_weak_days.get(code, 0) >= 2:
        return "相对弱势：连续跌出综合排名前%d" % g.rank_exit
    median_rank = max(1, int(np.ceil(g.eligible_count / 2.0)))
    current_rank = g.ranks.get(code, 999999)
    if holding_days >= g.time_stop_days and peak_return < 0.015 and current_rank > median_rank:
        return "时间止损：持有%d日仍未形成有效趋势" % holding_days
    return None


def _valid_live_price(code, current_data):
    data = current_data[code]
    price = float(data.last_price)
    if data.paused or not np.isfinite(price) or price <= 0:
        return None
    if price >= float(data.high_limit) - 1e-4 or price <= float(data.low_limit) + 1e-4:
        return None
    return price


def execute_sells(context):
    current_data = get_current_data()
    for code, position in list(context.portfolio.positions.items()):
        try:
            price = _valid_live_price(code, current_data)
            if price is None:
                continue
            reason = _exit_reason(context, code, position, price, intraday_only=False)
            if reason:
                order_target(code, 0)
                log.info("卖出 %s：%s" % (code, reason))
        except Exception as error:
            log.warn("卖出检查失败 %s: %s" % (code, error))


def _passes_intraday_entry_filter(context, code, price, previous_close):
    if previous_close <= 0:
        return False
    gap = price / previous_close - 1.0
    if gap < -0.03 or gap > 0.03:
        return False
    try:
        minute = get_price(
            code,
            end_date=context.current_dt,
            count=30,
            frequency="1m",
            fields=["close", "volume"],
            skip_paused=False,
            fq="pre",
        )
        if minute is None or len(minute) < 10:
            return False
        volume = minute["volume"].astype(float)
        if volume.sum() <= 0:
            return False
        vwap = float((minute["close"].astype(float) * volume).sum() / volume.sum())
        return 0.98 * vwap <= price <= 1.015 * vwap
    except Exception:
        return False


def execute_buys(context):
    if not g.target_order:
        return
    current_data = get_current_data()
    held = set(context.portfolio.positions.keys())
    slots = g.max_positions - len(held)
    if slots <= 0:
        return

    for code in g.target_order:
        if slots <= 0:
            break
        if code in held or code in g.bought_today:
            continue
        try:
            price = _valid_live_price(code, current_data)
            feature = g.features.get(code)
            if price is None or feature is None:
                continue
            if not _passes_intraday_entry_filter(context, code, price, feature["close"]):
                continue
            target_value = context.portfolio.total_value * g.target_weights[code]
            target_value = min(target_value, context.portfolio.available_cash * 0.98)
            if target_value < price * 100 + 5:
                continue
            order = order_target_value(code, target_value)
            if order is not None:
                g.bought_today.add(code)
                g.entry_dates.setdefault(code, context.current_dt.date())
                g.high_water[code] = price
                slots -= 1
                log.info("买入 %s，目标仓位 %.1f%%" % (code, 100 * g.target_weights[code]))
        except Exception as error:
            log.warn("买入失败 %s: %s" % (code, error))


def intraday_risk_check(context):
    current_data = get_current_data()
    for code, position in list(context.portfolio.positions.items()):
        # 首版统一禁止当日回转，即使该ETF本身支持T+0。
        if code in g.bought_today:
            continue
        try:
            price = _valid_live_price(code, current_data)
            if price is None:
                continue
            reason = _exit_reason(context, code, position, price, intraday_only=True)
            if reason:
                order_target(code, 0)
                log.info("盘中风控卖出 %s：%s" % (code, reason))
        except Exception as error:
            log.warn("盘中风控失败 %s: %s" % (code, error))


def after_market_close(context):
    current_codes = {
        code
        for code, position in context.portfolio.positions.items()
        if position.total_amount > 0
    }
    for code in current_codes:
        position = context.portfolio.positions[code]
        g.entry_dates.setdefault(code, context.current_dt.date())
        g.high_water[code] = max(
            float(g.high_water.get(code, position.price)),
            float(position.price),
        )

    tracked = set(g.entry_dates.keys()) | set(g.high_water.keys())
    for code in tracked - current_codes:
        g.entry_dates.pop(code, None)
        g.high_water.pop(code, None)
        g.rank_weak_days.pop(code, None)
        g.trend_weak_days.pop(code, None)
    g.bought_today = set()
