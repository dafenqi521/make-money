"""Buy The Dip 策略回测 —— 对比不同补仓方式，找收益最高的一种。

策略：最多买10次，每次跌了就补，只要赚了就卖出。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.data.fetcher import fetch_etf_hist

# ---------------------------------------------------------------------------
# 测试 ETF
# ---------------------------------------------------------------------------
ETF_POOL = [
    ("513100", "纳指ETF", "nasdaq"),
    ("159941", "纳指ETF", "nasdaq"),
    ("510300", "沪深300ETF", "domestic"),
    ("510050", "上证50ETF", "domestic"),
    ("510500", "中证500ETF", "domestic"),
]

# ---------------------------------------------------------------------------
# 策略参数
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 100_000       # 初始资金
MAX_BUYS = 10                   # 最多买10次
COMMISSION_RATE = 0.0003        # 佣金万三（ETF免印花税）
SLIPPAGE = 0.001                # 滑点 0.1%
MIN_PROFIT = 1.001              # 只要赚就卖（覆盖手续费后微利即可）

# ---------------------------------------------------------------------------
# 补仓方式定义
# ---------------------------------------------------------------------------

@dataclass
class DipMethod:
    """一种补仓方式"""
    name: str                    # 显示名称
    key: str                     # 简短标识
    drop_threshold: float        # 跌多少触发补仓（如 0.03 = 3%）
    amount_type: str             # "equal" | "pyramid" | "double"
    first_buy_pct: float = 0.1   # 首次买入占总资金比例


def get_amounts(method: DipMethod, remaining_capital: float, buy_count: int) -> float:
    """根据补仓方式计算本次买入金额。

    首次买入使用 first_buy_pct；后续买入根据 amount_type 分配剩余资金。
    """
    if buy_count == 0:
        return remaining_capital * method.first_buy_pct

    # 剩余可买次数
    remaining_buys = MAX_BUYS - buy_count
    if remaining_buys <= 0:
        return remaining_capital

    if method.amount_type == "equal":
        return remaining_capital / max(remaining_buys, 1)

    elif method.amount_type == "pyramid":
        # 金字塔：第i次买入权重 = i+1（i从0开始）
        # 剩余分配：当前权重 / 剩余所有权重之和
        weights = [i + 1 for i in range(buy_count, MAX_BUYS)]
        total_weight = sum(weights)
        current_weight = buy_count + 1
        return remaining_capital * current_weight / total_weight

    elif method.amount_type == "double":
        # 倍数递增：第i次买入 = 2^(i) * base
        # 剩余分配：当前权重 / 剩余所有权重之和
        weights = [2 ** i for i in range(buy_count, MAX_BUYS)]
        total_weight = sum(weights)
        current_weight = 2 ** buy_count
        return remaining_capital * current_weight / total_weight

    else:
        return remaining_capital / max(remaining_buys, 1)


# ---------------------------------------------------------------------------
# 补仓方式列表
# ---------------------------------------------------------------------------
METHODS = [
    DipMethod("等额补仓(跌3%触发)",     "equal_3pct",    0.03, "equal"),
    DipMethod("金字塔补仓(跌3%触发)",   "pyramid_3pct",  0.03, "pyramid"),
    DipMethod("倍数补仓(跌3%触发)",     "double_3pct",   0.03, "double"),
    DipMethod("等额补仓(跌2%触发)",     "equal_2pct",    0.02, "equal"),
    DipMethod("等额补仓(跌5%触发)",     "equal_5pct",    0.05, "equal"),
    DipMethod("金字塔补仓(跌5%触发)",   "pyramid_5pct",  0.05, "pyramid"),
    DipMethod("等额补仓(跌7%触发)",     "equal_7pct",    0.07, "equal"),
    DipMethod("金字塔补仓(跌7%触发)",   "pyramid_7pct",  0.07, "pyramid"),
]

# ---------------------------------------------------------------------------
# 回测核心
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    date: str
    action: str           # "buy" | "sell"
    price: float
    shares: int
    amount: float
    buy_count: int
    cost_basis: float
    profit_pct: float


@dataclass
class BacktestResult:
    etf_code: str
    etf_name: str
    method_name: str
    method_key: str
    total_return: float         # 总收益率
    total_profit: float         # 总利润金额
    final_capital: float
    total_trades: int           # 总交易次数
    buy_count_total: int        # 总买入次数
    sell_count: int             # 总卖出次数（完整轮次）
    win_rate: float             # 胜率（盈利卖出/总卖出）
    max_drawdown: float         # 最大回撤（持仓期间）
    avg_holding_days: float     # 平均持仓天数
    trades: list = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    method: DipMethod,
    etf_code: str,
    etf_name: str,
) -> BacktestResult:
    """对给定 ETF 和补仓方式运行回测。"""

    # 按日期升序排列
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{etf_code} 无数据")

    capital = float(INITIAL_CAPITAL)
    cash = capital
    shares = 0
    cost_basis = 0.0          # 加权平均成本
    last_buy_price = 0.0
    buy_count = 0             # 本轮买入次数
    total_buy_count = 0
    sell_count = 0
    win_count = 0
    trades = []
    equity_peak = capital
    max_drawdown = 0.0
    holding_days_list = []
    entry_date = None

    # 分配首笔买入金额（后续用于等额）
    initial_buy_amount = capital * method.first_buy_pct

    for i, row in df.iterrows():
        price = float(row["close"])
        current_date = row["date"]

        # --- 持有中 ---
        if shares > 0:
            # 计算当前盈亏
            current_value = shares * price
            profit_rate = (current_value - cost_basis * shares) / (cost_basis * shares) if cost_basis > 0 else 0

            # 更新最大回撤
            equity = cash + current_value
            if equity > equity_peak:
                equity_peak = equity
            drawdown = (equity_peak - equity) / equity_peak if equity_peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            # 只要赚就卖出
            if profit_rate > 0:
                sell_amount = shares * price * (1 - SLIPPAGE)
                commission = sell_amount * COMMISSION_RATE
                cash += sell_amount - commission
                profit = cash - capital  # 本轮利润

                trades.append(TradeRecord(
                    date=str(current_date),
                    action="sell",
                    price=price,
                    shares=shares,
                    amount=sell_amount,
                    buy_count=buy_count,
                    cost_basis=cost_basis,
                    profit_pct=profit_rate * 100,
                ))

                if profit > 0:
                    win_count += 1
                sell_count += 1
                if entry_date:
                    holding_days = (pd.to_datetime(current_date) - pd.to_datetime(entry_date)).days
                    holding_days_list.append(holding_days)

                # 重置
                shares = 0
                cost_basis = 0.0
                last_buy_price = 0.0
                buy_count = 0
                entry_date = None
                capital = cash  # 更新基准

            # 补仓检查：跌了触发
            elif last_buy_price > 0 and buy_count < MAX_BUYS:
                drop_from_last = (price - last_buy_price) / last_buy_price
                if drop_from_last <= -method.drop_threshold:
                    # 补仓
                    buy_amount = min(get_amounts(method, cash, buy_count), cash)
                    commission = buy_amount * COMMISSION_RATE
                    actual_buy = buy_amount - commission
                    buy_shares = int(actual_buy / price / 100) * 100  # 100股整

                    if buy_shares >= 100:
                        buy_cost = buy_shares * price * (1 + SLIPPAGE) + buy_shares * price * COMMISSION_RATE
                        if buy_cost <= cash:
                            cash -= buy_cost
                            total_cost_basis = cost_basis * shares + buy_cost
                            shares += buy_shares
                            cost_basis = total_cost_basis / shares if shares > 0 else 0
                            last_buy_price = price
                            buy_count += 1
                            total_buy_count += 1

                            trades.append(TradeRecord(
                                date=str(current_date),
                                action="buy",
                                price=price,
                                shares=buy_shares,
                                amount=buy_cost,
                                buy_count=buy_count,
                                cost_basis=cost_basis,
                                profit_pct=0,
                            ))

        # --- 空仓中：寻找入场点 ---
        else:
            # 简单入场：首次触及即买（实际可加更多条件如均线等）
            # 这里用 "连续两天收阴后入场" 作为简单过滤
            if i >= 2:
                prev_close_1 = float(df.iloc[i - 1]["close"])
                prev_close_2 = float(df.iloc[i - 2]["close"])
                # 连续下跌后入场
                if price < prev_close_1 < prev_close_2:
                    buy_amount = min(initial_buy_amount, cash)
                    commission = buy_amount * COMMISSION_RATE
                    actual_buy = buy_amount - commission
                    buy_shares = int(actual_buy / price / 100) * 100

                    if buy_shares >= 100:
                        buy_cost = buy_shares * price * (1 + SLIPPAGE) + buy_shares * price * COMMISSION_RATE
                        if buy_cost <= cash:
                            cash -= buy_cost
                            shares = buy_shares
                            cost_basis = buy_cost / shares
                            last_buy_price = price
                            buy_count = 1
                            total_buy_count += 1
                            entry_date = current_date

                            trades.append(TradeRecord(
                                date=str(current_date),
                                action="buy",
                                price=price,
                                shares=buy_shares,
                                amount=buy_cost,
                                buy_count=buy_count,
                                cost_basis=cost_basis,
                                profit_pct=0,
                            ))

    # --- 回测结束，仍有持仓 ---
    final_value = cash + shares * df.iloc[-1]["close"] * (1 - SLIPPAGE)
    if shares > 0:
        final_commission = shares * df.iloc[-1]["close"] * COMMISSION_RATE
        final_value -= final_commission

    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
    total_profit = final_value - INITIAL_CAPITAL
    win_rate = win_count / sell_count if sell_count > 0 else 0.0
    avg_holding = np.mean(holding_days_list) if holding_days_list else 0.0

    return BacktestResult(
        etf_code=etf_code,
        etf_name=etf_name,
        method_name=method.name,
        method_key=method.key,
        total_return=total_return,
        total_profit=total_profit,
        final_capital=final_value,
        total_trades=len(trades),
        buy_count_total=total_buy_count,
        sell_count=sell_count,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        avg_holding_days=avg_holding,
        trades=trades,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    end_date = date.today()
    start_date = end_date - timedelta(days=730)  # 约2年
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    print("=" * 80)
    print(f"  Buy The Dip 策略回测")
    print(f"  回测区间: {start_date} ~ {end_date} (约2年)")
    print(f"  初始资金: ￥{INITIAL_CAPITAL:,.0f}  |  最多买入: {MAX_BUYS}次  |  赚就卖")
    print("=" * 80)

    all_results = []

    for code, name, category in ETF_POOL:
        print(f"\n{'─' * 80}")
        print(f"  [{category.upper()}] {code} {name}")
        print(f"{'─' * 80}")

        # 获取历史数据
        try:
            df = fetch_etf_hist(code, start_date=start_str, end_date=end_str)
            print(f"  数据: {len(df)} 个交易日  |  "
                  f"区间: {df['date'].min()} ~ {df['date'].max()}  |  "
                  f"价格: ￥{df['close'].min():.3f} ~ ￥{df['close'].max():.3f}")
        except Exception as e:
            print(f"  ❌ 数据获取失败: {e}")
            continue

        etf_results = []
        for method in METHODS:
            try:
                result = run_backtest(df, method, code, name)
                etf_results.append(result)
            except Exception as e:
                print(f"  ⚠ {method.name}: 回测异常 - {e}")

        if not etf_results:
            continue

        # 按收益率排序
        etf_results.sort(key=lambda r: r.total_return, reverse=True)

        print(f"\n  {'补仓方式':<30} {'收益率':>8} {'利润':>12} {'交易':>5} {'胜率':>7} {'最大回撤':>8} {'均持天':>6}")
        print(f"  {'─' * 30} {'─' * 8} {'─' * 12} {'─' * 5} {'─' * 7} {'─' * 8} {'─' * 6}")

        for r in etf_results:
            marker = "🏆" if r == etf_results[0] else "  "
            print(f"{marker} {r.method_name:<28} {r.total_return:>7.1%} "
                  f"￥{r.total_profit:>10,.0f} {r.total_trades:>4} "
                  f"{r.win_rate:>6.0%} {r.max_drawdown:>7.1%} {r.avg_holding_days:>5.0f}d")

        all_results.extend(etf_results)

    # --- 综合排名 ---
    print(f"\n{'═' * 80}")
    print(f"  📊 跨 ETF 综合排名 TOP 10（按收益率）")
    print(f"{'═' * 80}")
    all_results.sort(key=lambda r: r.total_return, reverse=True)

    print(f"\n  {'ETF':<20} {'补仓方式':<30} {'收益率':>8} {'利润':>12} {'胜率':>7} {'最大回撤':>8}")
    print(f"  {'─' * 20} {'─' * 30} {'─' * 8} {'─' * 12} {'─' * 7} {'─' * 8}")
    for r in all_results[:10]:
        label = f"{r.etf_code} {r.etf_name}"
        print(f"  {label:<20} {r.method_name:<30} {r.total_return:>7.1%} "
              f"￥{r.total_profit:>10,.0f} {r.win_rate:>6.0%} {r.max_drawdown:>7.1%}")

    # --- 推荐 ---
    best = all_results[0]
    print(f"\n{'═' * 80}")
    print(f"  🏆 最优方案")
    print(f"  ETF: {best.etf_code} {best.etf_name}")
    print(f"  补仓方式: {best.method_name}")
    print(f"  2年收益率: {best.total_return:.1%}")
    print(f"  利润: ￥{best.total_profit:,.0f}")
    print(f"  总交易: {best.total_trades} 笔  |  胜率: {best.win_rate:.0%}")
    print(f"  最大回撤: {best.max_drawdown:.1%}  |  平均持仓: {best.avg_holding_days:.0f}天")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
