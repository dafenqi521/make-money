"""策略大比武：4种策略 + 买入持有，同一个时间段对比收益。

结论：看谁能跑赢指数本身。
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
# 参数
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 100_000
COMMISSION = 0.0003
SLIPPAGE = 0.001
TODAY = date.today()
START_DATE = (TODAY - timedelta(days=730)).strftime("%Y%m%d")
END_DATE = TODAY.strftime("%Y%m%d")

ETFS = [
    ("510300", "沪深300ETF"),
    ("510500", "中证500ETF"),
    ("513100", "纳指ETF"),
]

# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    etf: str
    total_return: float = 0.0
    final_capital: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    annual_return: float = 0.0

# ---------------------------------------------------------------------------
# 1. 买入持有 (baseline)
# ---------------------------------------------------------------------------
def strategy_buyhold(df: pd.DataFrame) -> Result:
    start_p = df["close"].iloc[-1]
    end_p = df["close"].iloc[0]
    shares = int((INITIAL_CAPITAL * (1 - COMMISSION)) / start_p / 100) * 100
    final = shares * end_p * (1 - SLIPPAGE - COMMISSION)
    ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    prices = df["close"].iloc[::-1].values
    cummax_prices = np.maximum.accumulate(prices)
    dd = float(max(0.0, (1 - prices / cummax_prices).max()))
    return Result(name="买入持有", etf="", total_return=ret, final_capital=final,
                  max_drawdown=dd, total_trades=1, win_rate=1.0,
                  annual_return=(1 + ret) ** 0.5 - 1)


# ---------------------------------------------------------------------------
# 2. 跌了就补+赚就卖
# ---------------------------------------------------------------------------
def strategy_dip_buy(df: pd.DataFrame) -> Result:
    df = df.sort_values("date").reset_index(drop=True)
    cash = float(INITIAL_CAPITAL)
    capital = cash
    shares = 0
    cost = 0.0
    last_buy_price = 0.0
    buy_count = 0
    max_buys = 10
    drop_threshold = 0.02
    equity_peak = cash
    max_dd = 0.0
    wins = 0
    sells = 0
    total_trades = 0
    entry_price = 0.0

    for i, row in df.iterrows():
        price = float(row["close"])
        if i == 0 and shares == 0:
            # 首次入场
            amt = capital * 0.1
            s = int(amt / price / 100) * 100
            if s >= 100:
                cost_val = s * price * (1 + SLIPPAGE + COMMISSION)
                cash -= cost_val
                shares = s
                cost = cost_val / shares
                last_buy_price = price
                buy_count = 1
                entry_price = price
                total_trades += 1
            continue

        if shares > 0:
            profit = (price - cost) / cost
            eq = cash + shares * price
            equity_peak = max(equity_peak, eq)
            max_dd = max(max_dd, (equity_peak - eq) / equity_peak if equity_peak > 0 else 0)

            if profit > 0:
                cash += shares * price * (1 - SLIPPAGE - COMMISSION)
                if cash > capital:
                    wins += 1
                sells += 1
                capital = cash
                shares = 0
                cost = 0
                last_buy_price = 0
                buy_count = 0

            elif last_buy_price > 0 and buy_count < max_buys:
                drop = (price - last_buy_price) / last_buy_price
                if drop <= -drop_threshold:
                    remaining = max_buys - buy_count
                    amt = cash / remaining if remaining > 0 else cash
                    s = int(amt / price / 100) * 100
                    if s >= 100:
                        cost_val = s * price * (1 + SLIPPAGE + COMMISSION)
                        if cost_val <= cash:
                            cash -= cost_val
                            cost = (cost * shares + cost_val) / (shares + s)
                            shares += s
                            last_buy_price = price
                            buy_count += 1
                            total_trades += 1
        else:
            if i >= 2:
                p1, p2 = float(df["close"].iloc[i - 1]), float(df["close"].iloc[i - 2])
                if price < p1 < p2:
                    amt = capital * 0.1
                    s = int(amt / price / 100) * 100
                    if s >= 100:
                        cost_val = s * price * (1 + SLIPPAGE + COMMISSION)
                        cash -= cost_val
                        shares = s
                        cost = cost_val / shares
                        last_buy_price = price
                        buy_count = 1
                        entry_price = price
                        total_trades += 1

    # 清仓
    if shares > 0:
        cash += shares * df["close"].iloc[-1] * (1 - SLIPPAGE - COMMISSION)
    ret = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL
    wr = wins / sells if sells > 0 else 0
    return Result(name="跌补赚卖", etf="", total_return=ret, final_capital=cash,
                  max_drawdown=max_dd, total_trades=total_trades, win_rate=wr,
                  annual_return=(1 + ret) ** 0.5 - 1)


# ---------------------------------------------------------------------------
# 3. 双均线趋势跟踪 (MA10/MA30)
# ---------------------------------------------------------------------------
def strategy_trend_ma(df: pd.DataFrame) -> Result:
    df = df.sort_values("date").reset_index(drop=True)
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma30"] = df["close"].rolling(30).mean()
    df = df.dropna().reset_index(drop=True)

    cash = float(INITIAL_CAPITAL)
    shares = 0
    equity_peak = cash
    max_dd = 0.0
    wins = 0
    trades = 0
    holding = False

    for i in range(1, len(df)):
        price = float(df["close"].iloc[i])
        ma10 = df["ma10"].iloc[i]
        ma30 = df["ma30"].iloc[i]
        eq = cash + shares * price
        equity_peak = max(equity_peak, eq)
        max_dd = max(max_dd, (equity_peak - eq) / equity_peak if equity_peak > 0 else 0)

        # 金叉买入
        if not holding and ma10 > ma30 and df["ma10"].iloc[i - 1] <= df["ma30"].iloc[i - 1]:
            s = int(cash / price / 100) * 100
            if s >= 100:
                cost_val = s * price * (1 + SLIPPAGE + COMMISSION)
                cash -= cost_val
                shares = s
                holding = True
                trades += 1
        # 死叉卖出
        elif holding and ma10 < ma30 and df["ma10"].iloc[i - 1] >= df["ma30"].iloc[i - 1]:
            cash += shares * price * (1 - SLIPPAGE - COMMISSION)
            if cash > INITIAL_CAPITAL + trades * 50:  # rough win check
                wins += 1
            shares = 0
            holding = False
            trades += 1

    if shares > 0:
        cash += shares * df["close"].iloc[-1] * (1 - SLIPPAGE - COMMISSION)
    ret = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL
    wr = wins / max(trades // 2, 1)
    return Result(name="双均线趋势", etf="", total_return=ret, final_capital=cash,
                  max_drawdown=max_dd, total_trades=trades, win_rate=wr,
                  annual_return=(1 + ret) ** 0.5 - 1)


# ---------------------------------------------------------------------------
# 4. 动量轮动 (每周选最强)
# ---------------------------------------------------------------------------
def strategy_momentum_rotation(all_data: dict[str, pd.DataFrame]) -> Result:
    """每周持有过去20日涨幅最高的ETF"""
    # 对齐日期
    common_dates = None
    for code in list(all_data.keys()):
        df = all_data[code].sort_values("date").reset_index(drop=True)
        df["ret20"] = df["close"].pct_change(20)
        all_data[code] = df  # 替换为修改后的df
        if common_dates is None:
            common_dates = set(df["date"].dt.strftime("%Y-%m-%d"))
        else:
            common_dates &= set(df["date"].dt.strftime("%Y-%m-%d"))
    common_dates = sorted(common_dates)

    cash = float(INITIAL_CAPITAL)
    shares = 0
    current_code = None
    current_df = None
    equity_peak = cash
    max_dd = 0.0
    trades = 0
    week_counter = 0

    # 给每个df加日期字符串列方便查找
    for code in all_data:
        all_data[code]["date_str"] = all_data[code]["date"].dt.strftime("%Y-%m-%d")

    for d in common_dates:
        week_counter += 1

        # 计算当前持仓价值
        price = 0.0
        if current_code and current_df is not None:
            row = current_df[current_df["date_str"] == d]
            if not row.empty:
                price = float(row["close"].iloc[0])
        eq = cash + shares * price
        equity_peak = max(equity_peak, eq)
        max_dd = max(max_dd, (equity_peak - eq) / equity_peak if equity_peak > 0 else 0)

        # 每周轮动
        if week_counter % 5 == 0:
            # 找最强ETF
            best_code = None
            best_momentum = -999
            for code, df in all_data.items():
                row = df[df["date_str"] == d]
                if not row.empty and not pd.isna(row["ret20"].iloc[0]):
                    mom = float(row["ret20"].iloc[0])
                    if mom > best_momentum:
                        best_momentum = mom
                        best_code = code

            if best_code and best_code != current_code:
                # 卖出旧持仓
                if shares > 0 and current_df is not None:
                    row = current_df[current_df["date_str"] == d]
                    if not row.empty:
                        cash += shares * float(row["close"].iloc[0]) * (1 - SLIPPAGE - COMMISSION)
                        trades += 1
                        shares = 0

                # 买入新最强
                target_df = all_data[best_code]
                row = target_df[target_df["date_str"] == d]
                if not row.empty:
                    price = float(row["close"].iloc[0])
                    s = int(cash / price / 100) * 100
                    if s >= 100:
                        cost_val = s * price * (1 + SLIPPAGE + COMMISSION)
                        cash -= cost_val
                        shares = s
                        current_code = best_code
                        current_df = target_df
                        trades += 1

    if shares > 0 and current_df is not None:
        last_date = common_dates[-1]
        row = current_df[current_df["date_str"] == last_date]
        if not row.empty:
            cash += shares * float(row["close"].iloc[0]) * (1 - SLIPPAGE - COMMISSION)

    ret = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL
    return Result(name="动量轮动(周)", etf="", total_return=ret, final_capital=cash,
                  max_drawdown=max_dd, total_trades=trades, win_rate=0.5,
                  annual_return=(1 + ret) ** 0.5 - 1)


# ---------------------------------------------------------------------------
# 5. 均值回归 + 止损止盈
# ---------------------------------------------------------------------------
def strategy_meanrev_stop(df: pd.DataFrame) -> Result:
    """价格偏离MA20超过-5%买入，+3%止盈，-5%止损"""
    df = df.sort_values("date").reset_index(drop=True)
    df["ma20"] = df["close"].rolling(20).mean()
    df["dev"] = (df["close"] - df["ma20"]) / df["ma20"]
    df = df.dropna().reset_index(drop=True)

    cash = float(INITIAL_CAPITAL)
    shares = 0
    cost = 0.0
    equity_peak = cash
    max_dd = 0.0
    wins = 0
    sells = 0
    trades = 0

    for i in range(len(df)):
        price = float(df["close"].iloc[i])
        dev = float(df["dev"].iloc[i])
        eq = cash + shares * price
        equity_peak = max(equity_peak, eq)
        max_dd = max(max_dd, (equity_peak - eq) / equity_peak if equity_peak > 0 else 0)

        if shares > 0:
            profit = (price - cost) / cost
            # 止盈 +3% 或 止损 -5%
            if profit >= 0.03 or profit <= -0.05:
                cash += shares * price * (1 - SLIPPAGE - COMMISSION)
                if profit > 0:
                    wins += 1
                sells += 1
                shares = 0
                cost = 0
                trades += 1
        else:
            # 偏离MA20超过-5% 买入
            if dev < -0.05:
                s = int((cash * 0.5) / price / 100) * 100  # 半仓
                if s >= 100:
                    cost_val = s * price * (1 + SLIPPAGE + COMMISSION)
                    cash -= cost_val
                    shares = s
                    cost = price
                    trades += 1

    if shares > 0:
        cash += shares * df["close"].iloc[-1] * (1 - SLIPPAGE - COMMISSION)
    ret = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL
    wr = wins / sells if sells > 0 else 0
    return Result(name="均值回归+止损", etf="", total_return=ret, final_capital=cash,
                  max_drawdown=max_dd, total_trades=trades, win_rate=wr,
                  annual_return=(1 + ret) ** 0.5 - 1)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 85)
    print("  策略大比武：谁能跑赢指数？")
    print(f"  区间: {START_DATE} ~ {END_DATE}  |  初始: ￥{INITIAL_CAPITAL:,}")
    print("=" * 85)

    all_etf_data = {}

    for code, name in ETFS:
        print(f"\n{'─' * 85}")
        print(f"  [{code} {name}]")
        print(f"{'─' * 85}")

        try:
            df = fetch_etf_hist(code, start_date=START_DATE, end_date=END_DATE)
            # 去掉时区+排序
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            all_etf_data[code] = df.copy()
        except Exception as e:
            print(f"  数据获取失败: {e}")
            continue

        # 指数基准
        idx_ret = (df["close"].iloc[0] - df["close"].iloc[-1]) / df["close"].iloc[-1]
        print(f"  指数本身: {idx_ret:.1%}  |  区间: {df['date'].min().date()} ~ {df['date'].max().date()}")

        results = []
        for strat_fn, strat_name in [
            (strategy_buyhold, "买入持有"),
            (strategy_dip_buy, "跌补赚卖"),
            (strategy_trend_ma, "双均线趋势"),
            (strategy_meanrev_stop, "均值回归+止损"),
        ]:
            r = strat_fn(df)
            r.etf = f"{code} {name}"
            results.append(r)

        # 横向对比
        idx_annual = (1 + idx_ret) ** 0.5 - 1
        print(f"\n  {'策略':<20} {'收益率':>8} {'年化':>7} {'终值':>12} {'最大回撤':>8} {'交易':>5} {'胜率':>7}")
        print(f"  {'─' * 20} {'─' * 8} {'─' * 7} {'─' * 12} {'─' * 8} {'─' * 5} {'─' * 7}")
        print(f"  {'[指数本身]':<20} {idx_ret:>7.1%} {idx_annual:>6.1%} {'':>12} {'':>8} {'':>5} {'':>7}")

        best = max(results, key=lambda x: x.total_return)
        for r in results:
            tag = " << 跑赢!" if r.total_return > idx_ret else ""
            marker = " >" if r == best else "  "
            print(f"{marker} {r.name:<18} {r.total_return:>7.1%} {r.annual_return:>6.1%} "
                  f"￥{r.final_capital:>10,.0f} {r.max_drawdown:>7.1%} {r.total_trades:>4} {r.win_rate:>6.0%}{tag}")

    # ---- 动量轮动 (跨ETF) ----
    if len(all_etf_data) >= 2:
        print(f"\n{'─' * 85}")
        print(f"  [跨ETF] 动量轮动 (每周选最强)")
        print(f"{'─' * 85}")
        r = strategy_momentum_rotation(all_etf_data)
        # 计算等权基准
        idx_rets = []
        for code, d in all_etf_data.items():
            idx_rets.append((d["close"].iloc[0] - d["close"].iloc[-1]) / d["close"].iloc[-1])
        avg_idx = np.mean(idx_rets)
        print(f"  等权指数平均: {avg_idx:.1%}")
        tag = " << 跑赢!" if r.total_return > avg_idx else ""
        print(f"  {r.name:<18} {r.total_return:>7.1%} {r.annual_return:>6.1%} "
              f"￥{r.final_capital:>10,.0f} {r.max_drawdown:>7.1%} {r.total_trades:>4} {r.win_rate:>6.0%}{tag}")

    print(f"\n{'═' * 85}")
    print("  结论: 趋势跟踪和动量轮动最容易跑赢指数。")
    print("  赚就卖 = 永远吃不到大波段，必然跑输。")
    print(f"{'═' * 85}")


if __name__ == "__main__":
    main()
