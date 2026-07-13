"""Short-term momentum strategy — comprehensive backtest across all 25 ETFs.

Tests:
  1. Single-ETF backtest — per-ETF win rate & returns
  2. Multi-asset rotation backtest — the real strategy
  3. Parameter sensitivity analysis
  4. Comparison with existing strategies (UltraShort, FastBand, ShortTermBand)
  5. Candidate scanning test
  6. Scoring boundary tests
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data.fetcher import fetch_etf_hist, fetch_etf_info
from src.engine.backtest import BacktestEngine
from src.strategy.short_term_momentum import (
    ShortTermMomentumStrategy,
    CANDIDATE_ETFS,
    CANDIDATE_STOCKS,
    run_multi_asset_backtest,
    _compute_rsi,
    _daily_return_pct,
    _compute_drawdown_1y,
    _compute_return_1y,
    _compute_ma_slope,
)
from src.strategy.ultra_short import UltraShortStrategy
from src.strategy.short_term_band import ShortTermBandStrategy
from src.strategy.fast_band_4pct import FastBand4PctStrategy
from src.strategy.registry import get_registry

# ═════════════════════════════════════════════════════════════════════════════
# 0. Smoke tests — static helpers & strategy metadata
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("短线动量策略 — 全面回测验证")
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 80)

print("\n[0] 冒烟测试 — 静态函数 & 策略元数据")
print("-" * 60)

# Static helpers
test_closes = np.array([10.0, 10.2, 10.5, 10.3, 10.1, 10.4, 10.8, 11.0])
dd = _compute_drawdown_1y(test_closes)
ret = _compute_return_1y(test_closes)
slope = _compute_ma_slope(np.tile(test_closes, 30))
print(f"  _compute_drawdown_1y: {dd:.3f}")
print(f"  _compute_return_1y:   {ret:.3f}")
print(f"  _compute_ma_slope:    {slope:.3f}")

# RSI
rsi_arr = _compute_rsi(np.tile(test_closes, 3), 14)
print(f"  _compute_rsi:         last={rsi_arr[-1]:.1f}" if not np.isnan(rsi_arr[-1]) else "  _compute_rsi:         nan (too few bars)")

# Daily return
dr = _daily_return_pct(test_closes, 2)
print(f"  _daily_return_pct:    {dr:.3f}")

# Strategy metadata
strategy = ShortTermMomentumStrategy()
assert strategy.name == "短线动量", f"Name mismatch: {strategy.name}"
params = strategy.get_default_params()
assert len(params) >= 30, f"Expected >=30 params, got {len(params)}"
assert params["min_prev_day_change"] == 0.02
assert params["take_profit_pct"] == 0.04
assert params["stop_loss_pct"] == 0.03
assert params["max_hold_days"] == 4
assert params["min_mcap_yi_stock"] == 50.0
assert params["min_mcap_yi_fund"] == 10.0
assert params["max_drawdown_1y"] == 0.20
assert params["min_return_1y"] == 2.0

# Param descriptions
descs = strategy.get_param_descriptions()
assert len(descs) == len(params), f"Descriptions {len(descs)} != params {len(params)}"
for key in params:
    assert key in descs, f"Missing description for param: {key}"
    d = descs[key]
    assert "label" in d, f"Missing label for {key}"
    assert "help" in d, f"Missing help for {key}"

print(f"  ✅ 策略名称: {strategy.name}")
print(f"  ✅ 参数数量: {len(params)}")
print(f"  ✅ 参数描述: {len(descs)}项全部覆盖")
print(f"  ✅ 所有冒烟测试通过")

# Registry check
registry = get_registry()
assert "短线动量" in registry, f"Strategy not in registry! Available: {registry.get_names()}"
print(f"  ✅ 已注册到策略注册表")

# ═════════════════════════════════════════════════════════════════════════════
# 1. Single-ETF backtest
# ═════════════════════════════════════════════════════════════════════════════

def test_single_etf(code, name, capital=50000):
    """Run single-ETF backtest with ShortTermMomentumStrategy."""
    try:
        df = fetch_etf_hist(code)
        if df is None or df.empty or len(df) < 30:
            return None
    except Exception:
        return None

    strategy = ShortTermMomentumStrategy()
    engine = BacktestEngine(initial_capital=capital)
    try:
        result = engine.run(df, strategy, backtest_capital=capital)
    except Exception as e:
        return None

    return {
        "code": code,
        "name": name,
        "total_return": result.total_return,
        "annual_return": result.annual_return,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "winning": result.winning_trades,
        "losing": result.losing_trades,
        "max_dd": result.max_drawdown,
        "sharpe": result.sharpe_ratio,
        "avg_hold": result.avg_holding_days,
    }


print(f"\n[1] 单ETF回测（25只宽基ETF）")
print("-" * 80)

results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    future_map = {
        executor.submit(test_single_etf, e["code"], e["name"]): e["code"]
        for e in CANDIDATE_ETFS
    }
    for future in as_completed(future_map):
        r = future.result()
        if r:
            results.append(r)
            print(f"  {r['code']} {r['name']:<20s} "
                  f"交易{r['total_trades']:>3}笔  胜率{r['win_rate']:>5.0%}  "
                  f"年化{r['annual_return']:>+6.1%}  回撤{r['max_dd']:>6.1%}  "
                  f"持有{r['avg_hold']:>4.1f}天")

if results:
    valid = [r for r in results if r["total_trades"] > 0]
    if valid:
        avg_wr = np.mean([r["win_rate"] for r in valid])
        total_w = sum(r["winning"] for r in valid)
        total_l = sum(r["losing"] for r in valid)
        overall_wr = total_w / (total_w + total_l) if (total_w + total_l) > 0 else 0
        total_t = sum(r["total_trades"] for r in valid)
        avg_hold = np.mean([r["avg_hold"] for r in valid])
        print(f"\n  📈 单ETF汇总: {len(valid)}只ETF有交易, 共{total_t}笔, "
              f"整体胜率{overall_wr:.1%}, 平均胜率{avg_wr:.1%}, 平均持有{avg_hold:.1f}天")
    else:
        print(f"\n  ⚠️ 所有ETF均无交易信号（动量策略在近期市场环境下信号较稀缺）")

# ═════════════════════════════════════════════════════════════════════════════
# 2. Multi-asset rotation backtest
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n[2] 多标的轮动回测（核心策略）")
print("-" * 80)

param_sets = [
    {"label": "默认参数", "params": {}},
    {
        "label": "宽松动量（涨幅1%+低门槛）",
        "params": {"min_prev_day_change": 0.01, "entry_score_threshold": 50, "min_return_1y": 0.5},
    },
    {
        "label": "严格精选（涨幅3%+高门槛）",
        "params": {"min_prev_day_change": 0.03, "entry_score_threshold": 70},
    },
    {
        "label": "快进快出（2天持有+大止盈）",
        "params": {"max_hold_days": 2, "take_profit_pct": 0.06, "stop_loss_pct": 0.02},
    },
]

for ps in param_sets:
    label = ps["label"]
    extra_params = ps["params"]
    print(f"\n  ── {label} ──")

    bt_result = run_multi_asset_backtest(
        initial_capital=50000,
        asset_mode="etf",
        **extra_params,
    )

    if "error" in bt_result:
        print(f"    ❌ {bt_result['error']}")
        continue

    m = bt_result["metrics"]
    print(f"    总收益: {m['total_return']:+.1%}  |  年化: {m['annual_return']:+.1%}")
    print(f"    胜率: {m['win_rate']:.0%} ({m['winning_trades']}W/{m['losing_trades']}L)")
    print(f"    平均盈利: {m['avg_win_pct']:+.2%}  |  平均亏损: {m['avg_loss_pct']:+.2%}")
    print(f"    最大回撤: {m['max_drawdown']:.1%}  |  Sharpe: {m['sharpe_ratio']:.2f}")
    print(f"    周交易: {m['trades_per_week']:.1f}次  |  平均持有: {m['avg_holding_days']:.1f}天")
    print(f"    盈亏比: {m['profit_factor']:.2f}  |  期望值: {m['expectancy_pct']:+.2%}/笔")
    print(f"    预估周收益: {m['weekly_return_est']:+.2%}")

    trades = bt_result.get("trades", [])
    if trades:
        wins = [t for t in trades if t["winning"]]
        losses = [t for t in trades if not t["winning"]]
        top_wins = sorted(wins, key=lambda x: -x["pnl_pct"])[:3]
        top_losses = sorted(losses, key=lambda x: x["pnl_pct"])[:3]
        win_strs = [f"{t['pnl_pct']:+.1%}({t['code']} {t['holding_days']}d)" for t in top_wins]
        loss_strs = [f"{t['pnl_pct']:+.1%}({t['code']} {t['exit_reason']})" for t in top_losses]
        if win_strs:
            print(f"    Top3盈利: {', '.join(win_strs)}")
        if loss_strs:
            print(f"    Top3亏损: {', '.join(loss_strs)}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. Strategy comparison
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n[3] 策略对比（510300 沪深300ETF）")
print("-" * 80)

try:
    df_compare = fetch_etf_hist("510300")
    n_bars = len(df_compare) if df_compare is not None else 0
    print(f"  使用 510300 沪深300ETF: {n_bars}根K线")

    strategies = [
        ShortTermMomentumStrategy(),
        UltraShortStrategy(),
        ShortTermBandStrategy(),
        FastBand4PctStrategy(),
    ]

    for strat in strategies:
        engine = BacktestEngine(initial_capital=50000)
        try:
            result = engine.run(df_compare, strat, backtest_capital=50000, use_pe_filter=False)
            print(f"  {strat.name:<10s} | "
                  f"交易{result.total_trades:>3}笔 | "
                  f"胜率{result.win_rate:>5.0%} | "
                  f"年化{result.annual_return:>+6.1%} | "
                  f"回撤{result.max_drawdown:>6.1%} | "
                  f"Sharpe{result.sharpe_ratio:>5.2f} | "
                  f"持有{result.avg_holding_days:>4.1f}天")
        except Exception as e:
            print(f"  {strat.name:<10s} | ❌ 错误: {e}")
except Exception as e:
    print(f"  ❌ 无法获取 510300 数据: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# 4. Candidate scanning test
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n[4] 候选扫描测试")
print("-" * 80)

try:
    scanned = ShortTermMomentumStrategy.scan_candidates(
        asset_mode="etf",
        top_n=10,
        min_return_1y=0.3,   # Lower bar for practical scanning
        entry_score_threshold=40,
    )
    print(f"  扫描结果: {len(scanned)}只候选ETF")

    for s in scanned:
        action_icon = "✅" if s["passed"] else ("👀" if s["score"] >= 40 else "⏳")
        print(f"  {action_icon} {s['code']} {s['name_from_api']:<20s} "
              f"评分{s['score']:>5.0f}  日涨幅{s['daily_return']:>+5.1%}  "
              f"RSI={s.get('rsi_value', 'N/A')}  "
              f"回撤={s.get('drawdown_1y', 0):.1%}  "
              f"年收益={s.get('return_1y', 0):.1%}  "
              f"{s['action']}")

except Exception as e:
    print(f"  ❌ 扫描失败: {e}")
    import traceback; traceback.print_exc()

# ═════════════════════════════════════════════════════════════════════════════
# 5. Scoring boundary tests (unit-test style)
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n[5] 评分边界测试")
print("-" * 60)

params = ShortTermMomentumStrategy().get_default_params()

# Test 1: Strong momentum = high score
score1 = ShortTermMomentumStrategy._compute_composite_score(
    daily_ret=0.05, dist_ma20=0.05, ma20_slope=0.02,
    mcap_yi=100, turnover_pct=5.0, turnover_pool_pctile=0.9,
    drawdown_1y=0.05, return_1y=3.0, rsi_value=60, vol_ratio=2.0,
    params=params,
)
print(f"  强动量(5%涨幅+大市值+低回撤): {score1['total']:.0f}分")
assert score1["total"] > 70, f"Expected >70, got {score1['total']}"

# Test 2: Weak momentum = low score
score2 = ShortTermMomentumStrategy._compute_composite_score(
    daily_ret=0.015, dist_ma20=0.01, ma20_slope=0.005,
    mcap_yi=5, turnover_pct=0.5, turnover_pool_pctile=0.2,
    drawdown_1y=0.25, return_1y=0.5, rsi_value=40, vol_ratio=0.8,
    params=params,
)
print(f"  弱动量(1.5%涨幅+小市值+高回撤): {score2['total']:.0f}分")
assert score2["total"] < 55, f"Expected <55, got {score2['total']}"

# Test 3: Score monotonicity — better daily return → higher momentum score
score3a = ShortTermMomentumStrategy._compute_composite_score(
    daily_ret=0.02, dist_ma20=0.02, ma20_slope=0.01,
    mcap_yi=50, turnover_pct=3.0, turnover_pool_pctile=0.5,
    drawdown_1y=0.10, return_1y=2.0, rsi_value=55, vol_ratio=1.5,
    params=params,
)
score3b = ShortTermMomentumStrategy._compute_composite_score(
    daily_ret=0.04, dist_ma20=0.02, ma20_slope=0.01,
    mcap_yi=50, turnover_pct=3.0, turnover_pool_pctile=0.5,
    drawdown_1y=0.10, return_1y=2.0, rsi_value=55, vol_ratio=1.5,
    params=params,
)
print(f"  单调性: 2%涨幅→{score3a['total']:.0f}分 vs 4%涨幅→{score3b['total']:.0f}分")
assert score3b["total"] > score3a["total"], "Higher return should score higher"

# Test 4: Negative daily return = 0 momentum score
score4 = ShortTermMomentumStrategy._compute_composite_score(
    daily_ret=-0.02, dist_ma20=0.02, ma20_slope=0.01,
    mcap_yi=50, turnover_pct=3.0, turnover_pool_pctile=0.5,
    drawdown_1y=0.10, return_1y=2.0, rsi_value=55, vol_ratio=1.5,
    params=params,
)
print(f"  负收益日(-2%): momentum_score={score4['momentum_score']:.0f} (should be 0)")
assert score4["momentum_score"] == 0, f"Negative days should get 0 momentum score"

# Test 5: Drawdown penalty — higher drawdown → lower score
score5a = ShortTermMomentumStrategy._compute_composite_score(
    daily_ret=0.03, dist_ma20=0.02, ma20_slope=0.01,
    mcap_yi=50, turnover_pct=3.0, turnover_pool_pctile=0.5,
    drawdown_1y=0.05, return_1y=2.0, rsi_value=55, vol_ratio=1.5,
    params=params,
)
score5b = ShortTermMomentumStrategy._compute_composite_score(
    daily_ret=0.03, dist_ma20=0.02, ma20_slope=0.01,
    mcap_yi=50, turnover_pct=3.0, turnover_pool_pctile=0.5,
    drawdown_1y=0.30, return_1y=2.0, rsi_value=55, vol_ratio=1.5,
    params=params,
)
print(f"  回撤惩罚: 5%回撤→{score5a['drawdown_score']:.0f}分 vs 30%回撤→{score5b['drawdown_score']:.0f}分")
assert score5a["drawdown_score"] > score5b["drawdown_score"], "Lower drawdown should score higher"

print(f"  ✅ 所有评分边界测试通过")

# ═════════════════════════════════════════════════════════════════════════════
# 6. Param validation
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n[6] 参数验证")
print("-" * 60)

errors = strategy.validate_params(strategy.get_default_params())
if errors:
    for e in errors:
        print(f"  ❌ {e}")
else:
    print(f"  ✅ 所有参数验证通过")

# Min/max boundary check
all_params = strategy.get_default_params()
assert 0.01 <= all_params["min_prev_day_change"] <= 0.05
assert 0.01 <= all_params["stop_loss_pct"] <= 0.05
assert 0.02 <= all_params["take_profit_pct"] <= 0.08
assert 1 <= all_params["max_hold_days"] <= 7
assert all_params["max_hold_days"] == 4, "Default hold days should be 4"
print(f"  ✅ 参数边界检查通过")

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print(f"✅ 短线动量策略 测试完成")
print(f"{'='*80}")
print(f"  策略名称: {strategy.name}")
print(f"  参数数量: {len(params)}")
print(f"  候选ETF池: {len(CANDIDATE_ETFS)}只")
print(f"  候选股票池: {len(CANDIDATE_STOCKS)}只")
print(f"  注册状态: {'已注册' if '短线动量' in registry else '未注册'}")
if results:
    has_trades = sum(1 for r in results if r["total_trades"] > 0)
    print(f"  ETF回测: {len(results)}只中{has_trades}只有交易信号")
