#!/usr/bin/env python
"""Autonomous paper-trading bot — ETF momentum strategy.

Usage:
  python auto_trade.py --once              Dry run: scan once, print signals
  python auto_trade.py --live              Live auto-trading loop
  python auto_trade.py --live --auto-refine Live + auto parameter tuning
  python auto_trade.py --reset             Reset paper account to ¥4,000
  python auto_trade.py --report            Print trade cycle analysis
  python auto_trade.py --report --detailed Full per-cycle breakdown

Environment:
  Logs saved to logs/auto_trader.log
  Portfolio saved to src/data/portfolio_db/portfolio.sqlite3
  Params saved to src/engine/auto_trader_params.json
"""

from __future__ import annotations

import argparse
import os
import sys

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def cmd_once(_args):
    """Scan once, print signals, do not execute trades."""
    from src.engine.auto_trader import AutoTrader

    trader = AutoTrader()
    if not trader.is_market_open():
        print("[INFO] 市场当前休市，扫描结果可能基于昨日收盘价。\n")

    result = trader.run_once(dry_run=True)
    print(f"\n扫描 {result['scanned']} 个标的 | 信号 {result['signals_found']} 个")
    print(f"当前权益 ¥{result['equity']:,.2f} | 现金 ¥{result['cash']:,.2f} | 持仓 {result['positions']}")


def cmd_live(args):
    """Run continuous auto-trading loop."""
    from src.engine.auto_trader import AutoTrader

    interval = getattr(args, "interval", 5)
    auto_refine = getattr(args, "auto_refine", False)
    trader = AutoTrader(auto_refine=auto_refine)
    trader.run_loop(interval_minutes=interval)


def cmd_reset(_args):
    """Reset paper trading account."""
    from src.data.portfolio_db import PortfolioDB

    db = PortfolioDB()
    db.reset()
    # Also clear params
    params_file = os.path.join(
        os.path.dirname(__file__), "src", "engine", "auto_trader_params.json"
    )
    if os.path.exists(params_file):
        os.remove(params_file)
    print("[OK] 模拟账户已重置。下次启动将创建新的 ¥4,000 账户。")


def cmd_report(args):
    """Print trade cycle analysis."""
    from src.engine.auto_trader import AutoTrader

    trader = AutoTrader()
    trader.print_report(detailed=args.detailed)


def cmd_params(_args):
    """Show current strategy parameters."""
    from src.engine.auto_trader import AutoTrader

    trader = AutoTrader()
    print("\n当前策略参数:")
    for k, v in sorted(trader._params.items()):
        print(f"  {k}: {v}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="自动模拟交易机器人 — ETF短线动量策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python auto_trade.py --once               # 扫描一次看信号
  python auto_trade.py --live               # 持续自动交易
  python auto_trade.py --report             # 查看交易报告
        """,
    )
    parser.add_argument(
        "--once", action="store_true",
        help="扫描一次，只看信号不下单（干跑）",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="启动持续自动交易循环",
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="扫描间隔（分钟），默认5",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="重置模拟账户到 ¥4,000",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="打印交易周期分析报告",
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="报告中显示每笔明细",
    )
    parser.add_argument(
        "--params", action="store_true",
        help="显示当前策略参数",
    )
    parser.add_argument(
        "--auto-refine", action="store_true",
        help="启用自动参数优化（与 --live 配合使用）",
    )

    args = parser.parse_args()

    modes = [
        args.once,
        args.live,
        args.reset,
        args.report,
        args.params,
    ]
    if sum(modes) == 0:
        parser.print_help()
        sys.exit(1)
    if sum(modes) > 1:
        print("[FAIL] 请只指定一种模式 (--once, --live, --reset, --report, --params)")
        sys.exit(1)

    if args.once:
        cmd_once(args)
    elif args.live:
        cmd_live(args)
    elif args.reset:
        cmd_reset(args)
    elif args.report:
        cmd_report(args)
    elif args.params:
        cmd_params(args)


if __name__ == "__main__":
    main()
