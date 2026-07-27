"""收盘扫描 → 对比上次目标 → 算出买卖清单 → 微信推送。

无数据库依赖，纯 JSON 文件记录历史目标，GitHub Actions 自动提交回仓库。
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, time
from pathlib import Path

import requests

from src.data.fetcher import fetch_etf_hist_primary
from src.engine.rotation_scanner import DEFAULT_ETF_POOL, RotationScanResult, scan_etf_pool
from src.engine.trading_schedule import is_trading_day, shanghai_now
from src.strategy.etf_rotation import RotationConfig

# ---------------------------------------------------------------------------
# 文件路径
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
SIGNAL_FILE = ROOT / "latest_signal.json"
PREV_FILE = ROOT / "previous_targets.json"  # 记录上次目标，用于算差额

# ---------------------------------------------------------------------------
# PushPlus 微信推送
# ---------------------------------------------------------------------------

def _push(title: str, content: str) -> bool:
    token = str(os.getenv("PUSHPLUS_TOKEN") or "").strip()
    if not token:
        print("[push] PUSHPLUS_TOKEN not set, skip")
        return False
    resp = requests.post(
        "http://www.pushplus.plus/send",
        json={"token": token, "title": title, "content": content},
        timeout=15,
    )
    resp.raise_for_status()
    print(f"[push] sent: {resp.json()}")
    return True


# ---------------------------------------------------------------------------
# 扫描 + 对比
# ---------------------------------------------------------------------------

def run_scan_job() -> dict:
    now = shanghai_now()
    if not is_trading_day(now.date()):
        return {"status": "skipped", "reason": "休市日"}

    config = RotationConfig()
    pool = [dict(row) for row in DEFAULT_ETF_POOL]

    print(f"[scan] 扫描 {len(pool)} 只 ETF ...")
    scan = scan_etf_pool(
        pool=pool,
        config=config,
        history_fetcher=fetch_etf_hist_primary,
        max_workers=int(os.getenv("SCAN_MAX_WORKERS", "12") or 12),
        now=now,
    )

    if scan.as_of is None:
        return {"status": "error", "reason": "扫描未产生有效信号"}

    # ---- 构建本次目标清单 ----
    current_targets = []
    for _, t in scan.targets.iterrows():
        code = str(t.get("code", ""))
        name = str(t.get("name", ""))
        weight = float(t.get("target_weight", 0))
        price = float(t.get("close", 0))
        score = float(t.get("score", 0))

        amt = 100_000 * weight
        shares = int(amt / price / 100) * 100 if price > 0 else 0

        current_targets.append({
            "code": code,
            "name": name,
            "category": str(t.get("category", "")),
            "weight": round(weight, 4),
            "price": round(price, 3),
            "score": round(score, 1),
            "shares": shares,
            "amount": round(amt, 0),
        })

    # ---- 读取上次目标，对比 ----
    prev_targets = []
    if PREV_FILE.exists():
        try:
            prev_data = json.loads(PREV_FILE.read_text(encoding="utf-8"))
            prev_targets = prev_data.get("targets", [])
        except Exception:
            pass

    prev_map = {t["code"]: t for t in prev_targets}
    curr_map = {t["code"]: t for t in current_targets}

    all_codes = set(list(prev_map.keys()) + list(curr_map.keys()))

    buy_list = []   # 新买入 / 加仓
    sell_list = []  # 卖出 / 减仓
    hold_list = []  # 不变

    for code in all_codes:
        prev = prev_map.get(code)
        curr = curr_map.get(code)

        if prev and curr:
            # 都有 → 比较权重变化
            diff = curr["weight"] - prev["weight"]
            if diff > 0.05:  # 权重增加超过5%算加仓
                buy_list.append({**curr, "action": "加仓", "prev_weight": prev["weight"]})
            elif diff < -0.05:  # 权重减少超过5%算减仓
                sell_list.append({**prev, "action": "减仓", "new_weight": curr["weight"]})
            else:
                hold_list.append(curr)
        elif curr and not prev:
            # 新增
            buy_list.append({**curr, "action": "新买入", "prev_weight": 0})
        elif prev and not curr:
            # 清仓
            sell_list.append({**prev, "action": "清仓", "new_weight": 0})

    # ---- 保存当前目标，供下次对比 ----
    PREV_FILE.write_text(
        json.dumps({"updated": scan.as_of.isoformat(), "targets": current_targets},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- 保存 latest_signal.json（人可读） ----
    signal_data = {
        "updated": scan.as_of.isoformat(),
        "scanned": scan.scanned_count,
        "targets": current_targets,
        "buy": buy_list,
        "sell": sell_list,
    }
    SIGNAL_FILE.write_text(
        json.dumps(signal_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- 构建微信推送 ----
    lines = [f"收盘扫描 {scan.scanned_count} 只 ETF\n"]

    if buy_list:
        lines.append("--- 买入 ---")
        for t in buy_list:
            lines.append(
                f"[{t['action']}] {t['code']} {t['name']}\n"
                f"  价格 ￥{t['price']:.3f}  |  目标仓位 {t['weight']:.0%}  |  {t['shares']}股 (约￥{t['amount']:.0f})"
            )

    if sell_list:
        lines.append("--- 卖出 ---")
        for t in sell_list:
            lines.append(
                f"[{t['action']}] {t['code']} {t['name']}"
            )

    if not buy_list and not sell_list:
        lines.append("持仓不变，无需操作")

    lines.append("\n--- 持有 ---")
    for t in hold_list:
        lines.append(f"{t['code']} {t['name']}  仓位 {t['weight']:.0%}  ￥{t['price']:.3f}")

    lines.append("\n明日 10:00 / 14:30 再次提醒")

    _push(f"ETF信号 {scan.as_of}", "\n".join(lines))

    return {
        "status": "success",
        "signal_date": scan.as_of.isoformat(),
        "scanned": scan.scanned_count,
        "targets": len(current_targets),
        "buy": len(buy_list),
        "sell": len(sell_list),
    }


# ---------------------------------------------------------------------------
# 早盘/尾盘提醒
# ---------------------------------------------------------------------------

def run_reminder_job() -> dict:
    now = shanghai_now()
    if not is_trading_day(now.date()):
        return {"status": "skipped", "reason": "休市日"}

    if not SIGNAL_FILE.exists():
        return {"status": "skipped", "reason": "暂无信号文件"}

    sig = json.loads(SIGNAL_FILE.read_text(encoding="utf-8"))
    buy_list = sig.get("buy", [])
    sell_list = sig.get("sell", [])
    targets = sig.get("targets", [])

    current_time = now.time().replace(tzinfo=None)
    if current_time < time(12, 0):
        header = "今日买卖计划 (10:00)"
        footer = "尾盘 14:30 再次提醒"
    else:
        header = "尾盘确认 (14:30)"
        footer = "距收盘 30 分钟，请执行！"

    lines = [header, f"信号日期: {sig.get('updated', '')}\n"]

    if buy_list:
        lines.append("--- 买入 ---")
        for t in buy_list:
            lines.append(
                f"[{t['action']}] {t['code']} {t['name']}\n"
                f"  价格 ￥{t['price']:.3f}  |  仓位 {t['weight']:.0%}  |  {t['shares']}股"
            )

    if sell_list:
        lines.append("--- 卖出 ---")
        for t in sell_list:
            lines.append(f"[{t['action']}] {t['code']} {t['name']}")

    if not buy_list and not sell_list:
        lines.append("持仓不变，无需操作\n")
        for t in targets:
            lines.append(f"{t['code']} {t['name']}  ￥{t['price']:.3f}  {t['weight']:.0%}")

    lines.append(f"\n{footer}")
    _push(header, "\n".join(lines))

    return {"status": "success", "notified": True}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("scan", "remind"))
    args = parser.parse_args()

    if args.task == "scan":
        result = run_scan_job()
    else:
        result = run_reminder_job()

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
