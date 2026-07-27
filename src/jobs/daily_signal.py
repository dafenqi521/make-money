"""Close scan and next-session reminder entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, time
from typing import Callable, Mapping

import pandas as pd
import requests

from src.data.fetcher import fetch_etf_hist_primary
from src.data.portfolio_db import PortfolioDB
from src.engine.etf_universe import (
    UniverseRefreshResult,
    discover_etf_universe,
    select_scan_pool,
)
from src.engine.rotation_scanner import (
    DEFAULT_ETF_POOL,
    RotationScanResult,
    scan_etf_pool,
)
from src.engine.signal_batch import (
    batch_id_for,
    config_hash,
    pool_hash,
    serialize_scan_result,
)
from src.engine.trading_schedule import (
    DAILY_BAR_READY_TIME,
    is_trading_day,
    next_trading_day,
    shanghai_now,
)
from src.strategy.etf_rotation import RotationConfig


def _notify(event: str, text: str, **details: object) -> bool:
    """Send via PushPlus (WeChat) when configured, otherwise log JSON."""

    payload = {"event": event, "text": text, **details}
    print(json.dumps(payload, ensure_ascii=False))

    # PushPlus 微信推送 (优先)
    pushplus_token = str(os.getenv("PUSHPLUS_TOKEN") or "").strip()
    if pushplus_token:
        title_map = {
            "signal_ready": "ETF轮动信号已生成",
            "morning_plan": "今日调仓计划",
            "preclose_remind": "尾盘确认 - 别忘下单",
            "confirmation_due": "ETF调仓待确认",
        }
        title = title_map.get(event, event)
        content = text
        resp = requests.post(
            "http://www.pushplus.plus/send",
            json={"token": pushplus_token, "title": title, "content": content},
            timeout=15,
        )
        resp.raise_for_status()
        print(f"PushPlus sent: {resp.json()}")
        return True

    # 兼容旧 webhook
    webhook = str(os.getenv("NOTIFY_WEBHOOK_URL") or "").strip()
    if not webhook:
        return False
    response = requests.post(webhook, json=payload, timeout=15)
    response.raise_for_status()
    return True


def _write_summary_json(
    scan: RotationScanResult,
    config: RotationConfig,
    pool: list[dict],
    result: dict,
) -> None:
    """Write a human-readable signal summary to the repo root."""
    from pathlib import Path

    targets = []
    for _, t in scan.targets.iterrows():
        code = str(t.get("code", ""))
        name = str(t.get("name", ""))
        weight = float(t.get("target_weight", 0))
        price = float(t.get("close", 0))
        score = float(t.get("score", 0))
        # 按10万本金算建议金额和股数
        suggested_amount = 100_000 * weight
        suggested_shares = int(suggested_amount / price / 100) * 100 if price > 0 else 0

        targets.append({
            "code": code,
            "name": name,
            "category": str(t.get("category", "")),
            "weight": round(weight, 4),
            "price": round(price, 3),
            "momentum_score": round(score, 1),
            "suggested_shares": suggested_shares,
            "suggested_amount": round(suggested_amount, 0),
        })

    summary = {
        "updated": scan.as_of.isoformat() if scan.as_of else "",
        "scanned": result.get("scan_count", 0),
        "targets": targets,
    }
    path = Path(__file__).resolve().parent.parent.parent / "latest_signal.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def refresh_universe(
    db: PortfolioDB,
    now: datetime | None = None,
    fetchers: Mapping[str, Callable[[], pd.DataFrame]] | None = None,
    config: RotationConfig | None = None,
) -> tuple[list[dict], dict | None]:
    """Refresh atomically; retain and return the previous snapshot on failure."""

    current = shanghai_now(now)
    previous_entries = db.get_etf_universe()
    try:
        result: UniverseRefreshResult = discover_etf_universe(
            fetchers=fetchers, refreshed_at=current, config=config
        )
        if (
            previous_entries
            and result.total_count < int(len(previous_entries) * 0.70)
        ):
            raise ValueError(
                f"ETF目录数量从{len(previous_entries)}降至{result.total_count}，"
                "超过30%收缩门槛，拒绝覆盖旧快照"
            )
        run_id = db.replace_etf_universe(
            result.entries, result.source, result.refreshed_at
        )
        return result.entries, {
            "run_id": run_id,
            "status": "success",
            "source": result.source,
            "refreshed_at": result.refreshed_at,
            "total_count": result.total_count,
            "eligible_count": result.eligible_count,
            "error": "；".join(result.errors.values()),
        }
    except Exception as error:
        db.record_universe_failure(str(error), current.isoformat(timespec="seconds"))
        return db.get_etf_universe(), db.get_universe_status()


def run_scan_job(
    db: PortfolioDB,
    now: datetime | None = None,
    universe_fetchers: Mapping[str, Callable[[], pd.DataFrame]] | None = None,
    scanner: Callable[..., RotationScanResult] = scan_etf_pool,
) -> dict:
    """Refresh the universe, scan it, and persist one deterministic batch."""

    current = shanghai_now(now)
    if not is_trading_day(current.date()):
        return {"status": "skipped", "reason": "休市日"}
    if current.time().replace(tzinfo=None) < DAILY_BAR_READY_TIME:
        return {"status": "skipped", "reason": "完整日线尚未就绪"}

    config = RotationConfig()
    entries, universe_status = refresh_universe(
        db, now=current, fetchers=universe_fetchers, config=config
    )
    portfolio = db.load()
    held_codes = portfolio.holdings.keys() if portfolio else ()
    max_count = int(os.getenv("MAX_SCAN_ETFS", "0") or 0)
    pool = select_scan_pool(
        entries,
        minimum_spot_amount=config.min_daily_amount,
        max_count=max_count or None,
        always_include=held_codes,
    )
    if not pool:
        pool = [dict(row) for row in DEFAULT_ETF_POOL]

    scan = scanner(
        pool=pool,
        config=config,
        history_fetcher=fetch_etf_hist_primary,
        max_workers=int(os.getenv("SCAN_MAX_WORKERS", "12") or 12),
        now=current,
    )
    requested = scan.scanned_count + len(scan.errors)
    coverage = scan.scanned_count / requested if requested else 0.0
    used_backup_pool = False
    if coverage < 0.80:
        backup_pool = [dict(row) for row in DEFAULT_ETF_POOL]
        backup_scan = scanner(
            pool=backup_pool,
            config=config,
            max_workers=1,
            now=current,
        )
        backup_requested = backup_scan.scanned_count + len(backup_scan.errors)
        backup_coverage = (
            backup_scan.scanned_count / backup_requested
            if backup_requested else 0.0
        )
        if backup_scan.as_of is not None and backup_coverage >= coverage:
            scan = backup_scan
            pool = backup_pool
            coverage = backup_coverage
            used_backup_pool = True
    if scan.as_of is None:
        raise RuntimeError("扫描未产生完整日线信号")
    batch_id = batch_id_for(scan, config, pool)
    saved = db.save_signal_batch(
        batch_id=batch_id,
        signal_date=scan.as_of.isoformat(),
        config_hash=config_hash(config),
        pool_hash=pool_hash(pool),
        payload=serialize_scan_result(scan),
        scan_count=scan.scanned_count,
        error_count=len(scan.errors),
        universe_run_id=(
            str(entries[0].get("run_id"))
            if entries and entries[0].get("run_id") else (universe_status or {}).get("run_id")
        ),
    )
    if not saved:
        raise RuntimeError("信号批次保存失败")
    result = {
        "status": "success",
        "batch_id": batch_id,
        "signal_date": scan.as_of.isoformat(),
        "universe_count": len(entries),
        "pool_count": len(pool),
        "scan_count": scan.scanned_count,
        "target_count": len(scan.targets),
        "coverage": coverage,
        "used_backup_pool": used_backup_pool,
    }
    # 构建微信推送：列出目标ETF + 价格 + 建议
    target_lines = []
    for _, t in scan.targets.iterrows():
        code = str(t.get("code", ""))
        name = str(t.get("name", ""))
        weight = float(t.get("target_weight", 0))
        price = float(t.get("close", 0))
        amt = 100_000 * weight
        shares = int(amt / price / 100) * 100 if price > 0 else 0
        target_lines.append(
            f"{code} {name}\n"
            f"  价格 ￥{price:.3f}  |  仓位 {weight:.0%}  |  建议 {shares}股(约￥{amt:.0f})"
        )

    target_text = "\n".join(target_lines) if target_lines else "空仓"

    notify_text = (
        f"扫描 {scan.scanned_count} 只 ETF\n"
        f"---\n"
        f"{target_text}"
    )
    _notify("signal_ready", notify_text, **result)

    # 写一份人类可读的 JSON 摘要到仓库根目录
    _write_summary_json(scan, config, pool, result)

    return result


def run_reminder_job(db: PortfolioDB, now: datetime | None = None) -> dict:
    """推送当日买卖计划（10:00早盘 / 14:30尾盘）。"""
    current = shanghai_now(now)
    if not is_trading_day(current.date()):
        return {"status": "skipped", "reason": "休市日"}

    # 直接读 latest_signal.json，不依赖数据库锁
    from pathlib import Path
    sig_path = Path(__file__).resolve().parent.parent.parent / "latest_signal.json"
    if not sig_path.exists():
        return {"status": "skipped", "reason": "暂无信号文件"}

    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    targets = sig.get("targets", [])
    if not targets:
        return {"status": "skipped", "reason": "无目标标的"}

    signal_date = sig.get("updated", "")

    # 早盘 / 尾盘
    current_time = current.time().replace(tzinfo=None)
    if current_time < time(12, 0):
        event_type = "morning_plan"
        header = "今日买卖计划 (10:00)"
        footer = "尾盘14:30再次提醒"
    else:
        event_type = "preclose_remind"
        header = "尾盘确认 (14:30)"
        footer = "距收盘30分钟，请执行！"

    lines = []
    for t in targets[:8]:
        price = t.get("price", 0)
        weight = t.get("weight", 0)
        shares = t.get("suggested_shares", 0)
        lines.append(
            f"{t['code']} {t['name']}\n"
            f"  价格 ￥{price:.3f}  |  仓位 {weight:.0%}  |  {shares}股"
        )
    message = (
        f"{header}\n"
        f"信号日期: {signal_date}\n"
        f"---\n"
        f"{chr(10).join(lines)}\n"
        f"---\n"
        f"{footer}"
    )
    sent = _notify(event_type, message)
    return {"status": "success", "notified": sent}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("scan", "remind"))
    args = parser.parse_args()
    db = PortfolioDB(database_url=os.getenv("DATABASE_URL"))
    result = run_scan_job(db) if args.task == "scan" else run_reminder_job(db)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
