"""Close scan and next-session reminder entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
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
    """Send a provider-neutral webhook when configured, otherwise log JSON."""

    payload = {"event": event, "text": text, **details}
    print(json.dumps(payload, ensure_ascii=False))
    webhook = str(os.getenv("NOTIFY_WEBHOOK_URL") or "").strip()
    if not webhook:
        return False
    response = requests.post(webhook, json=payload, timeout=15)
    response.raise_for_status()
    return True


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
    _notify(
        "signal_ready",
        f"ETF轮动信号已生成：扫描{scan.scanned_count}只，目标{len(scan.targets)}只",
        **result,
    )
    return result


def run_reminder_job(db: PortfolioDB, now: datetime | None = None) -> dict:
    current = shanghai_now(now)
    if not is_trading_day(current.date()):
        return {"status": "skipped", "reason": "休市日"}
    stored = db.get_latest_signal_batch()
    if not stored:
        return {"status": "skipped", "reason": "没有已保存信号"}
    signal_date = pd.to_datetime(stored["signal_date"], errors="coerce")
    if pd.isna(signal_date) or next_trading_day(signal_date.date()) != current.date():
        return {"status": "skipped", "reason": "今天不是该信号的确认日"}
    execution = db.get_execution_batch(stored["batch_id"])
    if execution and execution.get("status") == "completed":
        return {"status": "skipped", "reason": "该信号已确认执行"}
    message = "ETF轮动调仓待确认：请在北京时间09:35–10:00打开Streamlit页面复核。"
    sent = _notify(
        "confirmation_due",
        message,
        batch_id=stored["batch_id"],
        signal_date=stored["signal_date"],
    )
    return {"status": "success", "notified": sent, "batch_id": stored["batch_id"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("scan", "remind"))
    args = parser.parse_args()
    if os.getenv("GITHUB_ACTIONS") == "true" and not os.getenv("DATABASE_URL"):
        raise RuntimeError("GitHub Actions需要配置DATABASE_URL仓库Secret")
    db = PortfolioDB(database_url=os.getenv("DATABASE_URL"))
    result = run_scan_job(db) if args.task == "scan" else run_reminder_job(db)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
