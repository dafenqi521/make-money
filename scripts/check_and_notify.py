#!/usr/bin/env python
"""离线通知检查脚本 — 零 Streamlit 依赖.

可在 Windows 计划任务、Cron、或 Claude Code CronCreate 中调用。

用法:
  python scripts/check_and_notify.py --code 510300                # 检查信号并推送（有信号才发）
  python scripts/check_and_notify.py --code 510300 --summary      # 仅发送定时简报
  python scripts/check_and_notify.py --code 510300 --summary 午间  # 发送简报（自定义时段标签）
  python scripts/check_and_notify.py --test                        # 测试推送通道
  python scripts/check_and_notify.py --code 510300 --force         # 强制发送（忽略去重）

环境要求:
  pip install requests pandas          # 无 Streamlit 依赖
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on sys.path (in case script is run from outside)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ETF 通知检查 — 信号推送 / 定时简报",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python check_and_notify.py --code 510300              检查信号并推送
  python check_and_notify.py --code 510300 --summary    发送午间/收盘简报
  python check_and_notify.py --test                     测试 PushPlus 通道
        """,
    )
    parser.add_argument(
        "--code", type=str, default="",
        help="ETF 代码（6位数字），默认使用 config.json 中的第一个代码",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="发送定时简报（而非信号推送）",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="发送测试消息以验证推送通道",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制发送，忽略去重逻辑",
    )
    parser.add_argument(
        "--time-label", type=str, default="",
        help="简报时段标签，如「午间」「收盘前」",
    )

    args = parser.parse_args()

    # --- Test mode ---
    if args.test:
        from src.notify import NotificationSender

        sender = NotificationSender()
        ok = sender.send_test()
        if ok:
            print("[OK] 测试消息发送成功！请检查您的微信。")
        else:
            print("[FAIL] 测试消息发送失败。请检查：")
            print("  1. pushplus_token 是否填写正确？")
            print("  2. 是否已在 pushplus.plus 注册并关注公众号？")
            print("  3. 网络是否能访问 api.pushplus.plus？")
        sys.exit(0 if ok else 1)

    # --- Determine ETF code ---
    code = args.code
    if not code:
        from src.notify.config import load_config
        cfg = load_config()
        codes = cfg.get("etf_codes", [])
        code = codes[0] if codes else ""
    if not code:
        print("[FAIL] 未指定 ETF 代码。请使用 --code 参数或在 config.json 中配置 etf_codes。")
        sys.exit(1)

    code = code.strip()
    if len(code) != 6 or not code.isdigit():
        print(f"[FAIL] 无效的 ETF 代码: {code}")
        sys.exit(1)

    # --- Fetch data ---
    print(f"[INFO] 正在获取 {code} 数据...")
    from src.data.fetcher import fetch_etf_info, fetch_etf_hist

    try:
        info = fetch_etf_info(code)
        df = fetch_etf_hist(code, start_date=None)  # 全部历史
    except Exception as e:
        print(f"[FAIL] 数据获取失败: {e}")
        sys.exit(1)

    if df is None or df.empty:
        print(f"[FAIL] {code} 无历史数据")
        sys.exit(1)

    name = info.get("name", f"ETF {code}")
    pe_value = info.get("pe_ttm") or info.get("pe_static")
    current_price = info.get("current_price")

    print(f"[INFO] {code} {name} — 当前价 ¥{current_price}")

    # --- PE percentile (optional) ---
    pe_percentile = None
    try:
        from src.data.index_map import has_pe_data
        from src.data.pe_history import get_etf_pe_percentile

        if has_pe_data(code):
            pe_percentile = get_etf_pe_percentile(code, current_pe=pe_value)
            if pe_percentile is not None:
                print(f"[INFO] PE 历史分位: {pe_percentile.pe_percentile:.1%}")
    except Exception:
        pass  # PE data is optional

    # --- Macro pulse (optional) ---
    macro_pulse = None
    try:
        from src.data.macro_pulse import get_macro_pulse
        macro_pulse = get_macro_pulse()
    except Exception:
        pass

    # --- Compute signal ---
    from src.ui.signal_panel import compute_daily_signal

    daily_signal = compute_daily_signal(
        df, info, pe_percentile=pe_percentile, macro_pulse=macro_pulse,
    )

    print(
        f"[INFO] 综合信号: {daily_signal.action_icon} {daily_signal.action_label}"
        f"（评分 {daily_signal.composite_score:.2f}）"
    )

    # --- Send ---
    from src.notify import NotificationSender

    sender = NotificationSender()

    # Force mode: temporarily override config
    if args.force:
        sender.config["enabled"] = True
        sender.config["notify_on_actions"] = ["buy", "sell", "accumulate", "reduce", "hold"]

    if args.summary:
        # Build ETF data for summary
        # Get change_pct from the last row
        try:
            change_pct = float(df.iloc[-1].get("change_pct", 0) or 0)
        except (IndexError, TypeError):
            change_pct = 0.0

        etf_data = [{
            "code": code,
            "name": name,
            "price": current_price,
            "change_pct": change_pct,
            "signal": daily_signal,
            "pe_value": pe_value,
        }]

        time_label = args.time_label or ""
        ok = sender.send_daily_summary(etf_data, time_label=time_label)
        if ok:
            print(f"[OK] 定时简报已发送至微信！")
        else:
            print("[FAIL] 简报发送失败。请检查配置和网络。")
            sys.exit(1)
    else:
        # Signal alert mode
        ok = sender.send_signal_alert(
            code, name, daily_signal, info,
            pe_percentile=pe_percentile,
        )
        if ok:
            print(f"[OK] 信号推送已发送至微信！")
        else:
            action = daily_signal.composite_action
            if action == "hold":
                print("[SKIP] 当前为持有信号，无需推送。")
            elif not sender.config.get("enabled", False):
                print("[SKIP] 通知功能未启用。请在 config.json 中设置 enabled=true")
            else:
                print(f"[SKIP] 信号「{action}」被去重或不在推送白名单中。")
            sys.exit(0)


if __name__ == "__main__":
    main()
