# Task 4 Report: Streamlit App Entry Point

## Status: COMPLETE

**Commit:** `ad7b83ab577f729e5189a1dd4ee867d532a648d8`

---

## What was done

Created `F:\MyProject\app.py` — the Streamlit entry point for the ETF Investment Decision System v0.1.

The app provides:
- **Sidebar**: ETF code input (default "510300"), date range selector (6 presets), data source attribution, risk disclaimer
- **Main area**: ETF overview metrics (from `render_etf_overview`), tabbed K-line chart (`render_price_chart`) and data table (`render_data_table`)
- **Error handling**: ValueError from `fetch_etf_info` is caught and passed to `render_no_data`; generic exceptions show `st.error` with a retry hint
- **Input validation**: Empty symbol shows info prompt; non-6-digit or non-numeric input shows warning

## Verification

| Check | Result |
|-------|--------|
| `python -c "py_compile"` syntax check | PASS |
| Import resolution (fetcher + dashboard) | PASS |
| `streamlit run app.py` startup (10s) | PASS — app serves on localhost:8501 |
| `git commit` | PASS — `ad7b83a` |

## Concerns

1. **Static date_map**: The date_map uses hardcoded start dates (e.g., `"近1个月": "20250101"`). These are stale relative to the current date (June 2026) and the mapping is inconsistent — "近1个月" points to January 2025, while "近3个月" points to March 2025. The dates are as specified in the task brief; a future task should replace them with dynamically computed dates (e.g., `pd.Timestamp.today() - pd.DateOffset(months=1)`).

2. **Sequential fetches**: `fetch_etf_info` and `fetch_etf_hist` run sequentially. Since they are independent API calls, they could benefit from concurrent execution (e.g., `asyncio` or `threading`) for reduced latency.

3. **No test coverage for app.py**: The app entry point has no automated tests. Streamlit apps are inherently UI-heavy and difficult to unit-test, but integration verification (Task 5) will cover manual testing.

## Report path

`F:\MyProject\.superpowers\sdd\task-4-report.md`
