Task 1: complete (commits 78cbf66..fc354ca, review approved)
Task 2: complete (commits fc354ca..f928c42, review approved, 15/15 tests)
Task 3: complete (commit 1d09efc, dashboard UI components)
Task 4: complete (commit ad7b83a, Streamlit app entry point)
Task 5: complete (commit af9adb7, integration verification)

--- v0.2a: Multi-source data adapters ---
Commit c190554: Tencent + Baidu + Sina + AKShare 4-source adapter layer
- fetch_etf_info() → Tencent primary (PE/PB/mcap/turnover), Sina fallback
- fetch_etf_hist() → Baidu primary (built-in MA5/10/20), AKShare fallback
- New: fetch_multi_etf_info() batch API
- Dashboard: new valuation row (PE/TTM, PE/静, PB, 总市值, 换手率)
- 29 tests (23 unit + 6 integration), all passing
