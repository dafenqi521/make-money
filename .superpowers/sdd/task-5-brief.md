### Task 5: Final Integration Verification

**Files:**
- No new files. Verify all existing.

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Run the app and do a smoke test with a real ETF**

Open browser, verify:
- Default ETF (510300) loads correctly
- Metrics show non-empty values
- K-line chart renders
- Data table has records with dates
- Most recent record is yesterday or today

- [ ] **Step 3: Final commit and push**

```bash
git add -A
git commit -m "v0.1: ETF data display — project skeleton, AKShare fetcher, Streamlit dashboard"
git push origin main
```

---

## Test Plan Summary

| Test | Type | How |
|------|------|-----|
| `test_fetch_etf_hist_returns_dataframe` | Unit | pytest |
| `test_fetch_etf_hist_empty_symbol` | Unit | pytest |
| `test_fetch_etf_info_returns_dict` | Unit | pytest |
| `test_get_available_etfs_returns_dataframe` | Unit | pytest |
| App loads without crash | Smoke | Manual: `streamlit run app.py` |
| Real ETF data renders | Integration | Manual: input 510300, verify chart + table |
| Invalid code handled | Edge case | Manual: input 999999, verify error message |
