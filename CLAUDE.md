# make-money — ETF 投资决策系统

## 这是什么

基于 Streamlit 的 ETF 场内基金数据展示 + 策略回测系统。输入 ETF 代码即可看到实时行情（五档盘口、PE/PB/市值/换手率）、K线图（含MA5/10/20）、历史数据表，以及四大策略回测对比。

## 跑起来

```bash
pip install -r requirements.txt
python -m pytest tests/ -v     # 98个测试，确认全过
streamlit run app.py            # 打开 http://localhost:8501
```

## 架构

```
用户输入ETF代码 → app.py
  ├─ 行情数据 Tab → src/ui/dashboard.py
  │   ├─ src/data/fetcher.py (腾讯/Baidu/Sina/AKShare 4源)
  │   └─ src/ui/terminal_theme.py (主题/CSS/Plotly)
  └─ 策略回测 Tab → src/ui/strategy_ui.py
      ├─ src/strategy/ (4个策略实现)
      └─ src/engine/ (回测引擎: broker/risk/metrics/backtest)
```

## 数据源优先级

1. **腾讯财经** `qt.gtimg.cn` — 实时价+PE/PB/市值，不封IP
2. **百度股市通** — 日K线自带MA5/10/20
3. **新浪** — 兜底
4. **AKShare** — ETF列表

关键函数：
- `fetch_etf_info("510300")` → dict (35字段)
- `fetch_etf_hist("510300")` → DataFrame (11列: OHLCV + MA + 涨跌幅 + 振幅)
- `fetch_multi_etf_info(["510300","510050"])` → 批量查询

## 策略引擎

自定义轻量引擎，零外部依赖。四种策略：

| 策略 | 逻辑 | 类 |
|------|------|-----|
| 趋势跟随 | MA金叉买/死叉卖 | `TrendFollowingStrategy` |
| 网格交易 | 价格带N档低买高卖 | `GridTradingStrategy` |
| 估值定投 | PE阈值每月定投 | `ValueAveragingStrategy` |
| 混合策略 | 60%定投+40%网格 | `HybridStrategy` |

用法：
```python
from src.engine.backtest import BacktestEngine
from src.strategy.trend_following import TrendFollowingStrategy

engine = BacktestEngine(initial_capital=100_000)
result = engine.run(df, TrendFollowingStrategy(), pe_value=15.0)
print(result.summary())  # 年化/Sharpe/MaxDD/胜率
```

## UI 设计约束

- 当前主题：浅色卡片式 (`src/ui/terminal_theme.py`)
- 页面切换用 `st.tabs`（客户端无刷新切换）
- 所有标签用中文
- 盘口用横向深度条代替表格
- 数据表视觉层次：收盘+涨跌幅(T1) > OHLC(T2) > MA/振幅(T3)
- 用 `st.container(border=True)` 包裹相关内容形成卡片

## 下一步（v0.3）

1. **买入/卖出信号** — 根据当前PE+MA+网格，生成每日操作建议
2. **PE历史数据接入** — 通过指数PE计算ETF估值分位
3. **globalpercent宏观温度计** — Polymarket/Kalshi概率面板集成
4. **交易记录持久化** — SQLite存储交易记录+持仓跟踪

## 注意事项

- Python 3.8（建议升3.9+但当前可用）
- ETF的PE/TTM腾讯经常不返回（数据源特性，非Bug）
- `Styler.applymap()` 不是 `.map()`（pandas 2.0兼容）
- 已安装的 Skill：`a-stock-data`, `globalpercent`, `frontend-design`
