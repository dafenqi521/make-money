# make-money — 场内ETF趋势轮动

## 唯一策略

项目只支持“场内ETF多资产趋势轮动”。不要新增、恢复或保留网格、定投、短线波段、超短线、单标的均线等其他策略，也不要重新引入聚宽依赖。

策略核心：

- 流动性硬过滤；
- 收盘价 > MA20 > MA60，且 MA5 > MA10；
- 20/60/120日动量横截面排名；
- 波动率与最大回撤控制；
- 资产类别与60日相关性去重；
- 最多4只、波动率倒数仓位、保留现金；
- 动态止损、移动止盈、趋势/排名/时间退出。

## 架构

```text
app.py
  ├─ src/engine/rotation_scanner.py
  │    ├─ src/data/fetcher.py
  │    └─ src/strategy/etf_rotation.py
  ├─ src/engine/paper_trading.py
  │    └─ src/engine/portfolio.py
  ├─ src/engine/etf_universe.py
  ├─ src/engine/signal_batch.py
  ├─ src/engine/backtest.py
  ├─ src/data/portfolio_db.py
  └─ src/jobs/daily_signal.py
```

## 验证

```bash
python -m pytest tests -q
python -m py_compile app.py src/strategy/etf_rotation.py src/engine/rotation_scanner.py src/engine/paper_trading.py src/engine/backtest.py src/engine/portfolio.py src/data/portfolio_db.py
python -m streamlit run app.py
```

## 约束

- 数据缺失、扫描覆盖率不足或数据过期时不得产生自动交易；
- 信号只使用已完成交易日数据；
- 北京时间15:05前不得把当日动态K线视为完整日线；
- 模拟确认只允许在下一开市日09:35–11:25或13:05–14:50，首选09:35–10:00；
- 确认时必须校验当日实时盘口日期，买入使用卖一、卖出使用买一优先，覆盖率低于80%冻结整单；
- 盘口必须不超过30秒且深度覆盖订单；涨跌停或偏离信号价超过3%时冻结；
- 全市场目录刷新失败必须保留上一份有效快照，默认19只只能作为备用；
- 同一信号批次只能完成一次模拟执行；
- 后台任务只生成信号和通知，不得绕过人工确认或连接券商；
- ETF费用模型不含股票印花税；
- 买卖数量按100份取整，同日新买份额不可卖；
- 项目只保留一个模拟账户；SQLite用于本地，云端可用DATABASE_URL切换PostgreSQL；
- 默认不连接真实券商，不承诺收益；
- 用户已有未关联修改必须保留。
