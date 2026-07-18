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
  └─ src/data/portfolio_db.py
```

## 验证

```bash
python -m pytest tests -q
python -m py_compile app.py src/strategy/etf_rotation.py src/engine/rotation_scanner.py src/engine/paper_trading.py src/engine/portfolio.py src/data/portfolio_db.py
python -m streamlit run app.py
```

## 约束

- 数据缺失、扫描覆盖率不足或数据过期时不得产生自动交易；
- 信号只使用已完成交易日数据；
- ETF费用模型不含股票印花税；
- 买卖数量按100份取整，同日新买份额不可卖；
- SQLite用于单用户模拟账户；云端必须提供JSON备份恢复；
- 默认不连接真实券商，不承诺收益；
- 用户已有未关联修改必须保留。
