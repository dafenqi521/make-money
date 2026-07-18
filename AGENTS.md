# make-money — 场内ETF趋势轮动

## 唯一策略

项目只支持“场内ETF多资产趋势轮动”。不要新增、恢复或保留网格、定投、短线波段、超短线、单标的均线等其他策略。

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
  └─ src/engine/rotation_scanner.py
       ├─ src/data/fetcher.py
       └─ src/strategy/etf_rotation.py

joinquant/etf_rotation_strategy.py  # 聚宽独立执行适配器
```

## 验证

```bash
python -m pytest tests -q
python -m py_compile app.py src/strategy/etf_rotation.py src/engine/rotation_scanner.py joinquant/etf_rotation_strategy.py
streamlit run app.py
```

## 约束

- 数据缺失或过期时不得产生买入信号；
- 信号只使用前一交易日及更早的完整日线；
- ETF费用模型不含股票印花税；
- 本地应用只做研究和模拟建议；真实下单必须经过独立券商适配和安全门；
- 用户已有未关联修改必须保留。

