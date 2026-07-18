# make-money

只保留“场内ETF多资产趋势轮动”的选基、组合建议和聚宽模拟交易项目。

## 场内 ETF 趋势轮动 V1

- [策略规格](docs/strategy/exchange_traded_etf_rotation_v1.md)
- [开发与验证步骤](docs/development/etf_rotation_roadmap.md)
- 通用选基内核：`src/strategy/etf_rotation.py`
- 聚宽可运行适配器：`joinquant/etf_rotation_strategy.py`

快速验证：

```bash
python -m pytest tests -q
python -m py_compile app.py src/strategy/etf_rotation.py src/engine/rotation_scanner.py joinquant/etf_rotation_strategy.py
streamlit run app.py
```

## 部署

`main` 分支已连接 GitHub 仓库 `dafenqi521/make-money`。如果 Streamlit Community Cloud 仍绑定该仓库的 `app.py`，推送 `main` 后会自动重新部署。

> 本项目输出仅用于策略研究和模拟验证，不构成收益承诺或个性化投资建议。
