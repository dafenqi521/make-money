# make-money — 场内ETF趋势轮动

## 项目边界

本项目只保留一个策略：场内ETF多资产趋势轮动。

- 二级市场买卖沪深ETF；
- 不做一级申赎、LOF套利、杠杆或反向产品；
- 日线信号只使用已完成交易日数据；
- 本地 Streamlit 生成排名和目标组合；
- 聚宽脚本负责回测和模拟交易。

## 运行

```bash
pip install -r requirements.txt
python -m pytest tests -q
streamlit run app.py
```

## 核心文件

- `src/strategy/etf_rotation.py`：纯选基、配置、仓位和退出逻辑；
- `src/engine/rotation_scanner.py`：本地候选池数据扫描服务；
- `joinquant/etf_rotation_strategy.py`：聚宽执行适配器；
- `app.py`：唯一 Streamlit 入口；
- `docs/strategy/exchange_traded_etf_rotation_v1.md`：策略规格；
- `docs/development/etf_rotation_roadmap.md`：开发路线。

## 开发约束

1. 不重新引入第二套交易策略或策略注册器；
2. 横截面组合逻辑不得塞进旧的单标的回测模型；
3. 数据失败必须显式展示，禁止把缺失数据当作买入依据；
4. 买入数量按100份取整，卖出检查可卖数量；
5. 默认不连接真实券商，不承诺收益；
6. 修改后运行完整测试和 Streamlit 页面冒烟测试。
