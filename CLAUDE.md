# make-money — 场内ETF趋势轮动

## 项目边界

本项目只保留一个策略：场内ETF多资产趋势轮动。

- 二级市场买卖沪深ETF；
- 不做一级申赎、LOF套利、杠杆或反向产品；
- 日线信号只使用已完成交易日数据；
- 本项目行情扫描结果是唯一信号源；
- Streamlit负责扫描、调仓清单、本地模拟成交和绩效跟踪；
- 不使用聚宽，不连接真实券商。

## 运行

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m streamlit run app.py
```

## 核心文件

- `src/strategy/etf_rotation.py`：纯选基、配置、仓位和退出逻辑；
- `src/engine/rotation_scanner.py`：本地候选池数据扫描服务；
- `src/engine/paper_trading.py`：扫描到调仓清单与模拟执行；
- `src/engine/portfolio.py`：模拟账户领域模型；
- `src/data/portfolio_db.py`：模拟账户SQLite和JSON备份；
- `app.py`：唯一Streamlit入口。

## 开发约束

1. 不重新引入第二套交易策略、策略注册器或聚宽适配器；
2. 数据失败必须显式展示，禁止把缺失数据当作买卖依据；
3. 买入数量按100份取整，卖出检查T+1可卖数量；
4. 默认不连接真实券商，不承诺收益；
5. 修改后运行完整测试和Streamlit浏览器冒烟测试。
