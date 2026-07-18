# make-money

只保留“场内ETF多资产趋势轮动”一个策略，并使用本项目自己的行情扫描、目标组合和本地模拟账户，不依赖聚宽。

## 当前能力

- 默认19只代表性场内ETF，也支持输入自定义6位ETF代码；
- 使用最新完整日线进行流动性、趋势、动量、波动和回撤筛选；
- 生成最多4只ETF的目标组合、目标权重和100份整数建议；
- 将目标组合与本地模拟持仓比较，生成买入、卖出和继续持有清单；
- 模拟佣金、滑点、现金约束和统一T+1规则；
- 历史回测采用“收盘生成信号、下一交易日开盘成交”，比较基准、回撤、月度收益和参数稳健性；
- 默认SQLite保存单个模拟账户；配置`DATABASE_URL`后使用PostgreSQL长期保存；
- 支持JSON备份恢复；
- 数据过期、扫描覆盖率不足或单只持仓读取失败时冻结自动交易。

## 运行

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m streamlit run app.py
```

浏览器打开 `http://localhost:8501`，依次完成：

1. 创建模拟账户；
2. 确认默认或自定义ETF候选池；
3. 点击“扫描并生成调仓清单”；
4. 查看数据日期、覆盖率、目标组合和调仓差额；
5. 执行模拟调仓，或者登记手工成交；
6. 持续保存账户JSON备份并观察净值和交易记录。
7. 在“历史回测”页设置区间、滑点和基准，运行组合回测与持仓数量稳健性检查。

## 核心文件

- `app.py`：唯一Streamlit入口；
- `src/data/fetcher.py`：本项目行情数据源；
- `src/strategy/etf_rotation.py`：选基、仓位和退出规则；
- `src/engine/rotation_scanner.py`：候选池扫描；
- `src/engine/paper_trading.py`：调仓清单和模拟执行；
- `src/engine/backtest.py`：无前视偏差的组合历史回测、基准和参数检查；
- `src/engine/portfolio.py`：现金、持仓、成交和T+1状态；
- `src/data/portfolio_db.py`：SQLite/PostgreSQL持久化、净值快照和JSON备份。

## 部署

`main` 分支已连接 GitHub 仓库 `dafenqi521/make-money`。Streamlit Community Cloud若仍绑定该仓库的 `app.py`，推送 `main` 后会自动重新部署。

项目只保留一个模拟账户，不提供多人登录或账户切换。Streamlit Community Cloud的本地磁盘不应视为可靠永久存储：

1. 在Supabase、Neon或其他PostgreSQL服务创建数据库；
2. 在Streamlit Community Cloud的Secrets中设置：

```toml
DATABASE_URL = "postgresql://用户名:密码@主机:5432/数据库?sslmode=require"
```

3. 重新部署后，侧栏应显示“账户存储：PostgreSQL 持久化”；
4. 不配置`DATABASE_URL`时继续使用本地SQLite，并应定期保存JSON备份。

不要把真实数据库密码提交到Git仓库。

> 本项目仅用于策略研究和模拟验证，不会连接券商自动下单，也不构成收益承诺或个性化投资建议。
