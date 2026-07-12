# 定时通知配置指南

两种方式设置定时推送，任选其一。

## 方式一：Claude Code CronCreate（推荐）

在 Claude Code 会话中运行以下命令：

```
/cron "0 3 12 * * 1-5" "python scripts/check_and_notify.py --summary" --durable
/cron "30 6 14 * * 1-5" "python scripts/check_and_notify.py --summary" --durable
```

- 北京时间 12:00 和 14:30（UTC 4:00 和 6:30）
- 仅交易日（周一到周五）触发
- `--durable` 持久化到 `.claude/scheduled_tasks.json`，跨会话保持

## 方式二：Windows 任务计划程序

适合生产环境，不依赖 Claude Code 运行。

### 创建任务

1. 打开 **任务计划程序**（Task Scheduler）
2. 点击 **创建基本任务**
3. **名称**：`ETF 午间简报`
4. **触发器**：每天，12:00
5. **操作** → 启动程序：
   - 程序：`D:\python\python.exe`（你的 Python 路径）
   - 参数：`F:\MyProject\scripts\check_and_notify.py --summary 午间`
   - 起始于：`F:\MyProject`
6. 重复以上步骤创建第二个任务：
   - **名称**：`ETF 收盘前简报`
   - **触发器**：每天，14:30
   - **参数**：`F:\MyProject\scripts\check_and_notify.py --summary 收盘前`

### 确认 Python 路径

```powershell
(Get-Command python).Source
```

### 确认脚本能正常运行

```powershell
cd F:\MyProject
python scripts\check_and_notify.py --test
```

如果测试消息成功到达微信，说明配置正确。

## 通知内容说明

| 模式 | 命令 | 何时发送 |
|------|------|---------|
| 测试 | `--test` | 立即（手动触发） |
| 信号推送 | `--code 510300` | 仅在检测到可操作信号时发送 |
| 定时简报 | `--code 510300 --summary` | 每次都发送（午间/收盘概览） |
| 强制推送 | `--code 510300 --force` | 忽略去重，强制发送 |

## PushPlus 免费额度

- 200 条/天
- 定时简报每天 2 条 × 交易日 ≈ 44 条/月
- 信号推送仅在触发时发送，正常情况每月 < 20 条
- 总用量远低于免费额度
