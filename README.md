# 🤖 LLM-TradeBot

基于 LLM (DeepSeek) 的智能多 Agent 量化交易机器人，支持多时间框架同步、多 Agent 协作决策、技术指标自动计算及交易全链路审计。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ 核心特性

- 🤖 **Multi-Agent 协作**: 4 个专业 Agent 分工协作，从数据采集到风控执行全流程自动化
- ⚡ **异步并发**: 使用 `asyncio.gather` 并发获取多周期数据，减少 60% 等待时间
- 🎯 **智能决策**: 加权投票机制整合多周期信号，LLM 增强决策质量
- 🛡️ **严格风控**: 止损方向自动修正、资金预演、仓位控制、一票否决机制
- 📊 **全链路审计**: 每个环节的中间数据完整保存，方便复盘和调试
- 🔄 **双视图数据**: `stable_view` (已完成 K 线) + `live_view` (实时价格) 解决数据滞后

---

## 🚀 快速开始

### 启动流程

![快速开始流程](./docs/quick_start_flow_1766232535088.png)

### 详细步骤

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 配置环境

```bash
# 复制环境变量模板
cp .env.example .env

# 设置 API 密钥
./set_api_keys.sh
```

#### 3. 配置交易参数

```bash
# 复制配置文件模板
cp config.example.yaml config.yaml
```

编辑 `config.yaml` 设置交易参数：

- 交易对 (symbol)
- 最大仓位 (max_position_size)
- 杠杆倍数 (leverage)
- 止损止盈比例 (stop_loss_pct, take_profit_pct)

#### 4. 运行交易程序

```bash
# 测试模式 (模拟执行，推荐首次使用)
python main.py --test --mode continuous

# 实盘模式 (谨慎运行！)
python main.py --mode continuous
```

---

## 📁 项目结构

### 目录树形图

![项目结构](./docs/project_structure_tree_1766232597202.png)

### 目录说明

```text
LLM-TradeBot/
├── src/                    # 核心源代码
│   ├── agents/            # 多 Agent 定义 (DataSync, Quant, Decision, Risk)
│   ├── api/               # Binance API 客户端
│   ├── data/              # 数据处理模块 (processor, validator)
│   ├── execution/         # 交易执行引擎
│   ├── features/          # 特征工程模块
│   ├── monitoring/        # 监控和日志
│   ├── risk/              # 风险管理
│   ├── strategy/          # LLM 决策引擎
│   └── utils/             # 工具函数 (DataSaver, TradeLogger 等)
│
├── docs/                  # 项目文档
│   ├── data_flow_analysis.md          # 数据流转分析文档
│   └── *.png                          # 架构图和流程图
│
├── data/                  # 结构化数据存储 (按日期归档)
│   ├── market_data/       # 原始 K 线数据
│   ├── indicators/        # 技术指标
│   ├── features/          # 特征快照
│   ├── decisions/         # 决策结果
│   └── execution/         # 执行记录
│
├── logs/                  # 系统运行日志
├── tests/                 # 单元测试
├── config/                # 配置文件
│
├── main.py                # 统一程序入口 (Multi-Agent 循环)
├── config.yaml            # 交易参数配置
├── .env                   # API 密钥配置
└── requirements.txt       # Python 依赖
```

---

## 🎯 核心架构

### Multi-Agent 协作流程

系统由 5 个专业 Agent 协作完成交易全流程：

1. **🕵️ DataSyncAgent**: 异步并发获取多周期 (5m, 15m, 1h) K线数据，确保时间对齐
2. **👨‍🔬 QuantAnalystAgent**: 计算全量技术指标 (EMA, MACD, RSI, ATR 等)，提取 50+ 特征
3. **⚖️ DecisionCoreAgent**: 整合多周期趋势与震荡信号，加权投票决策
4. **🛡️ RiskAuditAgent**: 执行严格的风险审查（止损方向、仓位、杠杆、流动性）
5. **🚀 ExecutionEngine**: 负责交易信号的最终执行及全生命周期追踪

### 协作时序图

![Multi-Agent 协作时序](./docs/multi_agent_sequence_1766232561419.png)

### 数据流转架构

![数据流转架构](./docs/data_flow_diagram_1766231460411.png)

> 📖 **详细文档**: 查看 [数据流转分析文档](./docs/data_flow_analysis.md) 了解完整的数据流转机制和技术细节。

---

## 📄 数据全链路审计

### 数据存储结构

![数据存储层级](./docs/data_storage_hierarchy_1766232628608.png)

### 存储组织

系统自动将每一循环的中间过程记录在 `data/` 目录下，按日期组织，方便复盘和调试：

```text
data/
├── market_data/           # 原始多周期 K 线
│   └── {date}/
│       ├── BTCUSDT_5m_{timestamp}.json
│       ├── BTCUSDT_5m_{timestamp}.csv
│       ├── BTCUSDT_5m_{timestamp}.parquet
│       ├── BTCUSDT_15m_{timestamp}.json
│       └── BTCUSDT_1h_{timestamp}.json
│
├── indicators/            # 全量技术指标 DataFrames
│   └── {date}/
│       ├── BTCUSDT_5m_{snapshot_id}.parquet
│       ├── BTCUSDT_15m_{snapshot_id}.parquet
│       └── BTCUSDT_1h_{snapshot_id}.parquet
│
├── features/              # 提取的特征快照
│   └── {date}/
│       ├── BTCUSDT_5m_{snapshot_id}_v1.parquet
│       ├── BTCUSDT_15m_{snapshot_id}_v1.parquet
│       └── BTCUSDT_1h_{snapshot_id}_v1.parquet
│
├── context/               # 量化分析摘要
│   └── {date}/
│       └── BTCUSDT_quant_analysis_{snapshot_id}.json
│
├── llm_logs/              # LLM 输入上下文及 voting 过程
│   └── {date}/
│       └── BTCUSDT_{snapshot_id}.md
│
├── decisions/             # 最终加权投票结果
│   └── {date}/
│       └── BTCUSDT_{snapshot_id}.json
│
└── execution/             # 执行追踪
    └── {date}/
        └── BTCUSDT_{timestamp}.json
```

### 数据格式

- **JSON**: 可读性强，用于配置和决策结果
- **CSV**: 兼容性好，方便导入 Excel 分析
- **Parquet**: 高效压缩，用于大规模时序数据

---

## 🛡️ 安全提示

⚠️ **重要安全措施**:

1. **API 密钥**: 妥善保管，不要提交到版本控制
2. **测试模式先行**: 使用 `--test` 参数运行模拟交易，验证逻辑后再上实盘
3. **风险控制**: 在 `config.yaml` 中设置合理的止损和仓位限制
4. **权限最小化**: 为 API 密钥仅分配必要的合约交易权限
5. **监控告警**: 定期检查 `logs/` 目录，关注异常情况

---

## 📚 文档导航

| 文档 | 说明 |
|------|------|
| [README.md](./README.md) | 项目概览和快速开始 |
| [数据流转分析](./docs/data_flow_analysis.md) | 完整的数据流转机制和技术细节 |
| [API 密钥指南](./docs/API_KEYS_GUIDE.txt) | API 密钥配置说明 |
| [配置示例](./config.example.yaml) | 交易参数配置模板 |
| [环境变量示例](./.env.example) | 环境变量配置模板 |

---

## 🎉 最新更新

**2025-12-20**:

- ✅ **文档优化**: 新增 4 个可视化架构图，提升文档专业性和可读性
- ✅ **数据流转分析**: 完整的数据流转分析文档，包含 6 阶段详解
- ✅ **项目重命名**: 正式更名为 `LLM-TradeBot`
- ✅ **架构重构**: 迁移到纯 Multi-Agent 架构，放弃 legacy pipeline
- ✅ **全链路审计**: 实现从数据采集到决策执行的完整中间态归档
- ✅ **统一入口**: 合并所有运行脚本到 `main.py`

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

**由 AI 赋能，专注精准决策，开启智能量化新征程！** 🚀
