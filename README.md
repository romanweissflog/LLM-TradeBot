# 🤖 LLM-TradeBot

基于 LLM (DeepSeek) 的智能多 Agent 量化交易机器人,支持多时间框架同步、多 Agent 协作决策、技术指标自动计算及交易全链路审计。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

```bash
# 复制环境变量模板
cp .env.example .env

# 设置 API 密钥
./set_api_keys.sh
```

### 3. 配置交易参数

```bash
# 复制配置文件模板
cp config.example.yaml config.yaml
```

### 4. 运行交易程序

```bash
# 测试模式 (模拟执行)
python main.py --test --mode continuous

# 实盘模式 (谨慎运行)
python main.py --mode continuous
```

---

## 📁 项目结构

```text
LLM-TradeBot/
├── src/                    # 核心源代码
│   ├── agents/            # 多 Agent 定义 (DataSync, Quant, Decision, Risk)
│   ├── api/               # Binance API 客户端
│   ├── execution/         # 交易执行引擎
│   ├── features/          # 特征工程模块
│   ├── monitoring/        # 监控和日志
│   ├── risk/              # 风险管理
│   ├── strategy/          # LLM 决策引擎
│   └── utils/             # 工具函数 (DataSaver, TradeLogger 等)
│
├── tests/                 # 单元测试
├── config/                # 配置文件
├── docs_organized/        # 项目文档
├── data/                  # 结构化数据存储 (按 Agent 归档)
├── logs/                  # 系统运行日志
│
├── main.py                # 统一程序入口 (Multi-Agent 循环)
└── requirements.txt       # Python 依赖
```

---

## 🎯 核心架构 (Multi-Agent Flow)

系统由多个专业 Agent 协作完成交易全流程：

1. **🕵️ DataSyncAgent**: 异步并发获取多周期 (5m, 15m, 1h) K线数据，确保时间对齐。
2. **👨‍🔬 QuantAnalystAgent**:
   - 计算全量技术指标 (EMA, MACD, RSI, ATR 等)。
   - 提取 50+ 特征及生成量化分析上下文。
3. **⚖️ DecisionCoreAgent**:
   - 整合多周期趋势与震荡信号。
   - 结合 DeepSeek LLM 进行智能决策增强。
4. **🛡️ RiskAuditAgent**: 执行严格的风险审查（仓位、止损、杠杆、市场流动性）。
5. **🚀 ExecutionEngine**: 负责交易信号的最终执行及全生命周期追踪。

---

## 📄 数据全链路审计

系统自动将每一循环的中间过程记录在 `data/` 目录下，方便复盘和调试：

```text
data/
├── data_sync_agent/       # 原始多周期 K 线 (JSON/CSV/Parquet)
├── quant_analyst_agent/   # 加工数据
│   ├── indicators/        # 全量技术指标 DataFrames
│   ├── features/          # 提取的特征快照
│   └── context/           # 量化分析摘要 (JSON)
├── decision_core_agent/   # 决策逻辑
│   ├── llm_logs/          # LLM 输入上下文及 voting 过程 (Markdown)
│   └── decisions/         # 最终加权投票结果 (JSON)
└── execution_engine/      # 执行追踪
    ├── orders/            # 单笔交易记录
    └── tracking/          # TradeLogger 详细追踪日志
```

---

## 🛡️ 安全提示

⚠️ **重要安全措施**:

1. **API 密钥**: 妥善保管,不要提交到版本控制。
2. **测试模式先行**: 使用 `--test` 参数运行模拟交易，验证逻辑后再上实盘。
3. **风险控制**: 在 `config.yaml` 中设置合理的全局止损。
4. **权限最小化**: 为 API 密钥仅分配必要的合约交易权限。

---

## 🎉 最新更新

**2025-12-20**:

- ✅ **项目重命名**: 正式更名为 `LLM-TradeBot`。
- ✅ **架构重构**: 迁移到纯 Multi-Agent 架构，放弃 legacy pipeline。
- ✅ **全链路审计**: 实现从数据采集到决策执行的完整中间态归档。
- ✅ **统一入口**: 合并所有运行脚本到 `main.py`。

---

**由 AI 赋能，专注精准决策，开启智能量化新征程!** 🚀
