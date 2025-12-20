# Data目录索引

## 📂 目录结构

```
data/
├── multi_agent/                      # 多Agent架构数据（v2.0）
│   ├── agent1_data_sync/            # 数据同步官
│   ├── agent2_quant_analysis/       # 量化分析师
│   ├── agent3_decision_core/        # 决策中枢
│   ├── agent4_risk_audit/           # 风控审计官
│   ├── agent_integration/           # 多Agent集成
│   ├── execution/                   # 执行结果
│   └── daily_stats/                 # 每日统计
│
├── legacy/                          # 旧版数据（向后兼容）
│   ├── step1/ ... step9/           # 原9步数据流
│   └── ...
│
├── diagnostic_reports/              # 诊断报告
│
├── MULTI_AGENT_DATA_ARCHIVE.md     # 多Agent数据归档文档
└── LIVE_TRADING_DATA_ARCHIVE_SUMMARY.md  # 旧版归档总结
```

## 📚 文档说明

### 1. 多Agent架构数据归档（推荐）

**文档**: [MULTI_AGENT_DATA_ARCHIVE.md](MULTI_AGENT_DATA_ARCHIVE.md)

- ✅ 6步数据流
- ✅ Agent分层数据
- ✅ JSON + TXT双格式
- ✅ 完整审计日志
- ✅ 性能统计

**适用于**: v2.0多Agent架构

### 2. 旧版数据归档（兼容）

**文档**: [LIVE_TRADING_DATA_ARCHIVE_SUMMARY.md](LIVE_TRADING_DATA_ARCHIVE_SUMMARY.md)

- ✅ 9步数据流
- ✅ JSON/CSV/Parquet
- ✅ 向后兼容

**适用于**: v1.0单体架构

## 🔄 数据迁移

旧版数据已归档到 `data/legacy/` 目录，新系统数据保存在 `data/multi_agent/` 目录。

两套系统可并行运行，互不冲突。

## 📊 使用建议

| 场景 | 推荐方案 |
|------|---------|
| 新项目 | 使用 `multi_agent/` 数据结构 |
| 旧项目迁移 | 逐步迁移到 `multi_agent/`，保留 `legacy/` 用于回溯 |
| 数据分析 | 优先使用 `multi_agent/` 的JSON格式 |
| 性能监控 | 查看 `multi_agent/daily_stats/` |

---

**更新时间**: 2025-12-19  
**版本**: v2.0
