# 研究审阅指南

本目录镜像本地 `NAM_guitar_modeling` 的代码、协议与数字证据，保持内部目录关系。它支持方法与实现审查；原始音频、权重、标签及大型缓存未发布，因此不承诺仅凭此 checkout 完整重算实验。

## 建议阅读顺序

1. [根 README](../README.md)与 [CLAUDE.md](../CLAUDE.md)：辨认根上游代码和 `vendor/NAM-EQ-Match/` 的实际修改版。
2. [当前目标与试听背景](docs/current-status-and-listening.md)、[第 8 阶段结论](docs/stage8-conclusions.md)：先理解目标、失败门槛和未完成项。历史本机音频链接在本快照不可播放。
3. [第 8 阶段协议](docs/stage8-protocol.md)、[协方差提案](docs/stage8-covariance-proposal.md)、[预运行审阅](docs/stage8-preflight-review.md)与 [`experiments/stage8.json`](experiments/stage8.json)：区分事前指定与事后诊断。
4. [`artifacts/stage8/real/locked-plan.json`](artifacts/stage8/real/locked-plan.json)、[`selection-locked.json`](artifacts/stage8/real/selection-locked.json)、[`results.json`](artifacts/stage8/real/results.json)、[`artifacts/stage8/analysis.json`](artifacts/stage8/analysis.json)、[完整发现](docs/stage8-findings.md)：核对冻结摘要、选择与评价。大 JSON 链接须先恢复。
5. [`scripts/stage8_metrics.py`](scripts/stage8_metrics.py)、[`scripts/stage8_quality.py`](scripts/stage8_quality.py)、[`scripts/stage8_benchmark.py`](scripts/stage8_benchmark.py)、[`scripts/stage8_analyze.py`](scripts/stage8_analyze.py)及对应 `tests/test_stage8_*.py`：检查实现如何兑现协议。
6. [`vendor/NAM-EQ-Match/`](vendor/NAM-EQ-Match/)、[`docs/upstream-changes.patch`](docs/upstream-changes.patch)、[阶段 1 环境](docs/environment-stage1.md)和阶段 2–7 的 protocol/conclusions：理解真实 NAM、FIR/search、输入标定和历史失败，避免把辅助前端结果当作生产音色收益。

## 先恢复大 JSON

从仓库根运行：

```powershell
rtk python research/scripts/materialize_review.py
```

RTK 只是本地命令输出代理，不是研究依赖；未安装时省略 `rtk`，使用相同 Python 命令。

工具用标准 Python 检查上传/public 哈希，无损恢复 9 个 `.json.gz`，拒绝覆盖已编辑文件。恢复后原 JSON 字节与 SHA256 才能用于历史摘要核验，压缩包哈希不同于原 JSON。四份历史报告/试听 Markdown 替换了本机路径，source/public 双哈希记录在 [`review-snapshot-manifest.json`](review-snapshot-manifest.json)，冻结锁原件未改；恢复后第 8 阶段 13 项 source_hashes 已全部匹配。某个哈希匹配不等于缺失的 WAV、模型、标签或缓存已验证。不要编辑冻结锁或历史结果来迎合当前目录。

## 阶段 8 的可核查规格

| 项目 | 已冻结设置 / 实际范围 |
| --- | --- |
| 输入与候选 | 192 真实 DI、12 条 NAM/IR 链、0/+6 dB trim，共 4608 个独特渲染组合 |
| 分组 | Ibanez、Fender 型号族、Aristides、LesPaul、Sire，训练排除整器材组 |
| 查询与裁判 | 95 个 Q 原件、83 个 H 原件；Q/H 不共享父原件，选择不读 H |
| 任务与结果 | 4 已知目标×2 真 trim×2 后 EQ×95 Q＝1520；5 方法×2 增益模式＝15200 行 |
| E0 / E1 | 无后 EQ / 固定 1200 Hz、+3 dB 高架；候选不加自动 EQ |
| 主候选 | 24 ERB 对数能量经固定 Helmert 到 23 维；逐链 LW 收缩后跨链等权平均，按 trim 共享 |
| 对照 | 同数据 spectrum160、同低维同中心 Euclidean、diagonal；OAS 仅敏感性对照 |
| 汇总 | 父源内平均、父源等权、类型等权、五器材组等权；尾部另按琴×类型逐任务检查 |

主指标是同 DI 输出多分辨率 STFT spectral convergence（SC），先按协议构造 H 聚合质量表，再计算 `R = D_H(selected) - min D_H(candidate)`。不能让每个 H 各选 oracle；低 R 也不等于候选库的绝对误差低。

未知 trim、E1：spectrum160 的 R=0.2792，LW=0.1481，改善 0.1311/46.96%；同条件链命中宏平均 46.79%→69.79%。5/5 组和 20 次训练重抽改善。然而 Ibanez 单音、Fender 单音、LesPaul 乐句的退化 P90 分别 0.1215、0.1619、0.1150，均超过 0.05；完整门槛失败，不能 promote。OAS 接近也不能替换指定 LW。

LW E1 绝对 SC=0.2621，库内 oracle 约 0.1140，仍有明显差距。已知 trim 后 SC=0.2096、R=0.0955，链也重新排列，所以差值不等于纯增益效应。矩阵条件数 134.76–270.36、20/20 重抽有效只支持本次数据上的数值与组成稳定性，不是独立部署保证。

## 优先待审问题

- 是否隔离训练/Q/H，保留父源关联及全部链/trim 的联动块？相同内容跨琴可入训练、接口/拾音器/乐手/会话混杂、Sire 历史校准分别限制了哪些结论？
- ERB 归一、去电平投影、按链收缩再平均、logdet 和数值下限是否与协议一致？降维、对角尺度和完整协方差的收益各有多少证据？
- 选择先锁、H 后读以及 oracle/宏平均/尾部实现是否一致？任务复制是否被误当样本量？
- 退化集中在哪些链、增益档、音域和素材类型？哪些可被增益约束或候选保留/拒答解释，哪些即使已知 gain 仍存在？只提出新冻结实验，不在旧 H 上继续选参。
- 真实 NAM 0.13、FIR 复用、哈希/尾部/增益重放的边界是否完整？不要通过根目录上游 `match.py` 推断本地 CLI 行为。

## 检查层级与复现限制

不带素材可运行阶段 8 的指标、质量函数与基准汇总单元检查；从 `research/`、在安装相应数值依赖后执行：

```powershell
rtk python -m pytest -q tests/test_stage8_metrics.py tests/test_stage8_quality.py tests/test_stage8_benchmark.py tests/test_materialize_review.py --junitxml=review-portable-tests.xml
```

本次在不含真实音频、NAM 权重或 IR 的 `research/` 快照运行上述检查，使用本地已有虚拟环境，50 通过、0 失败/错误/跳过（48 项原数值测试＋2 项恢复工具测试），见 [JUnit 记录](review-portable-tests.xml)。没有独立新装依赖验证。通过只说明被覆盖的数值、汇总与恢复逻辑，不是完整真实渲染复现。完整 281 项通过是原本地全素材环境的历史记录。其他测试会读取资产、XML、冻结来源或要求 CUDA；缺材料时不能宣称全套可跑。

恢复结果 JSON 后可以交叉核算已有数字。原分析脚本可能还核验缺失材料或默认正式目录，故不要把“能读 JSON”写成“所有分析脚本无需素材可重跑”。完整重渲染须按 provenance 补齐合法素材与环境，在新目录先冻结再运行，禁止覆盖既有完成结果。大型探针/指纹缓存未发布，不能承诺零计算续跑。

本批已观察完毕。下一轮应补固定录音增益、明确父源关系的独立会话，针对失败音域和链型；真实力度、带标签真实和弦/闷音及快速同音仍缺。不得继续调旧 H，不把 20/20 解释成概率，也不以平均改善开启自动 EQ、Free Bird 或硬件部署。研究 EQ 旁路与 legacy CLI 仍生成校正 IR 是两个事实。
