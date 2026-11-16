# 第三方来源、材料与公开边界

这个 checkout 发布用于审阅的代码、实验协议、来源记录和数值结果。它没有把所有研究资产重新分发，也没有为新增研究内容或第三方材料授予统一 MIT 许可。以下按本地 provenance 和保留的许可记录说明来源；获取和使用原材料仍应遵守各来源条款。

## 代码与依赖

| 来源 | 用途与公开记录 |
| --- | --- |
| [SD456/NAM-EQ-Match](https://github.com/SD456/NAM-EQ-Match) | 代码起点，固定 `1074151da12654fcf1125ed6443a1e53ea5b973d`。根上游代码与 [README 原文](../README.upstream.md)保留；实际本地修改版在 `vendor/NAM-EQ-Match/`。上游 MIT [LICENSE](../LICENSE)及 vendor 内许可保留。 |
| [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) | 运行已有 WaveNet 权重，版本 0.13.0；未随包分发安装环境。依赖版本见 `requirements-core.txt`、`requirements-lock-win.txt`，各依赖保留自己的许可。 |
| NumPy / SciPy / SoundFile / PyTorch 等 | 数值、音频、真实 NAM 推理；依赖不因根 MIT 声明而被重新授权。Windows/CUDA 实测范围见 `docs/environment-stage1.md`。 |

本地研究修改、文档及结果的许可范围没有由本次公开打包新增统一声明。保留上游 MIT 是保留既有权利通知，不意味着把第三方数据、论文、模型或全部新增研究成果自动纳入 MIT。

## 研究资产来源

| 来源 | 本地用途 / provenance 中记录的许可 | 本快照 |
| --- | --- | --- |
| [VoLum](https://github.com/guitarlum/VoLum)，Lum / Steffen Dangmann，固定 `a0b2f08b6be3fa2f7fdb58543f579d32e0ebe2f3` | 10 个 NAM、两条来源明确的箱体 IR，组合成 12 条候选链。模型来源记录 MIT；两条 IR 按作者来源记录 CC0。详见 `assets/stage2/models-ir-provenance.json` 与 `assets/test_models/model_provenance.json`。上游 IR 已有峰值归一化，不能称物理电平标定。 | 保留来源、哈希、许可文本；不上传 `.nam` 或 IR 音频 |
| [Guitar-TECHS](https://guitar-techs.github.io/)，Pedroza 等，ICASSP 2025；[数据记录](https://zenodo.org/records/14963133) | 真实 DI，provenance 记录 CC BY 4.0；固定片段、原父源和处理见 `assets/stage2/di-provenance.json`、`assets/stage3/di-provenance.json`、`assets/stage4/di-provenance.json`。早期多片段来自同录音，不能作为独立会话。 | 保留来源、署名和处理/哈希记录；不上传 DI 音频 |
| [IDMT-SMT-GUITAR](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html)；[发布记录](https://zenodo.org/records/7544110) | 单音与连续乐句 DI/XML，用于音高、起音及阶段 8。provenance 记录 CC-BY-NC-ND-4.0，见 `assets/stage6/data-provenance.json`、`assets/stage7/data-provenance.json`。不得视为 MIT 或开放商用素材。 | 保留来源、选择、许可字符串和摘要；不上传 WAV、XML、来源 PDF |
| 用户原曲 / Free Bird 相关 WAV 与用户提供论文 | 原曲只读保留；Free Bird 匹配仍暂停。论文是方向背景，当前检索不是盲逆建模论文的复现。 | 不上传原曲 WAV 或论文 PDF；可能保留历史路径、摘要和引用 |

阶段 8 完整材料表和父源分组记录在 `artifacts/stage8/real/locked-plan.json`，配置在 `experiments/stage8.json`；路径/摘要是溯源信息，不是发布对应原件的声明。许可证名称沿用本地来源证据，不能据此推断本地处理后的所有派生材料具有同样的再分发权。

## 本次保留与省略

保留研究 Python/PowerShell 源码、测试源码、冻结配置、Markdown 协议/结论、provenance/许可文本、CSV 和可审阅数字结果。9 个大于 1 MiB 的结果 JSON 以无损 `.json.gz` 保存；用 `scripts/materialize_review.py` 恢复后按原 JSON 哈希核验。四份历史报告/试听 Markdown 的本机路径已替换，`review-snapshot-manifest.json` 记录 source/public 双哈希；冻结锁原件未改。

省略原始/裁切/渲染音频、NAM 权重、IR、XML 标签、PDF、NPZ、虚拟环境、嵌套 `.git` 以及大型探针指纹缓存。原工程绝对路径、下载来源、文件名、哈希或缺失音频链接可能仍出现在历史证据中；保留它们用于审查原实验，不表示原件可从本仓库读取。

公开数值报告可供核对历史结论；缺少材料时无法独立验证每条来源哈希、重算全部渲染或跑完资产相关回归。历史 281 项通过不等于本快照 281 项均可无资产运行。补齐材料应单独遵循来源许可，在新冻结运行目录复算，保留旧结果和父源关系。
