# 吉他参考音色检索：Windows 研究审阅快照

这个分支供 Claude 或人工审阅已有代码、实验设计和数字结果：从参考吉他湿声中检索已有 NAM＋箱体 IR＋输入增益，使其能处理新的干声演奏。它没有训练新的 `.nam`，也没有证明可以从任意歌曲恢复原设备或真实旋钮。

分支：`research/windows-tone-retrieval-stage8`。上游起点为 [SD456/NAM-EQ-Match](https://github.com/SD456/NAM-EQ-Match) 的 `1074151da12654fcf1125ed6443a1e53ea5b973d`；[上游 README 原文](README.upstream.md)保留供溯源。

## 先分清两个代码位置

| 路径 | 本快照中的含义 |
| --- | --- |
| 根目录 `tonematch/`、`match.py`、`app.py`、启动器及 `tests/` | 上游参考，不是本地研究实际运行入口 |
| [`research/vendor/NAM-EQ-Match/`](research/vendor/NAM-EQ-Match/) | 本地实质修改后的真实核心，普通文件镜像，省略嵌套 `.git` |
| [`research/scripts/`](research/scripts/)、[`research/experiments/`](research/experiments/)、[`research/tests/`](research/tests/) | 实验、冻结配置与回归代码 |
| [`research/docs/`](research/docs/)、[`research/artifacts/`](research/artifacts/) | 协议、结论、锁定计划与数值证据 |

请先读 [审阅指南](research/REVIEW_GUIDE.md)和 [Claude 约束](CLAUDE.md)。`research/` 保留原工程内部相对目录结构；原工程未被这次公开打包改写。四份历史报告/试听 Markdown 的本机路径已替换，source/public 双哈希见[快照清单](research/review-snapshot-manifest.json)，冻结锁原件未改。历史音频链接不代表公开提供音频。

## 继承与实质变化

复用上游 NAM 后端、FIR/EQ 和两阶段搜索结构，并修复真实 `neural-amp-modeler==0.13.0` WaveNet 适配、卷积/搜索路径及输出一致性。本地 CLI 严格限制原生 48 kHz WaveNet；立体声需明确选择声道，候选须声明 `full-rig` 或带基础箱体 IR 的 `amp-only`。输出记录模型/IR 哈希、完整输入与输出增益，保留卷积尾部。原 GUI、一键启动器、Demucs 与 Bass/Mid/Treble/Hybrid 路线未针对本地 CLI 分支适配。

阶段 8 的 ERB24→23 维共享 LW/OAS 协方差是独立研究评分，**未替换 legacy CLI 评分**。研究阶段的自动新增 EQ 保持旁路；**legacy CLI 仍会生成校正 IR，不能理解成整个仓库已禁用 EQ**。

## 阶段 8 的结论

192 段真实 DI×12 条链×两档输入 trim＝4608 个独特真实 NAM 渲染组合；五个器材组内部留组验证形成 1520 个任务、15200 行方法/增益模式结果。未知 trim 的 E1 宏平均遗憾值从 0.2792 降到 0.1481（下降 46.96%），5/5 组及 20 次预定训练重抽均改善。但三类素材的逐任务退化 P90 超过预设 0.05 上限，**未通过全部门槛，不晋升、不替换生产评分**。

见 [简明结论](research/docs/stage8-conclusions.md)、[完整发现](research/docs/stage8-findings.md)、[预先冻结协议](research/docs/stage8-protocol.md)。遗憾值不是听感百分比，20/20 不是部署成功概率；本批所有录音都已观察，下一步确认需要新录音。Free Bird、实体设备、NAM_LV2 与 Orange Pi 验证尚未完成。

## 阅读与运行边界

这是可审阅代码与数字结果的快照，**不是不带素材也能完整复现的发行版**。不上传原曲 WAV、DI、箱体 IR、NAM 权重、XML 标签、PDF、NPZ、虚拟环境或大型探针指纹缓存；保留 provenance、许可文本、协议和数值结果。超过 1 MiB 的结果 JSON 无损存为 `.json.gz`，需先恢复再交给原分析器、哈希校验或普通文本审阅：

```powershell
rtk python research/scripts/materialize_review.py
```

该恢复工具仅需标准 Python；恢复原 JSON 不会补齐缺失的音频或缓存。命令中的 RTK 只是本地命令输出代理，不是研究依赖；未安装时省略 `rtk`，运行相同 Python 命令即可。运行研究代码时以 `research/` 为工作目录，依赖及 Windows 环境记录见 [`requirements-core.txt`](research/requirements-core.txt)、[`requirements-lock-win.txt`](research/requirements-lock-win.txt) 与[环境记录](research/docs/environment-stage1.md)。不要把根目录上游 `requirements.txt` 当作研究环境锁。

本次在缺少音频/权重/IR 的公开快照、使用本地已有虚拟环境运行 48 项阶段 8 数值测试及 2 项恢复工具测试，50 通过、0 失败/错误/跳过，见 [JUnit 记录](research/review-portable-tests.xml)。这不是独立新装依赖验证，也不是完整真实渲染复现。完整重渲染、保护文件校验及依赖真实资产的测试需按 provenance 另备材料；历史“281 通过”是原本地全素材环境的记录。

来源和各自许可见 [THIRD_PARTY_AND_DATA.md](research/THIRD_PARTY_AND_DATA.md)。根目录 MIT [LICENSE](LICENSE)保留上游许可声明，不是为新增研究内容或第三方数据统一授予 MIT。
