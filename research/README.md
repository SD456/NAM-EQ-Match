# 吉他参考音色检索实验

第八阶段已完成192段DI×12链×两档增益的4608个真实NAM渲染组合及五组留琴检索。低维共享LW协方差使带EQ参考的宏平均遗憾值0.2792→0.1481，5/5组及20/20训练重抽改善，但三类素材的尾部退化超预设上限，整体未通过，不替换生产评分。详见[简明结论](docs/stage8-conclusions.md)、[完整数值](docs/stage8-findings.md)、[覆盖表](docs/di-coverage.md)。保留全部旧结果，自动EQ旁路，Free Bird暂停。

第八阶段完整重跑须新目录，并先冻结再执行（已有完成结果拒绝覆盖）：

```powershell
rtk .\.venv\Scripts\python.exe scripts/stage8_benchmark.py --out artifacts/stage8/repeat001 --freeze
rtk .\.venv\Scripts\python.exe scripts/stage8_benchmark.py --out artifacts/stage8/repeat001
```

`scripts/stage8_analyze.py`仅绑定默认正式目录`artifacts/stage8/real`，生成事后报告，不重新选链。完整校验记录见[final-audit.json](artifacts/stage8/final-audit.json)。

第七阶段已冻结后完成起音峰合并比较：18段新连续DI按乐句分组。最早峰/80ms使新留出多报169→93、macro F1 .5335→.6717，但新增1个严格未匹配事件且快速同音样本为0，结论为证据不足，不晋升。128ms音高候选、旧基线及结果保留。详见[stage7-findings.md](docs/stage7-findings.md)，待补素材见[stage7-recording-spec.md](docs/stage7-recording-spec.md)。

第六阶段已用公开带音高/起止标签的IDMT单音DI，先冻结再校准：78条Ibanez用于选参、78条Fender留出。128ms音高窗/0.8门槛将留出覆盖提高到99.85%；64ms主起音多报66次、F1=0.7027未达标，保留基线，先研究重复检峰。结论：[stage6-conclusions.md](docs/stage6-conclusions.md)，完整报告：[stage6-findings.md](docs/stage6-findings.md)。

第五阶段已完成真实三档力度 DI 检索、固定增益补录规范及冻结后的条件化评分开发回放。未核实完全合格的公开力度数据；两个条件法在32条旧素材上全部拒答，固定诊断确认概率门槛0.8无帧通过。保留多DI平滑谱基线，先校准音高/起音前端，Free Bird继续暂停。结论：[stage5-conclusions.md](docs/stage5-conclusions.md)，完整数值：[stage5-findings.md](docs/stage5-findings.md)，补录：[stage5-recording-spec.md](docs/stage5-recording-spec.md)。

第四阶段先冻结方案后完成真实实验，已补四份不同录音文件。输入trim与未知源电平存在结构性不可辨识；新空假设EQ规则全旁路、温和EQ检出0/8，研究门槛未过。旧基线保持，Free Bird仍暂停。当前结论：[stage4-conclusions.md](docs/stage4-conclusions.md)，完整表格：[stage4-findings.md](docs/stage4-findings.md)。

第三阶段已完成独立角色的多DI探针对照；多DI平滑谱改善了带EQ的链首选，增益漂移和凭空EQ仍未联合解决。继续暂停Free Bird，保留旧评分及第二阶段结果。当前结论见 [stage3-conclusions.md](docs/stage3-conclusions.md)，详表见 [stage3-findings.md](docs/stage3-findings.md)。

第二阶段已完成公开素材收集与64例真实跨DI基准。当前评分在跨演奏+固定后级EQ时首选命中8/16，尚不能可靠自动确定整体音色；继续暂停Free Bird。结论与下一步见 [stage2-conclusions.md](docs/stage2-conclusions.md)，完整数值见 [stage2-findings.md](docs/stage2-findings.md)。

第一阶段：Windows 本地 NAM 推理、两阶段检索与受限后 EQ 的可靠性验证。原始素材位于 `WAV_Data/`，程序不会就地改写它们；SHA256 基线在 `docs/original-wav-sha256.json`。

## 当前范围

- 复用 `vendor/NAM-EQ-Match`，上游固定 `1074151da12654fcf1125ed6443a1e53ea5b973d`；保留上游 MIT LICENSE 和 Git 历史。
- 仅本地 CLI、原生 48 kHz WaveNet、单声道处理。双声道 WAV 必须明确选择 left/right/mean。
- 候选必须说明 `full-rig`，或 `amp-only` 并指定基础箱体 IR；IR 不自动归一化。
- 首版只有一个最小相位后 EQ：默认 2048 taps、半倍频程平滑、±4.5 dB；边界验证针对实际 float32 校正 IR。
- 旧 WaveNet 的 head 字段在内存中适配 NAM 0.13，权重不变；legacy gated 模型明确拒绝。CUDA 推理禁用 TF32，以便与 CPU／插件 float32 对照。
- 输出是现有 `.nam` 副本、外置 IR、精确增益与 JSON 设置，不是新训练模型。未验证 Free Bird 音色相似度或 Orange Pi 实时性能。
- 上游 `app.py`、一键启动脚本和官方 NAM 插件 Bass/Mid/Treble/Hybrid 工作流不属于这个 CLI 分支，未适配，不能用于本阶段。

## 环境与回归

在本目录打开 PowerShell。无需激活虚拟环境，直接使用专属解释器：

```powershell
rtk .\.venv\Scripts\python.exe -X utf8 -m pytest -q --junitxml artifacts/stage1/pytest.xml
rtk .\.venv\Scripts\python.exe -X utf8 assets/test_models/verify_model.py
```

初次搭建或重建使用 `scripts/setup-env.ps1`；实际版本记录在 `requirements-lock-win.txt` 和 `docs/environment-stage1.md`。不要用全局 `pip`，也不要运行上游自动安装脚本。真实 CUDA 回归需要本机 RTX GPU；测试会明确失败而非静默退回 CPU。此处的命令不自动发起参考歌曲检索。

## 检索入口（下一阶段有素材后使用）

候选 manifest 是 JSON 数组。路径相对于 manifest 文件所在目录，`amp-only` 必须给出箱体；完整链禁止重复箱体。例如：

```json
[
  {"path": "captures/amp.nam", "rig_type": "amp-only", "cab_ir_path": "irs/cab.wav"},
  {"path": "captures/full_rig.nam", "rig_type": "full-rig"}
]
```

```powershell
rtk .\.venv\Scripts\python.exe -X utf8 vendor/NAM-EQ-Match/match.py --target work/reference_mono.wav --di work/probe_di.wav --manifest work/candidates.json --out artifacts/run001 --device cuda
```

`--target-start 5 --target-end 17 --target-channel left` 可从原 WAV 只读选段，声道应先经人工检查，不预设 Free Bird 应选左声道。输出目录必须新建或为空，已有结果不会被覆盖。

每条候选的 `*_settings.json` 记录完整输入增益（DI 校准＋搜索）、唯一输出增益、模型及 IR 哈希。用新 DI 复现时沿用这套输入校准，不逐文件重新归一化：

```powershell
rtk .\.venv\Scripts\python.exe -X utf8 vendor/NAM-EQ-Match/render.py --settings artifacts/run001/rank01_NAME_settings.json --di work/new_di.wav --out artifacts/new_di_render.wav --device cuda
```

目前预设使用绝对路径；移动输出目录后需相应更新 JSON 中路径。模型、IR 内容变更会被哈希检查拒绝。FLOAT WAV 允许样本超过 1，重放新 DI 时不加隐藏限幅；试听前检查峰值，必要时显式降低输出。

## 信号链和 NAM_LV2

原 DI → **完整输入增益** → 裸 NAM → **playback IR**（100% wet、关闭归一化）→ **输出增益**。

- `correction_ir` 只包含受限微调；`playback_ir` 对 amp-only 包含基础箱体×校正，对 full-rig 只含校正。只加载 playback IR，避免叠加两次箱体。
- `preview_di.wav` 保存实际预览使用的原始电平片段。完整卷积尾部保留，长度见 `tail_frames`。
- NAM_LV2 可能额外应用模型元数据增益；设置中给出按默认 12 dBu 假定计算的控件建议。实际目标插件、编译配置、IR 加载器仍须另验，`lv2_verified` 当前固定为 false。
- 检索分数是启发式距离，不是置信概率；不同音符/演奏会影响它。真实可用性需下一阶段跨 DI 和盲听验证。

## 下一阶段素材

1. 8–12 个来源、许可、采样率和箱体范围明确的真实 NAM；包含至少一个 full-rig、一个 amp-only＋真实箱体 IR，覆盖 clean/crunch/lead。
2. 三段真实吉他 DI：A/B 独立演奏用于交叉验证，C 留出试听；固定录音硬件增益，覆盖弱强拨弦、延音和主音演奏。现在的合成测试探针不能替代它们。
3. 从 Free Bird 第 5 秒后听选三个 8–12 秒不重叠片段；两个开发、一个留出，并记录声道选择理由。原 WAV 保留。
4. 实际 NAM_LV2 版本/编译来源、48 kHz 宿主及最终 IR 加载器设置，用于部署一致性对照。

## 第二阶段复算与素材

- `assets/stage2/models-ir-provenance.json`：10个MIT NAM、2个来源作者CC0箱体IR，原件/上游派生处理、许可和SHA256完整记录。IR上游已有峰值归一化，不能声称物理电平校准。
- `assets/stage2/di-provenance.json`：Guitar-TECHS（Pedroza等，ICASSP 2025），CC BY 4.0，原始文件与A/B/C裁切说明。三段来自同一次录音，硬件增益未在源中给出。
- `experiments/stage2.json`：真实结果前冻结的候选、目标、条件及门槛。未调生产权重。
- `artifacts/stage2/real_fp64`：正式结果、600条候选指纹及锁定方案；`artifacts/stage2/real`是早期被数值一致性检查拦住的未完成试跑，不能混入统计。

```powershell
rtk .\.venv\Scripts\python.exe -X utf8 scripts/stage2_analyze.py
rtk .\.venv\Scripts\python.exe -X utf8 -m pytest -q --junitxml artifacts/stage2/pytest.xml
```

重跑完整研究应另设新目录，脚本禁止覆盖已经完成的结果：

```powershell
rtk .\.venv\Scripts\python.exe -X utf8 scripts/stage2_benchmark.py --out artifacts/stage2/repeat001 --device cuda
```

## 第三阶段复算

`assets/stage3/di-provenance.json`记录P0–P2/R0–R1/H0源范围及SHA；样本不重叠，但都是同次录音，未达成跨录音验证。`experiments/stage3.json`与`docs/stage3-protocol.md`为预先冻结的研究配置；生产评分未替换。

```powershell
rtk .\.venv\Scripts\python.exe -X utf8 scripts/stage3_analyze.py --self-check
rtk .\.venv\Scripts\python.exe -X utf8 -m pytest -q --junitxml artifacts/stage3/pytest.xml
```

完整重跑需新的输出目录；先有第二阶段基线及`artifacts/stage3/baseline-sha256.json`：

```powershell
rtk .\.venv\Scripts\python.exe -X utf8 scripts/stage3_benchmark.py --out artifacts/stage3/repeat001 --device cuda
```

默认正式结果在`artifacts/stage3/real`。`audit_stage3_scores.py`是该默认运行的独立目标重渲染审计，输出`reference-features.json`；已存在时拒绝覆盖。后处理同样绑定默认正式目录，并核对结果与源文件SHA。

## 第四阶段复算

`assets/stage4/di-provenance.json`记录四份新的Guitar-TECHS源文件及九段固定片段：02探针、03校准、04测试、05留出；全部同乐手/吉他，硬件增益和真实力度未知。`docs/stage4-protocol.md`、`experiments/stage4.json`在NN实验前冻结。

```powershell
rtk .\.venv\Scripts\python.exe -X utf8 scripts/stage4_analyze.py --self-check
rtk .\.venv\Scripts\python.exe -X utf8 -m pytest -q --junitxml artifacts/stage4/pytest.xml
```

数学反例保存在`artifacts/stage4/gain_witness`，独立录音试验在`artifacts/stage4/independent`；两个runner均禁止覆盖已完成结果。新空假设规则未通过真链/真增益的误报与功效双门槛，选中链EQ组按预先停止规则未验证。代码回归通过不代表科学门槛通过。

## 第五阶段记录与复核

`docs/stage5-data-search.md` 汇总公开证据；`assets/stage5/recording/` 提供空表与36条提示卡。按 `docs/stage5-recording-spec.md` 录制原始固定增益 DI，不依据响度重贴力度标签。当前缺少合格数据，真实力度试验仍待测。

`experiments/stage5.json` 与 `docs/stage5-protocol.md` 在目标实验前冻结；`artifacts/stage5/development/locked-plan.json` 绑定代码、版本及素材。正式结果已存在，runner会拒绝覆盖。历史运行顺序为先 `scripts/stage5_benchmark.py --freeze`，再运行该脚本；后续修改必须新建冻结运行，不能就地改锁续跑。

```powershell
rtk .\.venv\Scripts\python.exe -X utf8 -m pytest -q --junitxml artifacts/stage5/pytest.xml
```

主结果在 `artifacts/stage5/development/results.json`，展开表在 `retrieval.csv`；固定参数门槛拆解在 `artifacts/stage5/conditioning-diagnostic.json`。诊断脚本 `scripts/stage5_condition_diagnostic.py` 也拒绝覆盖既有诊断。原基线32个分数矩阵逐元素一致，157项回归通过；新方法全拒答不能当作距离质量验证。

## 第六阶段记录与复核

`assets/stage6/data-provenance.json` 逐条记录156对原始DI/XML的来源、许可、范围、CRC及SHA256。`experiments/stage6.json`和`docs/stage6-protocol.md`由独立的 `stage6_benchmark.py --freeze` 锁定，再运行校准；正式结果已经存在，runner拒绝覆盖。不要就地改锁重跑，本轮留出现在已被观察。

```powershell
rtk .\.venv\Scripts\python.exe -X utf8 scripts/stage6_analyze.py
rtk .\.venv\Scripts\python.exe -X utf8 -m pytest -q --junitxml artifacts/stage6/pytest.xml
```

`stage6_analyze.py`只核验摘要并从既有结果导出表格、图与后验诊断，不重跑音频或改选参数。正式结果在 `artifacts/stage6/calibration/results.json`，先于留出保存的选择在 `selected-setting.json`，完整参数表在 `pitch-grid.csv`。190项回归通过；音高覆盖达标而主起音未达标，不能替代生产前端或开始Free Bird匹配。

## 第七阶段记录与复核

`assets/stage7/data-provenance.json`记录18对原始连续DI/XML的来源与摘要；同一乐句的三个吉他版本没有跨组。`experiments/stage7.json`与协议由独立`--freeze`命令先行锁定，正式结果已经存在，runner拒绝覆盖。`artifacts/stage7/onsets/selected-setting.json`在新留出分析前保存。

```powershell
rtk .\.venv\Scripts\python.exe -X utf8 scripts/stage7_analyze.py
rtk .\.venv\Scripts\python.exe -X utf8 -m pytest -q --junitxml artifacts/stage7/pytest.xml
```

分析脚本只核验摘要并导出完成结果的表、图与后验诊断；不重新检测或选参。正式结果在`artifacts/stage7/onsets/results.json`，完整汇总为`method-grid.csv`。232项回归通过；历史156单音的两窗预测逐条复现，旧结果未改。当前新留出以后也属于已观察数据，后续须新锁与新确认录音。Free Bird、湿声检索和自动EQ保持暂停。
