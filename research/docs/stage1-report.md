# 第一阶段实施与验证报告

日期：2026-11-15，Windows 主机。范围为可靠的推理、检索执行、受限 EQ 和导出回放基础；没有运行 Free Bird 检索，没有改写或裁剪原始 WAV。

## 结果

- 独立 `.venv` 已搭建，Python 3.12.7 / torch 2.10.0+cu128 / CUDA runtime 12.8 / neural-amp-modeler 0.13.0。RTX 5060 实际能力 sm_120，真实 CUDA 前向推理通过。
- 根目录 `pytest`：**76 passed，0 failed，0 skipped，24.07 秒**；其中 DSP 34 项、检索/后端 31 项、真实模型与整链 11 项。证据：`artifacts/stage1/pytest.xml`。
- `pip check`：No broken requirements found；安装脚本在现有环境重用执行通过，没有重新下载大包。
- 三份原始 WAV 的大小和 SHA256 与修改前一致，基线见 `original-wav-sha256.json`；该检查也包含在 76 项回归中。
- 完整依赖锁：`../requirements-lock-win.txt`；环境细节：`environment-stage1.md`。

## 来源与复用

`vendor/NAM-EQ-Match` 固定上游提交 `1074151da12654fcf1125ed6443a1e53ea5b973d`，保留原 Git 历史和 MIT LICENSE。原有音频工具、统计特征、两阶段搜索、PyTorch 后端及 FIR 设计继续使用。当前为本地 CLI 分支，Gradio、云模型服务、Demucs、官方 NAM 插件音调旋钮和 Hybrid 工作流没有启用；旧 GUI 入口会明确提示使用 CLI。

源代码差异保存在 `upstream-changes.patch`。若在一个干净的相同提交上复建，可应用该补丁；它包括新 `render.py`。根目录 README、tests、scripts、assets 和环境文件为本实验配套文件。

## 主要修复

| 范围 | 最终行为 |
| --- | --- |
| 搜索 | 粗排只筛选，最终榜单仅包含相同精排条件下的候选；不混用粗分。默认粗增益步长6 dB，精排附近±3 dB、1 dB步长，去重并限制在用户范围内。 |
| 独立演奏 | DI 与参考可以不同长度、不同演奏；参考独立提取统计，不按DI索引裁切。粗、精排各自所有候选使用同一DI片段。 |
| 静音及短输入 | 修复短RMS窗口、短active段、静音前缀导致粗排中断；空/非有限输入和静音搜索明确拒绝。RMS不再人为添加静音底噪。 |
| 特征 | 不先拼接删除静音后的波形；时间变化量只统计原始相邻有效帧。保留统计评分基线，明确它不具备真正EQ不变性。 |
| 声道 | 双声道文件要求显式left/right/mean，避免反相抵消被静默引入。 |
| 模型 | 只支持明确声明原生48k的WaveNet；LSTM、容器、缺失或其他SR明确拒绝。旧head配置仅内存适配NAM0.13；legacy gated=True拒绝；严格检查全部权重被消费。 |
| 精度 | CUDA后端关闭cuDNN/matmul TF32，使用float32；避免默认TF32影响CPU/GPU对照。 |
| 信号链 | amp-only需要指定基础箱体IR，full-rig禁止重复箱体。搜索、预览、导出均使用同一完整链，另提供裸NAM接口。 |
| EQ | 仅一个最小相位校正，默认2048 taps、半倍频程平滑、±4.5 dB；验证最终float32 FIR实际全频响应并留采样间峰值余量。约束只针对新增校正。 |
| 输出 | WAV显式FLOAT，保存完整卷积尾部；amp-only将基础cab与校正合成为playback IR；预览从落盘IR回读渲染。 |
| 增益 | 完整输入增益=DI一次校准＋搜索增益；唯一输出增益精确保存。新DI沿用该校准，无隐藏重归一/限幅。 |
| 复现 | 模型、基础IR和播放IR带SHA256；导出文件可由render.py重新加载。输出目录必须新建或为空，重放禁止覆盖既有文件。 |
| NAM_LV2 | 保存基于模型元数据及默认12dBu设置的控件补偿建议，未自动施加到裸Python模型；实际插件校验标记固定false。 |

## 真实模型与数值证据

验证资产为 Lum 的 Marshall 2204 amp-only capture，来自 VoLum 固定提交 `a0b2f08b6be3fa2f7fdb58543f579d32e0ebe2f3`。文件、来源、仓库MIT许可、校准限制在 `../assets/test_models/model_provenance.json`。文件名中的索引3不作为物理功放通道声明。

- 原文件 SHA256：`a9579e9a136b9dd0a360ecd64235bedaa58f4468e7c58dc8e734ca2b19fd042f`。
- 格式0.5.4，WaveNet，48k，感受野4093 samples；CPU及CUDA均为真实预训练权重推理，不是MockAmp替身。
- 旧格式适配后，重新导出的全部权重与原文件float32值逐项精确相等；原文件SHA不变。
- 1秒输入48000 samples → 输出48000 samples，有限且非零；CUDA参数实际位于cuda:0。CPU/GPU最大绝对误差 **2.011656761e-6**。记录：`../artifacts/stage1/real-nam-environment.json`。
- 10.25秒包含持续信号的探针，整段处理与默认10秒分块一致；重复调用输出一致。另验证短于感受野的分块。
- 同DI、同模型、已知+6 dB目标：在多档增益搜索中恢复+6 dB，评分接近0。该项验证搜索执行正确，不证明跨演奏准确率。
- 真NAM＋测试用短FIR：验证完整链、合并IR、总增益、FLOAT写盘、尾长、文件回读、CLI match/render。测试短FIR只验证路由，不冒充真实箱体。
- 真实模型用合成探针测试数值行为；尚未用真实吉他DI做音色质量验收。

上面的1秒运行时间为冷启动约CPU0.063秒、CUDA0.217秒；没有做预热/批量检索性能基准，不能据此比较部署性能。

## 回归检查的有效性

`scripts/verify-regression-guards.py` 临时恢复“粗排只取静音前缀”和“卷积截断尾部”两个旧问题，两项对应检查均明确失败；还原修复后均通过。源码已恢复。证据：`../artifacts/stage1/regression-guards.json`。

旧synthetic smoke在修正总增益后暴露了原来的验收问题。因此将其改成同DI、候选内完全相同处理、无额外EQ的工程闭环：有效drive恢复14 dB、导出频谱无明显新增误差，运行通过。没有放宽原跨演奏阈值来宣称成功。

## 必须保留的失败诊断

旧例子使用不同随机种子的合成演奏，参考另有倾斜EQ与presence而候选没有。按完整输入增益计算：目标有效drive14 dB，检索结果27.7767 dB，误差13.7767 dB，未达到旧±5 dB条件。诊断可见 `../artifacts/stage1/unpaired_toy/diagnostic.json`，由 `scripts/diagnose-unpaired.py` 生成。

这反映目前启发式特征受EQ、音高、攻击和演奏影响的局限。尚无证据证明它适合Free Bird；不能将76项工程回归通过写成跨演奏音色匹配已成功。下一阶段应先做已知真值的跨DI小实验，失败时再研究评分/候选覆盖，不直接堆模型或增加EQ幅度。

## 下一阶段需要的素材

1. 8–12个来源和许可清晰、原生48k的真实NAM，涵盖clean/crunch/lead；至少一条full-rig、一条amp-only＋真实箱体IR，记录哈希与内容范围。
2. 三段真实干声A/B/C：固定硬件增益，A/B不同乐句用于双向交叉验证，C留出试听；包括弱强拨弦、延音和主音，避免使用已响度处理的试听版代替原DI。当前目录仍只有三首湿声，未补充真实DI。
3. Free Bird第5秒后人工听选三个不重叠8–12秒片段，检查左右声道，两个开发、一个留出；原WAV保留。尚未选段或音源分离。
4. 实际NAM_LV2版本、编译配置、48k宿主和最终IR加载器。Python加载与回放通过不等于实际插件、Orange Pi实时性能已经通过。

## 复验

从NAM_guitar_modeling目录运行：

```powershell
rtk .\.venv\Scripts\python.exe -X utf8 -m pytest -q --junitxml artifacts/stage1/pytest.xml
rtk .\.venv\Scripts\python.exe -X utf8 assets/test_models/verify_model.py
rtk .\.venv\Scripts\python.exe -X utf8 -m pip check
```

本机CUDA是第一阶段显式验收条件，因此对应回归不会在无GPU时静默skip。模型输入dBu校准元数据为空，结果不代表物理功放电平校准等价。
