# 固定增益、真实三档力度 DI：检索结论

2026-11-15 两路检索均未核实同时具备**真实电吉他 DI、同音或同乐句三档实演、可对应的力度标签、固定采集增益且无逐条电平处理、明确数据许可**的公开数据。没有把“没有找到证据”解释为“公开数据不存在”，也没有用 RMS 大小补造力度标签。本轮没有下载新的大音频包或联系作者。

| 最相关候选 | 公开资料能支持什么 | 本轮不能用于什么 |
| --- | --- | --- |
| [IDMT-SMT-Audio-Effects](https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html) | 电吉他无效果音及 hard/soft 手指拨弦类别；发布页明确 CC BY-NC-ND 4.0、评价用途 | picked 是另一种奏法，不能补成 medium；同音对应、固定增益和完整原始 DI 路径未核实 |
| [Guitar-TECHS](https://guitar-techs.github.io/) | 真实电吉他 DI、音高/技巧资料；本地已有可追溯原件和切片 | MIDI 不等于三档实演力度；同型号接口也不能证明采集增益固定 |
| [EGFxSet](https://egfxset.github.io/) | 真实电吉他、弦品和效果设置；作者明确干声和效果音均归一化 | 不满足本轮保持原始力度电平的要求 |
| [EG-IPT](https://zenodo.org/records/15205644)、[GADA](https://gingerpianist.github.io/GADA_demo_page/)、EGDB、GOAT | 可用于进一步查证演奏与音色；各自有 DI 或技巧资料 | 本轮未核实完整三档力度、固定增益及相应数据授权链，未准入真实力度实验 |
| NSynth、GuitarSet、TinySOL 等 | 可提供采样器 velocity、原声吉他或其他乐器资料 | 不能代替本课题的真人电吉他固定增益 DI |

一手链接、逐候选证据和未知项保存在[检索 A](stage5-data-search-a.md)、[检索 B](stage5-data-search-b.md)，对应机器记录为 `assets/stage5/discovery-a.json` 与 `discovery-b.json`。检索中论文的许可不能自动转写为原始音频许可；当前重用的 Guitar-TECHS 文件仍绑定既有 `assets/stage4/di-provenance.json` 中的来源记录。本轮没有为未知许可的新素材开下载或使用口子。

因此本轮分开交付：

1. [固定增益补录单](stage5-recording-spec.md)，附 `assets/stage5/recording/` 中会话、事件空表及 36 张预先指定的提示卡。按卡片指定的 weak/medium/strong 记录，不能按事后响度改标签。
2. [条件化评分协议](stage5-protocol.md)与冻结配置 `experiments/stage5.json`，先对已有无力度标签素材作开发回放。它只检验评分流程及已有演奏上的变化，不能证明分离了真实拨弦力度与增益，也不能算新录音确认。

后续若得到新的公开原件，先按上述五项登记证据和许可，再锁定按 take 划分的角色。若按补录单提供自录原件，则真实力度实验转为可执行；当前不以数字缩放或合成音替代。
