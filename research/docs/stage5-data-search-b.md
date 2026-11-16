# 力度标签 DI 公开数据检索 B

检索日期：2026-11-15。标准：真实电吉他拾音器干声；同音符或乐句真实弱、中、强演奏并有可信标签；采集增益、吉他控制固定；没有逐条归一化或压缩；原数据许可清楚。以下“未知”不是反证；目前没有找到全部条件被证实的数据，不可用音量大小倒推标签。没有下载大数据或联系作者。

| 数据集 | 拾音器 DI | 同素材多力度及标签 | 固定增益、控制与无逐条处理 | 原数据许可 | 结论 |
|---|---|---|---|---|---|
| [Guitar-TECHS](https://guitar-techs.github.io/) | 支持，分出 DI，三名真人演奏 | 未知：MIDI 与技巧、音高标注不等于弱中强实演标签 | 未知；不同演奏者使用不同器材；不可跨设备假定固定 | [论文](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/103641/Roman%20Guitar-TECHS%3A%20An%20Electric%20Guitar%20Dataset%20Covering%20Techniques%2C%20Musical%20Excerpts%2C%20Chords%20and%20Scales%20Using%20a%20Diverse%20Array%20of%20Hardware%202025%20Accepted.pdf?isAllowed=y&sequence=2)明示 CC BY 4.0 | 未通过，需原始录音细节与标注证据 |
| [EG-IPT](https://zenodo.org/records/15205644) | 支持：专业演奏，DI box 及前级 | 未知：19 种演奏技巧不证明同音符弱中强 | 未知：三个拾音配置不能假定相同 | Zenodo 网页文本显示 Rights/License 但未显示具体授权，未知 | 未通过；不可将技巧分类解释为力度标签 |
| [GADA](https://gingerpianist.github.io/GADA_demo_page/) | 支持，电吉他和贝斯 DI | 同设置十次，但无可信力度标签证据 | 音量与 tone 旋钮最大；固定 ADC gain 和原始未处理状态未知 | 原音频具体许可未知 | 未通过；重复不等于力度分层 |
| [EGFxSet](https://egfxset.github.io/) | 支持：真实 Stratocaster 接声卡乐器输入 | 未见力度层标签 | 明确写明干声和效果录音均归一化 | 网站称开放使用，具体原数据许可此处未确认 | 排除：归一化不满足 |
| [GuitarSet](https://zenodo.org/records/3371780) | 拾音器信号有，但源是原声吉他而非电吉他 | 风格等标签并非弱中强配对 | 未知 | 本次 Zenodo 可读文本未显示具体授权 | 排除：乐器不符；不要把 hex 信号称为电吉他 DI |
| [NSynth](https://magenta.tensorflow.org/datasets/nsynth) | 电吉他原始 DI 与录制路径未知 | 同音高 MIDI velocity 25–127 由商业采样器生成，不能当作真人逐档力度 | 原始采样录制及后处理未知 | Google 明示 CC BY 4.0 | 排除：velocity 不是可信实演力度标签 |
| [TinySOL](https://zenodo.org/records/3659365) / [OrchideaSOL](https://forum.ircam.fr/projects/detail/orchideasol/) | TinySOL 列出的 14 种乐器不含电吉他；OrchideaSOL 采用近距离麦克风 | TinySOL 有力度标签但乐器不符；OrchideaSOL 相同素材情况未知 | IRCAM 描述依据报告或估计恢复动态，非可证明原始固定增益 | TinySOL CC BY 4.0；该记录称其为唯一 CC 子集，OrchideaSOL 许可未核实 | 排除作为电吉他 DI |
| [IDMT-SMT-Bass](https://zenodo.org/records/7188892) | 电贝斯音符；具体干声链路本次未知 | 标签是拨弦和表现技法，不证明弱中强 | 未知 | 本次未确认 | 排除：电贝斯而非目标电吉他，且力度与增益未知 |

EGDB（[原论文](https://arxiv.org/abs/2202.09907)）是 240 个谱例和不同音色的转录数据；音色版本不是不同真实力度。EG-IPT 作者 Fiorini 等与 EGDB 的陈等并非同一个数据集，避免按相近名称合并。关于 GuitarLicks / expressive electric guitar，本次检索未定位足够明确的一手发布和使用许可，不据名称推断其存在或可用。

如果已有 Windows 声卡、吉他与录音软件，30–60 分钟补录可行：使用乐器输入录干声，一次按最强拨弦预留余量并固定输入增益和吉他旋钮、拾音器；保持同一拨片、触弦位置与奏法。选少量相同单音和一条简短原创乐句，每项按实际演奏指令录弱/中/强各数次，并记录每次标签、弦品、设备及设置；导出未归一化、未压缩的 PCM WAV，保留原始连续录音和元数据。强弱是演奏者指令而非后期响度推断；检查削波与底噪，如需调整增益应在正式录制前完成。此为可行性素材，非已执行的录制结果。
