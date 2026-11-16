# Stage 5：公开力度标注 DI 数据检索 A

检索日期：2026-11-15。依据作者资料、论文和数据发布页；未下载大数据集。结论是**本轮未证实任何候选满足全部条件**，不是证明公开数据不存在。各项 unknown 意味着证据不足，不表示一定没有。五项依次为：真实电吉他拾音器 DI；同一音符或乐句多力度；独立可信的弱/中/强标签；吉他音量及 ADC 增益固定且不归一化/压缩；授权和获取方式明确。MIDI velocity 或 RMS 不能自动视为实际拨弦力度标签。

| 候选 | DI | 同音/句多力度 | 力度标签 | 固定电平、无处理 | 授权/获取 | 判定 |
|---|---|---|---|---|---|---|
| IDMT-SMT-Audio-Effects | 电吉他、无效果音已证实；原始接线 DI unknown | hard/soft 类别存在；同音对应和三档 unknown | hard、soft 手指拨弦；无 medium 证明，picked 是手法而非中力度 | unknown | 评价用途 CC BY-NC-ND 4.0；公开 6.5 GB | proxy，最接近 |
| IDMT-SMT-GUITAR（含单音子集） | 混合直连和一例麦克风；须逐文件鉴别 | unknown | 无三档说明 | unknown | 评价用途 CC BY-NC-ND 4.0；公开 | unknown |
| Guitar-TECHS | 有电吉他 DI，与麦克风和音箱轨分开 | unknown | MIDI 每弦音符标注；没有独立物理力度校准的证明 | unknown | Zenodo 文件公开；具体 license 在所阅页面未明确显示 | unknown |
| GOAT | 真实电吉他 DI，另有 amp 渲染与合成轨 | unknown | 六线谱、对齐 MIDI；无物理力度标注证明 | unknown | 作者仓库要求在 Zenodo 申请访问；数据许可依发布记录核实 | unknown |
| EGDB 原版 | 作者展示 DI 轨及多音箱音色 | unknown | 乐谱/转录；三档力度 unknown | unknown | 作者页提供 Drive 链接；数据许可 unknown | unknown |
| EGFxSet | 真实电吉他 instrument input clean 轨 | 多力度 unknown | 参数含弦、品、拾音器等；无三档说明 | **不通过：clean 与效果轨都归一化** | 作者页称开放，Zenodo 下载；具体授权条款复核 | reject |

## 原始证据和边界

- [Fraunhofer Audio-Effects 作者页](https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html) 指出 “finger plucked - hard, finger plucked - soft, picked”，并列出 “no effect (unprocessed notes/sounds)”；[发布记录](https://zenodo.org/records/7544032) 说明音符、XML/文件名编码和下载。两种实际手指力度有价值，但不能把 picked 伪装成 medium；未逐文件验证同音配对。无效果并不独自证明接线、统一增益或处理历史。
- [Fraunhofer Guitar 作者页](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html)：接口 “directly connected to the guitar output or in one case to a condenser microphone”；单音/多音、手法与 XML 不是力度标签。[Zenodo](https://zenodo.org/records/7544110) 为实际发布。单音是此数据集的子集描述，未找到可证实为独立数据集的 IDMT-SMT-MONOPHONIC 权威记录，不将其重复计数。
- [Guitar-TECHS Zenodo](https://zenodo.org/records/14963133) 列出 “Direct input (DI)” 和 Fishman 每弦 MIDI；[论文](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/103641/Roman%20Guitar-TECHS%3A%20An%20Electric%20Guitar%20Dataset%20Covering%20Techniques%2C%20Musical%20Excerpts%2C%20Chords%20and%20Scales%20Using%20a%20Diverse%20Array%20of%20Hardware%202025%20Accepted.pdf?isAllowed=y&sequence=2) 描述设备、单音和 MIDI，未找到 velocity 或 normalization 说明。MIDI 存在并不证明数值对物理力度校准，亦不证明是三档采集。发布页提醒音轨可有 ≤100 ms 错位。
- [GOAT 作者仓库](https://github.com/JackJamesLoth/GOAT-Dataset) 称 “request access on the Zenodo page”；[论文](https://zenodo.org/record/17706552/files/000076.pdf) 区分 DI、音箱渲染、RSE 虚拟乐器，以及 MIDI 和六线谱。论文著作许可不应混同数据许可；无需将 sampler 音轨当作真实 DI。
- [EGDB 作者演示与数据链接](https://ss12f32v.github.io/Guitar-Transcription/) 将 DI 与不同音箱轨列开；[作者论文](https://arxiv.org/abs/2202.09907) 为 240 首谱的转录研究。EGDB-PG 发布的是音箱渲染数据，不等于原始 DI 证明：[作者发布页](https://zenodo.org/records/19542613)。
- [EGFxSet 作者页](https://egfxset.github.io/) 写明 “Both, the previously recorded clean guitar sounds and the processed recordings were normalized.” 因此不满足固定电平且无归一化的要求，即使是 clean 轨。

## 建议补录规范

同一把真实电吉他、拾音器档位、琴体音量/音色、弦、拨片、拨弦位置及接口高阻输入和 ADC 增益保持固定；直接录单声道干 DI，关闭 amp、插件、压缩、自动增益、归一化与后期电平变更。先按最强力度设置增益并留余量，确认无削波；设置完成后连弱档一起录，不逐条调整。每个选定的弦/品音和短乐句分别录弱、中、强多次，顺序交错，记录演奏者事先指定的类别，而非事后按响度分箱。记录每条文件的音符/乐句、力度、重复序号、乐器/拾音器、旋钮及接口增益、采样率位深、处理链、录音日期和授权；保存原始 WAV、校验和、录音日志。若需要绝对物理力值，额外使用独立校准传感器；主观弱中强仅是演奏者标签。检查音高、标注、削波和静音，保留原电平。
