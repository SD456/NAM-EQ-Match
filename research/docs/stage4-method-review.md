# 第四阶段方法审阅：先辨认内容差，再决定是否 EQ

这是只读代码审阅后的方法建议，不是实验结果，也不改变第三阶段结论。保留生产评分、旧结果和多 DI 平滑谱基线；不开展 Free Bird。

## 已确认的问题

`stage3_benchmark.eq_options()` 将每个探针通过 `design_match_ir()` 得到的实际、已限幅 FIR 响应送入 `make_consensus_ir()`。因此同号和极差门槛描述的是处理后的曲线，并不能证明原始谱差一致。更根本的是三个探针面对同一参考，共享该参考的音符/演奏内容偏差。第三阶段真链真增益 E0 的 16/16 反例已经说明，只要求探针一致不足以判断真实后置 EQ。

第四阶段应让探针与参考各自承担不同角色：探针降低候选表征的方差；至少两段参考检验偏差是否跨内容复现；独立 calibration 内容量化无 EQ 时也会出现多大的差异。三者不能互相替代。

## 最小可实现规则

固定 80–8000 Hz、半倍频程平滑和现有频率网格。设 `S(x)` 为未限幅的平滑 log-power 谱，`center(v)=v-median(v[80:8000])`。对于同一候选 rig、同一 gain：

1. 每个参考 r、探针 p 的原始差为 `D[r,p]=center(S(reference[r])-S(candidate_probe[p]))`。
2. 参考的汇总差为 `C[r]=center(median_p D[r,p])`。先在每个固定 rig/gain 聚合，不能给每段 probe 单独选 gain。
3. 独立 calibration DI 通过同一个 rig/gain、关闭后置 EQ 后，计算完全相同的 `N[i]`。probe 组与 calibration reference 不得重叠，calibration reference 与 test reference/holdout 不得重叠。单个 reference 只算一个内容单位，不能将 P×R×rig×gain 当成独立样本数。
4. 最简全曲线 null envelope：`q=max_i max_f(abs(N[i,f]))`，频带内取最大，作为预先固定的经验最坏上界；`threshold=max(1 dB,q)`。也可预先约定 higher 95% 分位，但有限样本不宜给它形式覆盖保证。所有阈值在新 test 前锁定，不能根据 test teacher/E0/E1 或 H 改动。
5. 每个频点只有在所有新参考同号、`min_r abs(C[r,f]) > threshold`，且原始参考间极差 `ptp_r C[r,f] <= 3 dB` 时支持修正。至少两段参考，比较使用严格大于。原始探针差和极差另存诊断，不再用已 clip 的 FIR 响应做一致性证据。
6. 建议修正为支持频点上的 `median_r C[r]`，其他频点置零。沿用现有半倍频程最终平滑、端点 taper、2048 taps 和实际 float32 FIR 全 Nyquist ±4.5 dB 检查。无支持频点必须直接返回 `[1.0]`，避免数值近似的假修正。平滑后会向附近频点扩散，应同时保存 raw support 和最终实际响应；不能声称每个最终非零频点均已单独通过检验。

上述全曲线标量上界最小、可解释、对多频率检验保守。若改用逐频 envelope，必须明确它只是逐频经验阈值；许多频点被检验后，不能将逐频 95% 称为整条曲线 95%。若将来要形式保证，可在独立 fit 内容上估计逐频尺度，再在独立 calibration 内容上对整曲线最大标准化偏差做校准；这需要额外素材，本轮不必实现。

## 校准库选择与小样本边界

实际候选库允许把任意独立 DI 送进已知 NAM/IR，因此最直接的是为全部 12 个已知 rig、固定 gain 网格建立各自 null，包含四个目标 rig 也不构成 test reference 泄漏。不能使用 test reference、holdout、test EQ 标签或它们的渲染。这样的实验是已知候选库内的新内容迁移，不能声称泛化到未见模型。

如协议已经选择仅用非 teacher rig 建立 pooled null，也可作为更保守的主要试验；必须在运行前冻结池化方式和 gain 覆盖。库中有同 NAM 配两 IR，也有同一放大器 family 的不同 capture；排除四个 teacher rig 不等于排除了相同 NAM，更不等于 family 留出。逐 family 的结果必须使用真实 family 身份分组。

不论按全部模型还是非 teacher 模型校准，独立信息主要来自新录音/乐句。重复渲染增加了条件覆盖，不会增加乐手、录音或内容的独立样本量。多个同录音八秒切片也可能相关。若独立 calibration 块数 m 小于 19，5% split-conformal 阈值的序号 `ceil(.95*(m+1))` 可能超过可用序号，此时严格阈值为无穷大；经验最大值仍可研究使用，但不能宣称 95% 控制误报。即使 m 足够，还需要 exchangeability，当前单录音素材不支持强外推。

## 不允许靠全部 bypass 宣布成功

预先分别报告 E0 误启动数、E1 启动数、真链真增益条件下对已知 +3 dB/1200 Hz shelf 的恢复，以及 H 上相对 bypass 的误差增量。`EQ enabled` 与 `caused harm` 是不同终点，不能只报告宽松质量过筛率。

在 calibration 内另保留不用于 null 拟合的 E1 功效内容，使用同一个固定 +3 dB shelf 和原始尺度。冻结前测出 +3 dB 信号的通过比例和响应幅度，并公开低功效结果；不能在 test 上换分位数、缩小频带或提高 shelf 幅度以求通过。若全曲线 q 大于大部分 centered shelf 幅度，这套检测器对本任务没有足够功效，是研究失败或材料不足，不能因 E0 安静而升级生产。

如果素材不足以留独立 calibration-power 内容，可将对 calibration 内容添加 shelf 的结果仅作为灵敏度诊断，不作为独立功效估计；最终功效只在冻结后的 test E1 上报告。+3 dB 后置 shelf 被中心化后低频/高频分别移动，其可检测幅度并非处处 3 dB。

## level 去偏与 gain 的可辨识性

保留一次固定的 DI 输入校准标量。不要为了比较内容而逐段归一化 NAM 输入，这会改变有效 drive。输出端统一用 `center` 去掉 log 谱的常量偏移，避免输出音量伪装成 EQ；同样去掉纯常量后置增益是该表征的定义，不能宣称检测这类 EQ。H 质量比较沿用成对 RMS 对齐，同时保存未对齐 RMS 以区分音量与谱形。

设 `F` 包括固定 NAM/IR，输入校准幅度为 a，输入档位为 g。对于任意 delta：

`F(a * 10**(g/20) * x) = F((a*10**(delta/20)) * 10**((g-delta)/20) * x)`。

这个等式对非线性 NAM 也成立，因为送入 NAM 的样本完全相同。因此数字前置 level 与数字 input gain 无法从输出分开辨认；可辨认的至多是相对于已知 DI 校准的有效 drive。真实拨弦力度还改变起音、频谱及动态，不能用同一 DI 乘常数代表全部力度变化，也不能把所有 nominal gain 误差都归因于力度。

最小可执行反事实检查：对每段固定 DI 和每个代表性 rig，比较 `(DI*db(delta), g-delta)` 与 `(DI,g)`，delta 取 -6/0/+6 dB；关闭 EQ、保持相同状态初始化/前置 padding。输入数组先断言 allclose，输出在固定 float32 容差内 allclose，居中谱和所有评分也应一致。失败首先查状态、dtype、padding 或实现错误，而不是宣称模型辨认了二者。另做只改变输出 level 的检查，居中谱距离应不变；只改变输入 level 而不补偿 gain 的检查则允许随 NAM 非线性变化。真实强弱弹奏需要新录制、已知输入标定的配对材料，当前无需伪造此结论。

## 实施与验证边界

先跑真 rig 真 gain 的 noEQ/E1 诊断，隔离 EQ 判定本身；再跑选中 rig/gain，单独报告检索错误造成的残差。错误候选之间的稳定差也可能通过 EQ 门槛，因此后者不能继承前者的 noEQ 保证。

最小实现只需一个研究 helper、stage4 runner 接线和小型单元检查；不修改旧 `design_match_ir()`、`make_consensus_ir()` 或生产调用。检查 raw 超大同号差在 clipping 前仍保留 spread、两参考反号/不足两参考严格 bypass、等于 null 阈值不触发、已知一致超阈值曲线会启用、无支持返回精确 identity、输出常量音量不影响结果、最终实际 FIR 响应满足 cap。以上是设计审阅，未运行新音频渲染或硬件测试。
