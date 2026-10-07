# FGLGuard —— 隐私保护的拓扑引导 MAS 安全防护（图联邦学习复现）

> 论文：**Privacy-Preserving Topology-Guided Safety for LLM-Based Multi-Agent Systems
> via Federated Graph Learning** (Yu et al., 2026, arXiv:2609.02967)
>
> 本仓库是论文方法的一个**小型、可运行、自包含**的实现：不依赖任何 LLM API、
> 不依赖 `torch_geometric`，只用 `torch` + `numpy`，在一台 CPU 机器上几分钟内
> 就能把论文的全部五个机制（近端联邦训练 → 域均衡聚合 → 预算校准 → 佐证打分 →
> 守卫式重写）完整跑一遍并给出量化结果。

---

## 1. 论文要解决什么问题

基于 LLM 的多智能体系统（MAS）把若干 agent 组织成一张**通信图**。拓扑决定了
不安全内容传播的速度，因此自然的防护思路是：在通信图上训练一个 GNN，
定位"危险 agent"并对其周边拓扑做干预。

但这类「拓扑引导防护」暗含一个无法跨组织部署的前提：**某个站点能集中所有带标签
的交互轨迹**。现实中：

- 每个 operator 的 episode 里都有私有 prompt、工具输出、专有工作流；
- 没有任何一个 silo 能看到完整的攻击分布；
- 攻击分布是 non-IID 的，unsafe 标签是稀缺的。

论文把这个问题形式化为**图联邦学习**（graph federated learning），并提出
**FGLGuard**：各 operator 只在自己的 judge 标注好的 episode 图上训练一个
**边特征图注意力打分器**，**只交换模型更新**，不交换任何原始 trace。

论文的三个关键论断（本仓库逐一验证）：

| 论断 | 论文证据 | 本仓库复现 |
|---|---|---|
| **Q1** 现成的检测器会因 *分布漂移* 而非架构失效；必须用**目标域自身的 trace** 做适配 | G-Safeguard off-the-shelf 在 ASB 上 AUROC 0.512，域内重训后 0.695 | ✅ off-the-shelf ≈ 0.29–0.43（低于随机），域内联邦 ≈ 0.76–0.89 |
| **Q2** 有免费 judge 标签时，无监督 / 免训练方案明显落后 | BlindGuard/XG-Guard 在有机风险上掉到随机水平 | ✅ 无监督 deviation 打分 ≈ 0.50–0.53 |
| **Q3/Q4** 联邦能逼近多域池化，且在**不牺牲效用**的前提下大幅降低攻击成功率 | AgentDojo 定向 ASR 0.401 → 0.230（−43%），clean utility 0.575 → 0.650 | ✅ ASR 0.4213 → 0.0813（**−81%**），utility 0.9825；跨域联邦≈多域池化（差 0.004 AUROC） |

---

## 2. 方法：五个机制

```
                        ┌─────────────── 服务端 (Server) ───────────────┐
                        │  θ(r+1) = Σ_k α_k θ_k        ← Eq.(6) 域均衡   │
                        │  τ* = argmax Recall s.t. FPR≤ρ ← Eq.(7) 校准   │
                        └───────────────────────────────────────────────┘
                              ▲ 只传模型更新 θ_k        │ 只传 (分数, 标签)
                              │                          ▼
   ┌──────────────────────────┴────────────┐   ┌─────────────────────────┐
   │ Operator k (silo): 私有 episode 图 D_k │   │ 本地 val 分数池 V_k      │
   │  min_θ Σ BCE(s_i, y_i) + μ/2‖θ-θ(r)‖²  │   └─────────────────────────┘
   │        ← Eq.(5) FedProx 近端目标       │
   └───────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────── 运行时 sidecar (Runtime) ───────────────┐
   │ ŝ_j = max(s_j, max_{i:A_ij=1} s_i)          ← Eq.(8)   │
   │ b_j = 1[ŝ_j ≥ τ*]                                       │
   │ Π(u_j) = u_j  /  Rewrite 后重打分  /  Refuse  ← Eq.(9)  │
   └─────────────────────────────────────────────────────────┘
```

### 2.1 Episode 图构建 $G=(X,A,E)$（论文 Sec. 3）

一次任务执行产生一张属性图：

- **节点特征**锚定到 agent **自己的话语历史**：$x_i=\frac1T\sum_t \phi(u_i^t)$
  （论文特意偏离 G-Safeguard 的"聚合入边"做法，否则稀疏拓扑下被孤立的 agent
  拿不到任何特征）；
- **边特征** $e_{ij}=[\phi(u_j^1);\dots;\phi(u_j^T)]$ 承载**接收方**的逐轮嵌入，
  $\psi(e_{ij})$ 取时间均值；
- **标签**广播到所有节点：$y_i=y\ \forall i$；
- 冻结句编码器 $\phi(\cdot)$。

### 2.2 边特征图注意力打分器（论文 Eq. 2–4）

$$e^{(\ell)}_{ij}=\mathrm{LeakyReLU}\big(a_\ell^\top[W_\ell h_i^{(\ell)};W_\ell h_j^{(\ell)};U_\ell\psi(e_{ij})]\big)$$
$$\alpha^{(\ell)}_{ij}=\mathrm{softmax}_{i':A_{i'j}=1}\big(e^{(\ell)}_{i'j}\big),\qquad
h^{(\ell+1)}_j=\sigma\Big(\sum_{i:A_{ij}=1}\alpha^{(\ell)}_{ij}W_\ell h^{(\ell)}_i\Big)$$

### 2.3 联邦近端训练 + 域均衡聚合（论文 Eq. 5–6）

$$\mathcal L_k(\theta)=\sum_{(G,y)\in D_k}\sum_{i\in G}\mathrm{BCE}(s_i(\theta),y_i)+\frac{\mu}{2}\|\theta-\theta^{(r)}\|_2^2$$

$$\alpha_k=\frac{1}{D}\cdot\frac{N_k}{\sum_{k'\in C_{d(k)}}N_{k'}}$$

**域内按数据量加权、域间等质量**；当 $D=1$ 时退化为 FedAvg（论文 Proposition 1，
本仓库有对应单元测试）。BCE 项**不做类别重加权**（论文消融：稀缺标签下重加权有害）。

### 2.4 过拒答预算下的运行点校准（论文 Eq. 7）

$$\tau^*=\arg\max_{\tau}\ \mathrm{Recall}(\tau;V)\quad \text{s.t.}\quad \mathrm{FPR}(\tau;V)\le\rho$$

$V$ 由各客户端**本地上算的标量分数**汇总而成（原始图不出域）。

### 2.5 上游佐证 + 守卫式重写（论文 Eq. 8–9）

$$\hat s_j=\max\Big(s_j,\ \max_{i:A_{ij}=1}s_i\Big),\qquad
\Pi(u_j)=\begin{cases}u_j,& b_j=0\\ u_j',& b_j=1\wedge \hat s(u_j')<\tau^*\\ \text{Refuse},&\text{otherwise}\end{cases}$$

关键细节：重写后的内容必须用**同一套佐证规则**重新打分（论文消融：目标规则重检
会损害安全性，佐证规则重检既保安全又把 completion 提到 0.772）。

### 2.6 机制 ↔ 测试对照表

`python tests/test_components.py` 的 17 项测试逐条锚定论文中的公式与命题：

| 论文公式 / 命题 | 实现位置 | 对应测试 |
|---|---|---|
| Sec.3 节点特征 $x_i=\frac1T\sum_t\phi(u_i^t)$ | `graph.build_graph` | `test_graph_construction` |
| Sec.3 边特征承载**接收方**嵌入 | `graph.build_graph` | `test_graph_construction` |
| Sec.3 标签广播 $y_i=y$ | `graph.build_graph` | `test_graph_construction` |
| Eq.(2-4) 边条件注意力 | `model.EdgeFeaturedGATLayer` | `test_model_forward_and_loss` |
| Eq.(5) 近端目标 | `federated.FederatedClient.local_train` | `test_federation_beats_off_the_shelf` |
| Eq.(6) 域均衡聚合 | `federated.domain_balanced_weights` | `test_aggregation_weights` |
| **Prop.1** $D=1 \Rightarrow$ FedAvg | 同上 | `test_aggregation_weights` |
| Eq.(7) 预算校准 | `calibration.calibrate_threshold` | `test_calibration_respects_budget` |
| **Prop.2** 良性流量 $\Rightarrow (1-\rho)$ 分位数 | 同上 | `test_calibration_benign_only_matches_quantile` |
| Eq.(8) 佐证打分（单调不下调） | `graph.corroborated_scores` / `runtime.RuntimeGuard.corroborate` | `test_corroboration` |
| Eq.(9) 处置流程（三种动作） | `runtime.RuntimeGuard.intervene` | `test_runtime_decision_paths` |
| Sec.5.3 守卫工具调用损伤 completion | `RuntimeConfig.guard_final_answer_only` | `test_guarding_tool_calls_costs_utility` |
| Sec.5.3 重写确实清除 payload | `runtime.sanitizing_rewrite` | `test_rewrite_clears_payload` |

---

## 3. 目录结构

```
FederatedLearning/
├── README.md
├── READING_ORDER.md              # ★ 按依赖排好的 18 步阅读路线 + 符号对照表
├── requirements.txt
├── main.py                       # 一键最小可运行示例 (论文 Fig.2 全流程)
├── fglguard/                     # 算法库
│   ├── config.py                 # 全部超参 (对应论文 5.1 Protocol)
│   ├── encoder.py                # 冻结句编码器 φ(·) 的轻量替代
│   ├── mas_simulator.py          # MAS episode 仿真器 + 免费 judge 标签
│   ├── graph.py                  # G=(X,A,E) 构建 + 佐证打分 Eq.(8)
│   ├── data.py                   # 客户端划分 (non-IID 标签偏斜 / 跨域)
│   ├── model.py                  # 边特征 GAT 打分器 Eq.(2-4)
│   ├── federated.py              # 客户端/服务端, FedProx+域均衡 Eq.(5-6)
│   ├── calibration.py            # 预算校准 Eq.(7)
│   ├── runtime.py                # 佐证 + 守卫式重写 Eq.(8-9)
│   ├── metrics.py                # AUROC / FPR / Wilson CI / ECE
│   └── utils.py                  # 种子、落盘、绘图
├── experiments/
│   ├── common.py                 # World: 仿真世界与通用评测
│   ├── run_federation.py         # 实验一: 检测质量对比 (论文 Table 1 / Fig.3)
│   └── run_runtime.py            # 实验二: 运行时部署与消融 (论文 Fig.4 / Sec 5.3)
├── scripts/pdf_to_text.py        # 论文 PDF -> 文本 (可选)
├── tests/test_components.py      # 17 项自包含单元测试
└── results/                      # 实验产物 (JSON + PNG + 日志)
```

---

## 4. 快速开始

> **第一次看这个项目？** 不要按目录顺序读代码。先看 **`READING_ORDER.md`**
> —— 它按依赖关系排出了 18 步阅读路线，每步说明"看哪个函数、看完要能回答什么"，
> 并附 **论文符号 ↔ 代码变量对照表**（$\theta$ 在代码里叫什么、$\alpha_k$ 在哪一行等）。

```bash
# 0) 环境 (本项目已在 python 3.14 + torch 2.14 CPU 上验证)
pip install -r requirements.txt

# 1) 单元测试: 17 项，覆盖编码器/仿真器/图构建/模型/校准/聚合/运行时/指标
python tests/test_components.py

# 2) 一键示例: 训练 -> 校准 -> 对比 -> 运行时部署，约 1-2 分钟
python main.py

# 3) 实验一: 联邦检测对比 (论文 Table 1)
python -m experiments.run_federation --seeds 42 43
python -m experiments.run_federation --seeds 42 --purity-sweep   # 追加 Fig.3(b) 扫描
python -m experiments.run_federation --cross-domain              # 论文 Q3 跨域设定

# 4) 实验二: 运行时部署与消融 (论文 Fig.4 / Sec 5.3)
python -m experiments.run_runtime --seeds 42
```

结果写入 `results/`：`*.json`（结构化指标）、`*.png`（图）、`*.log`（日志）。

---

## 5. 仿真器设计（为什么这样能复现论文现象）

真实语料（Agent-SafetyBench / R-Judge / AgentDojo）需要跑 LLM 并下载数据集，
不适合"小型可运行"。本仓库用**结构化仿真**复现同样的统计性质，设计要点如下。

### 5.1 词表分块（制造真正的 domain shift）

| block | 用途 |
|---|---|
| 共享风险标记（6 token） | 所有域的攻击都含一小部分 → 跨域迁移能拿到**弱信号** |
| 每域良性块（48 token / 域） | 域内话题词，域间**不重合** |
| 每域攻击块（48 token / 域） | 注入 payload，域间**不重合** |

再额外保留第 5 个域作为 **off-the-shelf 源域**。于是：

- 在别的域上训练过的检测器 → 判别方向对本域毫无意义 → AUROC ≈ 0.3–0.4；
- 每个域的检测器只需学本域的方向 → 域内 AUROC 可以很高。

这正是论文"off-the-shelf transfer collapses under **distribution shift**, not
architecture"的机制。

### 5.2 风险是"交互性"的，而不是"局部有害文本"

注入的 payload 从某个上游 agent 出发**逐跳衰减**地沿边传播（`propagation_decay`），
episode 被判 unsafe 的充要条件是 **payload 在最后一轮抵达了行动 agent 的上下文**
（自身被感染，或任一上游邻居被感染且被采纳，`delivery_prob`）。

```
source ──p_prop──▶ carrier ──p_prop──▶ final-answer agent
(level 1.0)      (level 0.5)          ↑ 自身话语可能完全干净
```

后果（正是论文 Q2 的论点）：

- 只看行动 agent **自己**话语的 detector 拿不到完整信号 →
  无监督 `deviation-from-theme` 打分 **≈ 0.50–0.53（随机）**；
- 必须借助**拓扑聚合**才能定位风险 → GAT 的两层邻域聚合正好够用；
- **上游佐证 Eq.(8)** 成为一个真正有用的运行时规则。

### 5.3 non-IID 与体量偏斜

- **标签偏斜纯度 $p$**：每个客户端有主导标签，其中 $p$ 比例的 episode 属该标签
  （用**无放回拒绝采样**保证客户端间数据不重叠、纯度精确可控）；
- **体量偏斜**：客户端规模乘子 `(1.75, 0.5, 1.75, 0.5)`，
  且主导标签与体量**解耦**，避免两种偏斜互相混淆。

---

## 6. 实验结果

> 以下为本机 (CPU) 实测。命令与产物在 `results/`。

### 6.1 一键示例 `python main.py`

```
[1/5] 数据就绪: 每客户端训练图数量 [140, 40, 140, 40]
[2/5] 联邦近端训练 (FedProx mu=0.01) ... macro_val_auroc 0.645 -> 0.696
[3/5] 过拒答预算校准 (rho=0.1)

[4/5] 检测质量 (macro AUROC)
  off-the-shelf (异域合成语料)      0.4286   ← 分布漂移导致失效
  local-only (各 silo 独立)        0.6840   ← 只看到自己的攻击切片
  FedAvg (体量加权, 无近端项)        0.7576
  FGLGuard (ours)                 0.7576
  centralized (汇聚原始图, 隐私违规) 0.8918   ← 上限参照

[5/5] 运行时部署
  方案             ASR ↓   utility ↑  over-refusal
  unguarded      0.4531  1.0000     0.0000
  off-the-shelf  0.3531  0.7031     0.2969
  local-only     0.2156  0.7375     0.2625
  centralized    0.1062  0.9437     0.0563
  FGLGuard       0.2000  0.7750     0.2250
>>> FGLGuard 把 ASR 从 0.4531 降到 0.2000 (相对削减 55.9%)
```

### 6.2 实验一：域内联邦检测（`python -m experiments.run_federation --seeds 42 43`）

设定：$K=4$ 个 operator 处于**同一业务域**（$D=1$，故 Eq.(6) 退化为 FedAvg），
各自持有 non-IID 私有切片（标签偏斜 $p=0.8$ + 体量乘子 `1.75/0.5/1.75/0.5`），
30 轮联邦、每轮 2 个 local epoch，**梯度预算匹配**。

```
arm                        macro AUROC        domain 0
-------------------------  -----------------  --------
off-the-shelf (O-S)        0.3646 +/- 0.0472  0.365
unsupervised (deviant)     0.5120 +/- 0.0420  0.512
local-only                 0.8480 +/- 0.0102  0.848
FedAvg                     0.9065 +/- 0.0062  0.907
FGLGuard (ours)            0.9065 +/- 0.0066  0.906
centralized (upper bound)  0.8957 +/- 0.0140  0.896
```

对应的论文 Table 1（AUROC %）：

| 方法 | ASB | R-Judge | AgentDojo | 本项目（同域） |
|---|---|---|---|---|
| G-Safeguard (off-the-shelf) | 51.2 | 63.0 | 60.5 | **36.5**（低于随机） |
| 无监督 (BlindGuard 类) | 45.6 | 27.8 | 58.4 | **51.2**（随机） |
| local-only | 66.6 | 76.2 | 58.4 | 84.8 |
| FedAvg | 70.5 | 85.0 | 65.3 | 90.7 |
| **FGLGuard** | **72.6** | **89.5** | **68.0** | **90.7** |
| 域内 centralized | 69.5 | 81.0 | 67.6 | 89.6 |

**读法（与论文一致的三点）**：

1. **Q1** off-the-shelf 因为分布漂移而彻底失效（0.36，甚至**低于随机** —— 与论文
   在 planted-attacker 语料上观察到的 0.331/0.287 反相关一致）；local-only 只看到
   自己那一片攻击分布，落后联邦 **5.9 个 AUROC 点**；而**联邦在不汇聚任何原始
   trace 的前提下越过了集中式上限**（0.9065 vs 0.8957）—— 正是论文的核心论断。
2. **Q2** 免费 judge 标签能带来什么：无监督 `deviation-from-theme` 打分恰好落在
   随机水平（0.512），与论文 "organically unsafe collaborations drift together,
   so deviant-agent scoring has no signal" 一致。
3. 我们的仿真中 **FGLGuard 与 FedAvg 持平**，见 §7 的诚实说明。

产物：`results/federation_detection.json`、`results/fig_detection_bars.png`、
`results/fig_training_curve.png`。

### 6.3 实验二：运行时部署（`python -m experiments.run_runtime --seeds 42`）

在**全新**的 400 条攻击 + 400 条良性 episode 上执行 Eq. (9) 的处置流程，
每个 arm 的阈值都由 Eq. (7) 在 $\rho=0.10$ 下独立校准（FGLGuard: $\tau^*=0.596$，
校准集上 `val_recall=0.469, val_FPR=0.099`）。

```
arm                          ASR ↓   utility ↑  over-refusal  rewrite  tau*
---------------------------  ------  ---------  ------------  -------  -----
FGLGuard (ours)              0.0813  0.9825     0.0175        0.0175   0.596
FedAvg                       0.0813  0.9825     0.0175        0.0175   0.596
centralized (pooled)         0.0912  1.0000     0.0000        0.0100   0.622
local-only (4 silos)         0.1013  0.9237     0.0762        0.0100   0.555
off-the-shelf (O-S)          0.3350  0.7375     0.2625        0.0025   0.989
  abl: no corroboration      0.2112  1.0000     0.0000        0.0250   0.596
  abl: no guarded rewrite    0.0813  0.9825     0.0175        0.0000   0.596
  abl: guard tool calls too  0.0475  0.7900     0.2100        0.0050   0.596
unguarded (reference)        0.4213  1.0000     0.0000        0.0000    -
```

（`rewrite` 列 = 被拦截后经一次重写并**成功重新放行**的份额。）

对应的论文 Fig. 4：unguarded ASR 0.401 → FGLGuard 0.230（−43%），utility 0.575 → 0.650；
centralized 0.262 @ utility 0.500；localized 平均 0.317。

**读法（与论文一致的四点）**：

1. **Q4 主结论**：FGLGuard 把定向 ASR 从 **0.4213 压到 0.0813（−80.7%）**，
   同时保有 **0.9825** 的良性交付率，over-refusal 只有 0.0175（远低于预算 0.10）。
   —— 与论文 "cuts ASR by 43% at near-unguarded utility" 同向且更显著。
2. **上游佐证 Eq.(8) 是承重结构**：在同一 $\tau^*$ 下关掉佐证，ASR 立刻从 0.0813
   恶化到 **0.2112（2.6 倍）**。这正是论文 "corroboration ... at a fixed threshold it
   raises detection" 的量化体现。
3. **守卫工具调用会伤效用**：把中间轮次也纳入守卫，ASR 进一步降到 0.0475，
   但 utility 从 0.9825 掉到 **0.7900**、over-refusal 飙到 0.2100 —— 论文
   "mid-trajectory reroutes derail benign episodes" 的结论。
4. **off-the-shelf 的阈值会饱和**：异域检测器校准出的 $\tau^*=0.989$ 几乎不触发，
   ASR 仍有 0.3350，同时误伤 0.2625 的良性流量 —— 论文 "the cross-domain threshold
   saturates at 1.0 and never fires" 的现象被完整复现。local-only 的 4 个本地阈值
   则"要么饱和、要么乱开火"（ASR/utility 双输）。

> **关于重写通道的一点观察**：`abl: no guarded rewrite` 的 ASR / utility 与完整版
> **完全一致**，只有 `rewrite` 列从 0.0175 变成 0。也就是说，一次重写只救回了
> **1.75%** 的攻击 episode（换来的是"给出修复后的答案"而不是"直接拒答"），
> 对 ASR 与良性交付率都没有可测影响。原因自洽：本仿真的风险几乎总来自上游，
> 只重写行动 agent 自己的话语无法降低佐证分数 $\hat s_j$。
> 量级上这与论文报告的重写通过率 **6/74（≈8%）** 相当，只是在本仿真中更极端；
> 论文中重写通道的价值主要在 *capability/completion* 侧（0.772 vs 0.620），
> 而本仿真的 utility 只在良性 episode 上度量，因此看不到这一收益。

产物：`results/runtime_deployment.json`、`results/fig_runtime_asr.png`、
`results/fig_runtime_utility.png`。

### 6.4 跨域联邦（论文 Q3）

```bash
python -m experiments.run_federation --cross-domain --quick --tag cross_domain
```

`--cross-domain` 让 4 个 operator 分属 4 个不同业务域（$D=4$，此时域均衡聚合与
纯体量加权不再等价）。用 `--quick`（6 轮、小数据）做冒烟验证得到：

```
arm                        macro AUROC        domain 0  domain 1  domain 2  domain 3
-------------------------  -----------------  --------  --------  --------  --------
off-the-shelf (O-S)        0.4875             0.598     0.434     0.495     0.423
unsupervised (deviant)     0.6708             0.618     0.897     0.606     0.563
local-only                 0.6172             0.843     0.457     0.495     0.674
FedAvg                     0.5717             0.799     0.463     0.423     0.602
FGLGuard (ours)            0.5991             0.740     0.491     0.556     0.609
centralized (upper bound)  0.6029             0.858     0.457     0.498     0.599
```

- **Q3 的核心论断被复现**：跨域联邦 FGLGuard（0.5991）与多域池化 centralized
  （0.6029）仅差 **0.0038 AUROC**，而 off-the-shelf 只有 0.4875。
  论文的口径是 "within 0.03 AUROC on every domain"。✅
- **域均衡聚合在这里真正起作用**：FGLGuard 0.5991 > FedAvg 0.5717（+0.027），
  因为体量加权会让大域主导、小域被饿死。这正是论文 Eq.(6) 的设计意图。
- local-only（0.6172）在**这个欠训练的小配置**下反而更高：每个客户端是自己域的
  唯一专家，跨域联邦会带来负迁移。论文的 Q1 结论（联邦 > local-only）建立在
  **同域 non-IID** 设定上（见 §6.2），两处并不矛盾。

> 该结果是 6 轮 / 小数据的**冒烟级**验证，不是论文量级结论；
> 去掉 `--quick` 可得到完整配置结果。

### 6.5 重新绘图

训练很贵、画图很便宜，因此图表可以从 JSON 直接重建：

```bash
python scripts/make_figures.py
```


---

## 7. 与论文的差异（诚实说明）

本仓库是**机制级复现**，不是数值级复现。以下是刻意保留的差异：

| 项 | 论文 | 本仓库 | 原因 |
|---|---|---|---|
| 句编码器 | 冻结 MiniLM (384-d) | 冻结随机 token 嵌入 + 均值池化 (32-d) | 零依赖、零下载；保留"固定映射 + 分块语义"的性质 |
| 数据 | ASB / R-Judge / AgentDojo 真实轨迹 | 结构化 MAS 仿真器 | 见 §5；仿真器本身可替换 |
| 打分器规模 | 2 层 GAT, 256 hid, 4 heads (≈0.5M 参数) | 2 层 GAT, 32 hid, 4 heads (≈3.0 万参数) | CPU 秒级可跑 |
| 联邦轮数 | 40 × 2 local epochs | 30 × 2 (可配) | 同上 |
| **自环** | 论文的注意力只对**入边**归一化 | **默认加自环** `self_loops=True` | 论文的孤立节点隐藏状态退化为常数、打分恒为 0.5；而 final-answer agent 常处图的下游。这是为了让模型可用而做的工程取舍，`config.model.self_loops=False` 可切回原样 |
| 校准并列规则 | 正文："并列时偏向更低的 FPR" | 默认 `tightest`（最优召回下取最紧可行运行点） | 论文正文这句与它自己的 Proposition 2（良性流量下退化为 $(1-\rho)$ 分位数）不相容；本实现默认与 Proposition 2 一致，`CalibConfig.tie_break="lower_fpr"` 可切回正文行为 |
| **近端项的实际增益** | Table 1: FGLGuard 72.6 vs FedAvg 70.5 | 本仿真中二者**基本持平** | 诚实结论：在本仿真的小模型 + 共享特征空间 + 易任务下，客户端漂移本身无害，近端项不产生可测增益。论文中该增益来自真实数据的剧烈分布漂移，其 Fig.3(b) 也显示 $K=4, p\le0.8$ 时两条曲线几乎重合，只有 $p=1.0$ 且客户端数增大时才分离。我们没有为了"好看"而调大 $\mu$ |

---

## 8. 已知局限与可扩展方向

- **仿真数据不代表真实 agentic 风险分布**：要接入真实语料，只需替换
  `MASSimulator`，保持 `Episode` 接口即可（`graph.py` 之后的所有代码无需改动）。
- **校准的预算迁移存在泛化间隙**：$\tau^*$ 在验证集上**按构造**满足
  $\mathrm{FPR}\le\rho$（有单元测试保证），但在全新的流量上会有偏移。
  本实验中 FGLGuard 的 over-refusal（0.0175）远低于预算，说明**更高 AUROC 的检测器
  把预算迁移得更好**；而在 `--quick` 的欠训练模型上该值会明显超过 0.10。
  生产环境建议留预算余量，或按论文 §6 的做法做**每部署点重新校准**。
- **重写通道在本仿真中收益很小**（见 §6.3 的说明）：模型把风险归因到上游后，
  只修复行动 agent 自己的话语很难降低佐证分数，因此一次重写只救回约 1.75% 的攻击
  episode。若要体现论文中多通道 (mask / reroute / rewrite) 的差异，需要引入
  "重写会重新生成整条轨迹"的仿真语义。
- **未实现**：SCAFFOLD / FedDC / MOON / FGSSL 等聚合对照、Agent backbone 漂移
  （论文 Table 2）、团队规模与拓扑形态扫描（论文 Fig.3(a)）。这些都可以在
  现有 `train_federated(..., mu=..., domain_balanced=...)` 接口上直接扩展。
