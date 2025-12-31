# Roadmap（从 0 到可交付的 MATLAB 主动学习可靠性代理模型）

> 目标：实现 `design.md` 中描述的“pool-based active learning + 主模型（加权 RBF‑SVR / 无工具箱时 RBF‑KRR）预测 + bootstrap 仅用于不确定度 σ(x) + 失效概率 Pf 估计（A-mean）+（可选）在线纠偏”完整工程，支持 2D 解析算例与后续对接 FEM/黑盒仿真；总真实调用预算默认 100 次，可配置。

## 0. 约束与假设（先定边界，避免返工）

- **语言/运行环境**：MATLAB（已确认本机可 `matlab -batch` 调用）。
- **黑盒接口**：极限状态函数 `g(x)` 视为可调用的 evaluator（解析/仿真均可），返回标量 LSF。
- **输入随机变量**：默认独立；每维提供分布类型+参数+物理截断区间（见 `design.md`）。
- **pool-based**：后续所有选点均从固定候选池 `pool` 中挑选（便于可视化、收敛监控、可复现）。
- **预算**：真实 evaluator 调用（FEM）计入预算；pool 生成与 surrogate 训练不计入预算。
- **工具箱依赖（可选）**：
  - 推荐：Statistics and Machine Learning Toolbox（`fitrsvm`/`kmeans`）。
  - 可选：bayesopt（若无则用网格/随机搜索替代，保证可运行）。

## 1. 交付物定义（什么算“做完”）

### 1.1 必须交付（MVP）

- 可运行的 MATLAB 工程骨架：离线主动学习完整闭环（从 0 生成 pool → DOE → 迭代选点 → 停止 → 输出）。
- 2D 解析算例（仅离线）：
  - 输出 Pf 收敛曲线（随 query 次数）。
  - 输出采样点与真实边界 `g(x)=0` 可视化。
  - 产出 `model_offline.mat` 与 `log.mat`（结构与字段见下文）。
- 命令行一键运行：`matlab -batch "run('examples/run_2d_offline.m')"`

### 1.2 扩展交付（v1）

- 在线纠偏模式（离线 + 在线两阶段）：能接收 RBDO 给出的设计点序列，并按触发条件进入局部加密。
- 黑盒/FEM 适配层：提供统一接口、缓存、失败重试、日志落盘与可中断恢复。
- 可复现实验：统一随机种子、配置文件化、结果版本信息记录。

## 2. 项目结构（建议落地的目录与命名）

建议从第一天就按“可测试 + 可复用”组织（后续接 FEM 才不会推倒重来）：

```
src/
  +lqy/
    config/
    dist/
    norm/
    pool/
    doe/
    svr/
    al/
    reliability/
    io/
    viz/
examples/
tests/
design.md
roadmap.md
```

- `src/+lqy/...`：核心库（函数尽量纯函数、输入输出明确）。
- `examples/`：可直接 `matlab -batch` 运行的脚本（2D/高维占位/FEM stub）。
- `tests/`：`matlab.unittest` 单测与小规模集成测试。

## 3. 模块分解与接口（先把“胶水”接口定好）

> 目标：所有模块围绕统一数据结构协作，避免脚本散落与隐式全局变量。

### 3.1 核心数据结构（MATLAB struct）

- `spec`：问题定义
  - `spec.d`（维度）
  - `spec.lb, spec.ub`（物理边界 1×d）
  - `spec.dist(j)`（分布定义：type/params；支持截断）
  - `spec.evalFcn(x)`（真实 evaluator：输入 1×d 或 n×d，输出 n×1 的 g）
- `cfg`：算法配置
  - `cfg.Npool, cfg.N0, cfg.budgetTotal`
  - `cfg.bootstrapM`（默认 20）
  - `cfg.weightTauPolicy`（如按 |g| 分位数自适应）
  - `cfg.acq`（U-function + 距离项 + α 调度参数）
  - `cfg.stop`（Umin 稳定 / Pf 收敛 / 预算等）
  - `cfg.seed`
- `normModel`：归一化参数（`design.md` 两层）
  - `normModel.lb, normModel.ub`（用于 [0,1]^d 缩放）
  - `normModel.mu, normModel.sigma`（z-score，来自训练集并冻结）
- `model`：代理模型（bootstrap 集成 + 超参）
  - `model.models{1..M}`（每个为 SVR）
  - `model.theta`（统一超参记录：KernelScale/BoxConstraint/Epsilon 等）
  - `model.normModel`
- `log`：全过程日志（必须落盘）
  - `log.X_all, log.g_all`（追加序列）
  - `log.stage`（0=DOE,1=offlineAL,2=onlineAL）
  - `log.Pf_hat(t), log.Umin(t), log.alpha(t), log.theta(t)`
  - `log.meta`（seed、时间戳、版本信息、配置快照）

### 3.2 关键函数接口（最小闭环）

- `pool = lqy.pool.buildPoolTruncatedLHS(spec, cfg)`
- `[X0, idx0] = lqy.doe.selectDOE_kmeans(pool, cfg, normModel)`
- `normModel = lqy.norm.fitNormModel(X_train, spec)`（包含 `[0,1]^d` + z-score 所需参数）
- `model = lqy.svr.trainBootstrapSVR(X_train, g_train, cfg, normModel)`
- `[ghat, sigma] = lqy.surrogate.predictSurrogate(model, X_query)`（ghat 来自主模型；sigma 来自 bootstrap）
- `score = lqy.al.acquisitionScore(ghat, sigma, X_query, X_train, cfg, normModel)`（实现 U + 探索项 + α）
- `[stop, state] = lqy.al.stopCriteria(log, cfg)`（Umin 稳定 / Pf 收敛 / 预算等）
- `Pf = lqy.reliability.estimatePf(ghat, cfg)`（A-mean：默认用 `ghat<=0` 指示函数；可选安全侧 `ghat+κsigma<=0`）
- `lqy.io.saveModelAndLog(model, log, outDir)`（写 `model_offline.mat`/`model_online.mat`/`log.mat`）
- `lqy.viz.plot2DResults(spec, pool, log, model, outDir)`（仅 2D 用）

### 3.3 在线纠偏接口（为 RBDO 留好“插槽”）

- `predictSurrogate(x) -> [ghat, sigma]`（A-mean：ghat 来自主模型，sigma 来自 bootstrap）
- `estimateReliability(design) -> Pf, beta`（`beta = -norminv(Pf)`；可先占位/只返回 Pf）
- `onlineRefine(design, budgetRemain) -> updatedModel, usedBudget, updatedLog`

## 4. 里程碑与迭代计划（按风险排序：先跑通，再变强）

### M0（0.5–1 天）：仓库初始化与可运行入口

- 建立目录结构：`src/`, `examples/`, `tests/`。
- 约定配置加载方式：`cfg = examples/config_*.m`（先不引入复杂配置系统）。
- 约定统一随机种子入口：`rng(cfg.seed, 'twister')`。
- 验收：命令行能跑通一个 “Hello pipeline”（即便 evaluator 先用简单函数）。

### M1（2–4 天）：pool + DOE + 日志骨架

- 实现截断分布的 **逆变换采样 + LHS** 生成 `pool_prob`；可选实现 `pool_space` 并合并打标签。
- 实现 `[0,1]^d` 归一化（物理边界缩放），用于距离/聚类/可视化。
- k-means 从 pool 选 DOE（含“中心投影回 pool + 去重/次近邻”）。
- 日志结构落盘（先不做 surrogate）。
- 验收：
  - 2D pool 分布与 DOE 覆盖可视化正常；
  - `log.X_all`/`log.stage` 字段正确累积。

### M2（3–6 天）：SVR 集成训练 + 不确定度输出

- 实现加权 SVR（LSF 邻域优先权重策略按 `design.md`）。
- 实现 bootstrap 集成：训练 `M=20` 个 SVR，输出 `mu/sigma`。
- 超参策略：
  - v0：固定合理默认值 + 可配置；
  - v1：定期触发调参（优先 `fitrsvm OptimizeHyperparameters='auto'`，无工具箱则降级为网格/随机搜索）。
- 验收：
  - 在 2D 解析函数上，训练后 `mu` 与真值误差在 LSF 附近显著优于均匀权重基线（定性+简单数值指标）。

### M3（3–6 天）：离线主动学习闭环（核心）

- 实现 U-function + 全局探索距离项 + α 调度（从探索到开发的过渡）。
- 每轮：
  - 在 pool 上预测 `mu/sigma`；
  - 计算 acquisition，选取一个新点；
  - 调用 `spec.evalFcn` 获取真值并追加训练集；
  - 更新模型与日志（包含 `Pf_hat`、`Umin`、`alpha`、`theta`）。
- 停止条件（按 `design.md`）：预算 / `Umin` 稳定 / `Pf` 收敛（连续 K 次满足）。
- 验收（2D 离线）：
  - 生成 Pf 收敛曲线；
  - 采样点逐步贴近 `g(x)=0`；
  - 预算不超限、能稳定停止并保存 `model_offline.mat`。

### M4（2–4 天）：2D 结果输出与复现实验

- `examples/run_2d_offline.m`：一键运行、输出到 `out/yyyymmdd_HHMMSS_*`。
- 图与数据落盘：
  - Pf 曲线（标注 DOE/AL 阶段分割线）；
  - 真实边界等值线 + 采样点时间序列；
  - `log.mat` + `model_offline.mat`。
- 验收：同一 `seed` 重跑得到一致的采样序列与曲线（浮点容忍范围内）。

### M5（3–6 天）：在线纠偏（离线 + 在线两模式）

- 实现触发条件：
  - 设计点迭代距离阈值；
  - 设计点邻域 `U(x)` 过小/边界判别不稳。
- 构造邻域子池 `C_local`（归一化半径 r 可收缩），加大 α（强开发）。
- 输出：`model_online.mat`（若未触发则复制离线模型），日志 `stage=2` 明确标注在线新增点。
- 验收：在“模拟 RBDO 设计点序列”的脚本里能触发在线加密并改善局部边界拟合（以邻域误差/稳定性指标验证）。

### M6（按需，1–2 周）：对接 FEM/黑盒与工程化

- 统一 evaluator 适配层：
  - 支持外部进程调用/文件 I/O；
  - 缓存（`hash(x)` → `g`）避免重复计算；
  - 失败重试、超时、断点续跑（log 驱动恢复）。
- 并行化（可选）：pool 上预测向量化；FEM 若可并行则增加队列/批量接口（不强制）。
- 验收：替换 `spec.evalFcn` 为 FEM wrapper 后，主循环无需改动即可运行。

## 5. 测试与验收标准（避免“能跑但不可信”）

### 5.1 单元测试（`tests/`）

- `norm`：`[0,1]^d` 缩放边界正确、z-score 参数冻结且对新数据使用一致。
- `pool`：截断边界内、分布边界概率映射正确、可复现（seed 固定）。
- `doe`：k-means 投影回 pool 不越界、去重逻辑有效。
- `svr`：bootstrap 输出 `sigma>=0`，`mu` 维度正确；权重策略不崩溃（极端 g 值）。
- `al`：预算计数正确、停止条件触发逻辑正确、不会重复选同一点。

### 5.2 集成验收（`examples/`）

- 2D 离线：在预算 100 内产生收敛曲线与图；日志字段完整；输出文件齐全。
- 2D 离线+在线：能在给定设计点序列下触发在线阶段，并在日志/图中清晰标注。

### 5.3 质量阈值（建议量化指标，便于对齐）

- **预算**：`N_total <= cfg.budgetTotal` 强约束。
- **稳定性**：同 seed 重跑采样序列一致（或一致到 pool 索引级别）。
- **收敛**：满足至少一种停止条件（预算除外）才算“模型收敛成功”；若只因预算停止，需要在报告里标注“未收敛”。
- **2D 边界精度（建议指标）**：在真实边界附近采样的网格点上，`sign(mu)` 与 `sign(g_true)` 一致率达到预设阈值（例如 ≥90%，阈值可配置）。

## 6. 风险清单与缓解（提前准备 B 方案）

- **工具箱缺失**：`fitrsvm/kmeans/bayesopt` 不可用 → 提供降级实现（简化 kmeans 替代、网格搜索、或切换到 `fitrgp` 备选但保持接口不变）。
- **高维 pool 成本**：`Npool` 过大导致预测耗时 → 采用分批预测、子池筛选、或用 `pool_prob` 作为固定 MC 集降低冗余。
- **模型偏置/误判**：只用 `ghat<=0` 估 Pf 偏乐观 → 提供可选安全侧 `ghat+κsigma<=0` 与 κ 配置。
- **在线触发不稳定**：触发条件过敏/过钝 → 将阈值参数化，并在日志中记录触发原因与阈值命中情况，便于回放调参。
- **FEM 不稳定/失败**：引入缓存、重试、超时与断点恢复；把每次 evaluator 调用的输入/输出/状态写入日志。

## 7. 最小实现优先级（建议你们按此顺序开工）

1. M0 + M1：把 pool/DOE/log 跑通并可视化（最快暴露数据结构问题）。
2. M2：把 A-mean 的 `ghat/sigma` 跑通（主模型预测 + bootstrap σ；否则后续 U-function 无从谈起）。
3. M3：离线主动学习闭环（项目主价值）。
4. M4：复现实验与标准输出（利于协作与评审）。
5. M5：在线纠偏接口（为 RBDO 预留，不阻塞离线交付）。
6. M6：FEM 工程化（最后做，避免早期被外部系统拖慢）。

## 8. 运行方式（最终应支持的 CLI 入口）

- 运行离线 2D 示例：`matlab -batch "run('examples/run_2d_offline.m')"`
- 运行离线+在线 2D 示例：`matlab -batch "run('examples/run_2d_offline_online.m')"`
- 跑测试：`matlab -batch "results=runtests('tests'); assertSuccess(results)"`
