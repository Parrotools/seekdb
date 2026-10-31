# AMD 优化验证报告

## 状态

截至 2026-10-30，宿主验证已完成，Radeon GPU 性能验证待完成。本文不声称已通过正式评测，也不填报未测得的 CPU/GPU 加速比。

环境：`prometheus`，Ubuntu x86_64，GCC 15.2，32 个逻辑 CPU，约 30 GiB RAM，NVIDIA RTX 5070 Ti Laptop 12 GiB 与 AMD Raphael 核显。没有 `/opt/rocm/bin/hipcc`、`/opt/hipvs` 或比赛指定 Radeon PRO W7900D，亦未配置比赛冻结的离线依赖。

## 实现

- 保持 V5 的 GPU NN-Descent 建图参数（graph_degree=128、intermediate_graph_degree=256、nn_descent_niter=100）以及每次查询的 `itopk=max(默认值,512,4*k)`、search_width=4。此前减少图度数和迭代数的提交没有 Recall 证据，本版恢复冻结设置。
- 按单个索引句柄复用 GPU 查询/结果缓冲区与搜索参数；按实际需求扩容，不为每条单行 SQL 预分配 4096 条查询的缓冲区。最多保留 8 MiB，大批量临时工作区调用结束后释放。
- 互斥保护同一资源句柄，正确传递拷贝与同步错误，校验输入尺寸和整数溢出；显式同步本资源的 stream。
- 保留基线的 32 MiB cuVS 调用线程栈，避免在尚未验证的数据库小栈上直接执行搜索；移除对每个新索引一次性预留 262144 行向量的行为。
- 保留原提交的 4 MiB 快照中转块与 bridge O3 编译。没有改变快照格式、原始 FP32 数据、外部 ID 映射或 CPU VSAG 接口。
- 新增宿主错误注入测试和 SQL 开发测量器，详细复现步骤见 README。

## 已有测量

相同 GCC 优化选项下，分别编译冻结 V5 bridge 与候选 bridge，连接相同宿主测试替身，执行 4096 次 nq=1 / Top-100 调用：

| 指标 | 冻结 V5 | 候选 |
|---|---:|---:|
| 模拟 RMM 分配调用次数 | 12288 | 3 |
| 模拟设备工作区峰值 | 928 bytes | 928 bytes |
| 本次宿主模拟耗时 | 0.003831 s | 0.004482 s |

该模拟无 GPU、无真实 ANN 计算。它证明缓冲区分配次数下降，不证明真实延迟改善；本轮宿主模拟候选耗时还略高，不能据此宣传速度提升。原始数据位于运行机 `~/workspace/amd/evidence/host-tests/{baseline,candidate}-mock.json`，时间会随负载变化。

另外比较了 256 / 384 MiB 临时文件与 memfd 的宿主中转。差异相对整轮构建很小且未在比赛环境验证，因此没有引入 memfd 改动。

## 检查证据

- 实际 bridge 源码分别按 C11、C++17 编译，`-Wall -Wextra -Werror`：通过。
- C/C++ 宿主行为测试：通过。覆盖参数、增长、无额外分配的热路径、分配失败重试、拷贝/同步错误、四线程串行访问、大工作区释放及序列化重载。
- AddressSanitizer / UndefinedBehaviorSanitizer：通过。
- SQL 测量器的 6 个单元测试：通过，覆盖 Recall / ID 完整性、配对输入、非有限数、相对门槛和吞吐几何平均。
- shell 语法、Git 空白检查：通过。
- 正式 bridge 构建预检：受环境阻塞，输出 `hipcc is not executable: /opt/rocm/bin/hipcc`。未生成冒充 GPU bridge 的替身动态库。

## 需要真实 GPU 完成的对比

| 数据集/路径 | 端到端构建（含首查询） | Top-10/100 QPS与P95 | Recall@10/100 | 峰值显存/内存 |
|---|---|---|---|---|
| SIFT250K 冻结 V5 GPU | 待测 | 待测 | 待测 | 待测 |
| SIFT250K 候选 GPU | 待测 | 待测 | 待测 | 待测 |
| SIFT250K CPU VSAG | 待测 | 待测 | 待测 | 待测 |
| Hidden250K 冻结/候选 GPU | 由合法数据环境或云端评测测量 | 待测 | 待测 | 待测 |

仍需组织方固定 hipVS 版本的完整源码编译、真实 GPU profiler、重启/增删/CPU VSAG 回归、Recall 与查询吞吐门槛，以及资源配额检查。任何分数和加速比必须在这些检查完成后依据配对实测填写。

## 提测前的外部阻塞

平台绑定 URL 为不带令牌的 `https://github.com/Parrotools/OceanBase2026.git`。当前赛题写明要求含只读令牌的 URL，而报名页已锁定。用户选择继续使用公开仓库的原地址提测，以验证实际格式检查；若被格式检查拒绝，则需组织方处理。此前绑定仓库缺少 `amd-cagra-submit` 分支，原代码位于另一个仓库；本次会把候选分支推送到绑定仓库。

没有历史 failMsg，不能认定历史失败属于某个特定代码。解决仓库访问后应由正式评测确认。
