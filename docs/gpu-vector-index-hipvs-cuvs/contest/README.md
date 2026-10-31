# AMD V5 提交与复现

本分支基于冻结 commit `634888d9b037153fa143d2c913ba4b25503e1d62`。
当前实现保留 V5 的 NN-Descent 图参数（128 / 256 / 100）和 CAGRA 搜索参数，优化每个索引的查询工作区复用；尚无 Radeon 上的端到端性能结论。

## 提交前检查

```bash
git merge-base --is-ancestor 634888d9b037153fa143d2c913ba4b25503e1d62 HEAD
git rev-parse HEAD
# 用报名绑定的实际仓库检查分支，勿在输出中包含令牌：
git ls-remote git@github.com:Parrotools/OceanBase2026.git refs/heads/amd-cagra-submit
```

平台当前绑定 `Parrotools/OceanBase2026`，AMD 代码必须在该仓库的 `amd-cagra-submit` 分支。
原提测 `f0e885aaed480c523a92140c250af4e2b61e8553` 曾只存在于另一个仓库 `Parrotools/seekdb` 的同名分支；未取得 failMsg，不能据此断言历史失败代码。

按[赛题](https://oceanbase.yuque.com/org-wiki-obtech-vh7w9r/hc3pio/seebnbaoxot1b0tb)要求，答题仓库需保持私有，报名 URL 必须包含仓库只读令牌。令牌由用户直接在报名平台填写，不能提交到本仓库、日志或聊天中。建议选择 Contents: Read-only，访问范围仅该仓库。提测弹窗中的仓库字段只读。当前报名信息页明确显示报名截止后不支持修改（包括 GitHub 地址），因此需要联系组织方更新绑定；不能通过修改浏览器禁用字段绕过。

## 不依赖 GPU 的验证

所有示例产物放在 `~/workspace/amd/evidence/`，不使用系统临时目录。

```bash
cd ~/workspace/amd/seekdb
mkdir -p ../evidence
tools/hipvs_bridge/tests/run_host_tests.sh ../evidence/host-tests
SANITIZE=1 tools/hipvs_bridge/tests/run_host_tests.sh ../evidence/host-asan
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools/hipvs_bridge/bench -v
bash -n tools/hipvs_bridge/build.sh tools/hipvs_bridge/tests/run_host_tests.sh
git diff --check
```

宿主测试链接独立的 cuVS / runtime 测试替身，只验证 C/C++ 控制流、尺寸检查、内存生命周期、失败重试和互斥。它不计算真实近邻，也不能证明 Recall、GPU 路径或实际性能。测试替身只通过显式的测试 include 路径使用，正式 build.sh 不包含它。

## 正式 GPU 编译

须在比赛指定的 Linux / ROCm 7.2.4 / GCC 13.3 / CMake 3.31 / Rust 1.97 环境中运行，并具备冻结的 `deps/3rd` 和 Rust 离线 vendor。组织方固定 hipVS 位于 `/opt/hipvs`，不可替换或修改。

```bash
mkdir -p ../evidence/bridge
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm \
  tools/hipvs_bridge/build.sh "$PWD/../evidence/bridge/libseekdb_cuvs_bridge.so"
cmake -S . -B build_release -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DOB_ENABLE_UNITY=ON \
  -DOB_USE_LLD=ON \
  -DOB_BUILD_CUVS=ON \
  -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB="$PWD/../evidence/bridge/libseekdb_cuvs_bridge.so"
cmake --build build_release --target seekdb
```

使用原有 `validation/run_ddl_persistence_smoke.sh` 验证离线 DDL、重启后 GPU 快照重载、CPU VSAG SQL 及增删行为。该脚本要求隔离容器和真实 GPU 环境，参数详见脚本。`tools/hipvs_bridge/test_persistence.cpp` 可验证 bridge 跨进程读写，不能替代 SQL 和正式数据集测试。

## SQL 端到端测量

`tools/hipvs_bridge/bench/benchmark_sql.py` 是开发测量器，运行环境需提供 PyMySQL；它不增加 seekdb 的编译依赖。输入：

- `base.tsv`：每行 `外部整数ID<TAB>[FP32向量分量]`；ID 唯一。
- `queries.json`：二维数组，每行一条查询向量，正式尺寸为 4096 行。
- `truth.json`：与查询逐行对齐的精确 Top-100 外部 ID 数组；不能直接使用未经映射的行偏移。
- `--binary`：本轮实际运行的 seekdb 二进制路径，记录其 SHA256。
- `--commit`：生成该二进制的完整源码 commit；baseline 必须是冻结 V5。

使用相同硬件、资源限制、数据、输入顺序和独立的新进程/运行目录测试每个版本。每次测量创建全新随机数据库，不复用已有图。测量器不删除任何既有数据库，生成的测试数据库名称写入报告。结束后由操作者移除自己的测试数据库。

```bash
# 先启动 baseline 二进制，并设置必要的向量内存配置。
# 连接密码如需提供，使用测试终端的 MYSQL_PWD，不把密码写到命令或日志。
python3 tools/hipvs_bridge/bench/benchmark_sql.py measure \
  --variant baseline --dataset SIFT250K --dim 128 --nq 4096 \
  --commit 634888d9b037153fa143d2c913ba4b25503e1d62 \
  --binary /path/to/frozen/seekdb --port 2881 \
  --base-tsv /path/to/base.tsv --queries /path/to/queries.json \
  --truth /path/to/truth.json --output ../evidence/sift-baseline.json

# 停止 baseline，从干净运行目录启动 candidate 后执行同一命令，修改：
# --variant candidate --commit <完整候选commit> --binary <候选二进制>
# --output ../evidence/sift-candidate.json
# CPU 对比另起干净进程，使用 --variant cpu（SQL 中 lib=vsag）。

python3 tools/hipvs_bridge/bench/benchmark_sql.py compare \
  --baseline ../evidence/sift-baseline.json \
  --candidate ../evidence/sift-candidate.json \
  --output ../evidence/sift-comparison.json
```

计时从固定离线 `CREATE VECTOR INDEX` 开始，包含第一条返回合法 ID 的查询。后续分别运行 4096 条单行 Top-10 / Top-100 SQL，记录 QPS、近似 P95、Recall 和 ID 完整性。输入的 SHA256 必须一致；查询吞吐比使用两个 k 的几何平均。

Hidden250K 使用 `--dataset Hidden250K --dim 256`，仅在组织方提供合法访问的数据环境中运行；不尝试生成或推测隐藏数据。两个数据集采用 baseline→candidate、candidate→baseline 的交换次序。正式数据不可得时，可用自有数据作开发实验，但不能冒充正式成绩。

JSON 会明确将 GPU 路径和资源限制标为尚未核验，并且不输出官方分数。需要另用外部 profiler 检查真实 CAGRA 建图/搜索、无 CPU 回退和显存/内存峰值，同时完成 VSAG SQL 回归与重启测试。最终分数以组织方统一评测为准。
