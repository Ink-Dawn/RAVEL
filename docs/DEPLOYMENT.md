# RAVEL-Unified 三地域 4090 部署手册

本文档用于把本仓库部署到新的跨域 GPU 集群。正式策略是
`RAVEL-Unified`，scheduler id 为 `ravel_unified`。

## 1. 先固定服务契约

一次配对比较中的所有 replica 和策略必须使用相同：

- 模型、tokenizer、vLLM 版本和 dtype；
- `max_model_len`、`max_num_seqs`、`max_num_batched_tokens`；
- KV block size、APC、Chunked Prefill 和 eager mode；
- 请求 trace、SLO、输出长度、到达时间和 cache reset 协议。

RAVEL 的服务速率来自每个地域本地生成的 schema-v2 profile。禁止把旧
V100 profile 用于 4090，也禁止在 topology 中手写 Prefill/Decode 速率。
真实跨域 endpoint 必须设置 `RAVEL_NETWORK_DELAY_MODE=physical`，避免
在真实 TCP 延迟之外再次注入 topology RTT。

已验证 serving 版本为强制 V0 engine 的 vLLM 0.8.5.post1。adapter 使用 feature detection；其他版本只有在私有
Scheduler 接口兼容时才能启动，失败时不得关闭 adapter 后继续声称是同一
RAVEL 实现。service profile 的 `engine_impl` 必须与正式启动一致。

## 2. 仓库角色

Router 主机需要完整仓库、模型 tokenizer 和 Router 依赖。每个 serving
主机至少需要：

- `src/ravel_engine_adapter/`；
- `runtime/ravel_vllm_adapter/`；
- `scripts/start_vllm_ravel.sh`；
- `scripts/validate_profile_launch.py`；
- 本机 service profile；
- 模型和 vLLM 环境。

最简单的方式是在所有主机 clone 同一 commit：

```bash
git clone https://github.com/Ink-Dawn/RAVEL.git
cd RAVEL
```

Router 环境：

```bash
python3 -m venv .venv-router
source .venv-router/bin/activate
pip install -r requirements-router.txt
```

GPU 主机优先复用与 CUDA/PyTorch 匹配的 serving 环境。若使用仓库验证
版本：

```bash
python3 -m venv .venv-serving
source .venv-serving/bin/activate
pip install -r requirements-serving.txt
```

## 3. 每个地域本地校准

### 3.1 启动 vanilla 校准后端

校准时不要启用 RAVEL adapter。以下参数只是示例，必须替换成准备正式
运行的 4090 engine contract，并在之后保持完全一致：

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/vllm-python \
  -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /models/Qwen3-1.7B \
  --served-model-name qwen \
  --dtype float \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --block-size 16 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --scheduling-policy priority \
  --enforce-eager \
  --disable-log-requests
```

### 3.2 从同地域运行 profile

必须从 endpoint 所在 serving 主机运行，并选择 endpoint 使用的 GPU。
不要让中心 Router
跨 WAN 校准，否则网络延迟会进入 Prefill 曲线，并在 topology 中被再次
计费。

```bash
/path/to/vllm-python scripts/calibrate_local_service_profile.py \
  --python /path/to/vllm-python \
  --endpoint 127.0.0.1:8000 \
  --model-path /models/Qwen3-1.7B \
  --served-model-name qwen \
  --cluster-id region_a \
  --gpu 0 \
  --dtype float \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --block-size 16 \
  --repeats 5 \
  --output configs/service_profiles/local/region_a.schema-v2.json
```

在 B、C 地域分别生成 `region_b.schema-v2.json` 和
`region_c.schema-v2.json`。profile 会记录 GPU 型号、模型、vLLM、
dtype、engine dimensions、APC/Chunked-Prefill/eager 设置和测量曲线。

即使三地都是 4090，只要驱动、功耗、CPU、vLLM 或 engine 配置不同，
就应分别校准。完全相同的 service class 可以复用 profile，但必须保留
可审计的 fingerprint。

## 4. 测量 directed RTT

复制模板：

```bash
cp configs/three_cluster_rtt.example.json \
  configs/three_cluster_rtt.local.json
```

矩阵是“目标集群行 -> 请求来源列”。例如：

- `region_c.region_b` 是请求从 B 发往 C 的 RTT；
- `region_b.region_c` 是请求从 C 发往 B 的 RTT。

需要独立填写 A/B/C 所有方向，不能只凭 client->A/B/C 三个值推断
B<->C。若网络近似对称，也要显式写出两个方向。

## 5. 在 Router 组装 topology

将三个小型 profile JSON 放到 Router 的
`configs/service_profiles/local/`，然后运行：

```bash
python scripts/build_cluster_topology.py \
  --profiles \
region_a=configs/service_profiles/local/region_a.schema-v2.json,region_b=configs/service_profiles/local/region_b.schema-v2.json,region_c=configs/service_profiles/local/region_c.schema-v2.json \
  --replica-layout 2,2,1 \
  --rtt-matrix configs/three_cluster_rtt.local.json \
  --default-client-region region_a \
  --output configs/three_cluster_2_2_1.local.json
```

工具会拒绝模型、vLLM、dtype、engine dimensions 或 APC/Chunked Prefill
契约不一致的 profile。GPU 型号允许不同，因为各 cluster 使用自己的
测量曲线。

## 6. 启动正式 backend

在每台 serving 主机停止校准进程，然后使用该地域 profile 启动：

```bash
scripts/start_vllm_ravel.sh \
  --profile configs/service_profiles/local/region_a.schema-v2.json \
  --python /path/to/vllm-python \
  --model /models/Qwen3-1.7B \
  --served-model-name qwen \
  --port 8000 \
  --gpu 0 \
  --cpu-set 0-15 \
  --dtype float \
  --max-model-len 16384 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --block-size 16
```

启动脚本先选择指定 GPU，再验证 GPU 型号和完整 engine fingerprint，
随后加载 feature-detected adapter。普通 baseline priority 不使用 RAVEL
保留编码，因此同一 backend 可公平运行所有策略。

长期运行可使用 `screen`：

```bash
screen -S ravel-a0 -dm bash -lc \
  'cd "$HOME/RAVEL" && exec scripts/start_vllm_ravel.sh ...'
```

## 7. 配置 Router

```bash
cp .env.example .env.local
```

编辑模型路径、Python、endpoint 和 topology。endpoint 顺序必须严格对应
topology 中连续的 replica id：2/2/1 为 A0,A1,B0,B1,C0。
`.env.local` 中真实 WAN 使用 `RAVEL_NETWORK_DELAY_MODE=physical`；
只有所有 endpoint 在本机、需要模拟 RTT 时才使用 `synthetic`。

```bash
set -a
source .env.local
set +a
```

矩阵 runner 会从 topology 引用的 profile 自动读取
`max_model_len/max_num_seqs/max_num_batched_tokens/block_size`，新机器不需要
修改 Python 常量。

## 8. 部署检查

```bash
python scripts/validate_deployment.py \
  --topology "$RAVEL_TOPOLOGY" \
  --endpoints "$RAVEL_ENDPOINTS" \
  --expected-model "$RAVEL_SERVED_MODEL_NAME" \
  --json-output results/deployment_readiness.json
```

检查通过只表示 endpoint、topology 和 profile 一致，不代表性能结论已成立。
首次正式 cell 前应对所有 backend 做相同执行路径 warm-up，再重置 APC。

## 9. 运行一个 smoke cell

```bash
python scripts/run_jitserve_three_cluster_matrix.py \
  --topology "$RAVEL_TOPOLOGY" \
  --model-path "$RAVEL_MODEL_PATH" \
  --served-model-name "$RAVEL_SERVED_MODEL_NAME" \
  --endpoints "$RAVEL_ENDPOINTS" \
  --result-root results/smoke \
  --workloads lmsys \
  --speedups 1 \
  --policies RAVEL-Unified \
  --request-count 20
```

确认 20/20 完成、无 timeout、每个策略 arrival span 一致后，再使用新的
result root 运行正式矩阵。

## 10. 正式矩阵

```bash
screen -S ravel-matrix -dm bash -lc '
  cd "$HOME/RAVEL"
  set -a
  source .env.local
  set +a
  exec "$RAVEL_PYTHON" scripts/run_jitserve_three_cluster_matrix.py \
    --topology "$RAVEL_TOPOLOGY" \
    --model-path "$RAVEL_MODEL_PATH" \
    --served-model-name "$RAVEL_SERVED_MODEL_NAME" \
    --network-delay-mode physical \
    --endpoints "$RAVEL_ENDPOINTS" \
    --result-root results/4090_formal \
    --workloads lmsys,burst,deepresearch \
    --speedups 1,2,4,8,12,16 \
    --policies RAVEL-Unified \
    --request-count 500
'
```

runner 是 open-loop 绝对时间回放：

```text
target_i = start + (trace_ts_i - trace_ts_0) / speedup
```

Router 计算变慢不会自动降低到达率。已完成 cell 仅在 source、trace、
topology、profile、endpoint 和参数 fingerprint 全部一致时复用。

## 11. 输出与公平性

每个 cell 保留 `manifest.json`、`run.log` 和
`request_metrics.csv`；campaign 根目录生成 `RESULTS.md` 与
`summary.json`。生成结果默认不进入 Git。

正式对比必须满足：

- 所有请求均完成，SLO miss 不得从分母删除；
- 所有 RAVEL cells 使用相同输出 token 和 `ignore_eos` 语义；
- Router 不读取真实 output length、未来 arrival 或 dataset name；
- 每个 cell 前重置所有 endpoint 的 Prefix Cache；
- Full SLO、TTFT/E2E mean/p95、TPOT 等分别报告；
- 最优粗体只在同 dataset、同 speedup 内比较。

## 12. 常见故障

- **profile launch mismatch**：启动参数或 GPU 与校准不一致，重新校准或
  恢复原 engine contract，不能跳过校验。
- **unsupported vLLM**：adapter 找不到私有 Scheduler 方法。使用验证版本，
  或先为该版本实现并验证兼容层。
- **endpoint count mismatch**：`RAVEL_ENDPOINTS` 顺序/数量与 replica id
  不一致。
- **RTT 重复计费**：profile 是跨 WAN 测得的。必须改为地域内校准。
- **真实 WAN 延迟翻倍**：误用了 `network-delay-mode=synthetic`。真实
  endpoint 必须使用 `physical`；topology RTT 仍用于 Router 预测。
- **旧 cell 不复用**：fingerprint 已变化；使用新的 result root，这是预期
  的防污染行为。
- **首格异常慢**：backend 未执行统一 warm-up；不能只 warm RAVEL。
