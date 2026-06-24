# 最简集群示例：分布式 KV Cache（Docker Compose）

用 `docker compose` 拉起 `flour` 的分布式 KV Cache 集群，演示
**元数据/数据分离**的架构：

- **Master**（`flour-master`）：只管元数据——Worker 注册/心跳、容量、对象状态、
  放置与读路由，**从不经手 KV 字节**。
- **Worker**（`flour-worker`）：保存 KV 对象字节并直接提供读写；启动时向 Master
  注册并定期心跳，Master 重启后会自动重新注册。
- **engine**（可选，`flour`）：OpenAI 兼容推理服务，把集群当作远程前缀 KV 对象
  存储。Master/Worker 不可用时会自动回退到本地 prefill。

拓扑：`1 Master + 2 Worker (+ 1 可选 engine)`，所有服务共用同一镜像，只构建一次。

本示例支持**两种部署形态**，可按需选择：

- **分离式（默认）**：engine 与 Worker 分属不同进程/节点，见下面第 1–3 节。
- **co-located**：每个节点把 engine 和 Worker 放在同一进程、同一端口，写本地、读短路，见第 4 节。

两种形态的 Master 都是同一个独立进程，只管理 Worker。

## 1. 仅启动 KV 集群（Master + 2 Worker）

```sh
docker compose -f examples/cluster/docker-compose.yml up --build
```

Master 的元数据端口映射到宿主机 `localhost:8081`。Worker 的数据面地址
（`http://worker1:8090` 等）只在 compose 网络内可解析。

## 2. 冒烟测试

在集群网络内部跑一遍完整协议：两阶段 PUT（`put_start` → 直接把字节写到选中的
Worker → `put_commit`），再做一次带路由的 GET 并校验字节一致：

```sh
docker compose -f examples/cluster/docker-compose.yml \
  exec -T master sh -s < examples/cluster/smoke-test.sh
```

预期输出结尾：

```
OK: round-tripped "hello-distributed-kv-cache" via http://worker1:8090
```

## 3. 可选：接入推理 engine

engine 需要本地模型目录（含 `config.json`、`tokenizer.json` 与 safetensors 权重）。
用 `MODEL_DIR` 指定后，以 `engine` profile 启动：

```sh
MODEL_DIR=/abs/path/to/model \
  docker compose -f examples/cluster/docker-compose.yml --profile engine up --build
```

engine 的 OpenAI 兼容接口在 `localhost:8080`，已通过
`--remote-kv-enabled --remote-kv-master-url=http://master:8081` 接入集群。

## 4. 可选：co-located 模式（每节点 engine + worker 同进程）

每个节点在**同一个进程、同一个端口**上同时跑推理 engine 和一个内嵌的 KV-cache
Worker。写入优先落到本节点自己的 Worker，本地产生的 KV 读取完全不走网络；跨节点
复用仍走数据面。Master 仍是独立进程，只管理 Worker（从不管理 engine）。

启动一个 Master + 两个 co-located 节点：

```sh
MODEL_DIR=/abs/path/to/model \
  docker compose -f examples/cluster/docker-compose.yml --profile colocated up --build
```

`node1` 的 OpenAI 接口映射到宿主机 `localhost:8080`。每个节点用 `--worker-id` 向
Master 注册自己的内嵌 Worker，并用 `--advertise-url` 通告地址，供其他节点拉取它的
KV 字节。

## 涉及的二进制

本示例用到三个入口（`src/bin/` 与库 crate）：

- `flour-master` —— `--host --port`
- `flour-worker` —— `--worker-id --master-url --advertise-url --host --port --capacity-bytes`
- `flour`（推理服务）—— 分离式用 `--remote-kv-enabled --remote-kv-master-url`；
  co-located 用 `--colocated-worker --worker-id --advertise-url --remote-kv-master-url --distkv-capacity-bytes`

协议细节见 `docs/plan/plan-distkv.md`。
