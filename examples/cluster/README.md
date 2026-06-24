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

## 涉及的二进制

本示例新增了两个独立入口（`src/bin/`），把库里的 distkv 路由暴露成可独立运行的进程：

- `flour-master` —— `--host --port`
- `flour-worker` —— `--worker-id --master-url --advertise-url --host --port --capacity-bytes`

协议细节见 `docs/plan/plan-distkv.md`。
