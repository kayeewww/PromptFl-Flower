# PromptFL with Flower - Docker Deployment

基于当前完整版本的PromptFL with Flower的Docker分布式部署方案。

## 🐳 架构

- **1个Server**: 负责联邦学习的协调和参数聚合
- **2个Client**: 分别进行本地训练和评估
- **网络隔离**: 使用Docker网络进行通信

## 🚀 快速开始

### 1. 环境要求

```bash
# 检查Docker和Docker Compose
docker --version
docker-compose --version
```

### 2. 构建和运行

```bash
# 进入docker目录
cd docker

# 使用启动脚本（推荐）
bash run_flower.sh PromptFL cifar10 ViT-B/16 16 10 5 32

# 或者手动启动
docker-compose -f docker-compose-flower.yml up -d
```

### 3. 监控运行

```bash
# 查看服务状态
docker-compose -f docker-compose-flower.yml ps

# 查看日志
docker logs -f promptfl-flower-server
docker logs -f promptfl-flower-client1
docker logs -f promptfl-flower-client2

# 实时查看所有日志
docker-compose -f docker-compose-flower.yml logs -f
```

### 4. 停止服务

```bash
docker-compose -f docker-compose-flower.yml down
```

## ⚙️ 配置选项

### 启动脚本参数

```bash
bash run_flower.sh [TRAINER] [DATASET] [BACKBONE] [N_CTX] [ROUNDS] [LOCAL_EPOCHS] [BATCH_SIZE]
```

**参数说明**:
- `TRAINER`: PromptFL, CoOp, Baseline (默认: PromptFL)
- `DATASET`: cifar10, cifar100, caltech101 (默认: cifar10)
- `BACKBONE`: ViT-B/16, ViT-B/32, RN50 (默认: ViT-B/16)
- `N_CTX`: 上下文token数量 (默认: 16)
- `ROUNDS`: 联邦学习轮数 (默认: 10)
- `LOCAL_EPOCHS`: 本地训练轮数 (默认: 5)
- `BATCH_SIZE`: 批次大小 (默认: 32)

### 示例配置

```bash
# PromptFL with CIFAR-10
bash run_flower.sh PromptFL cifar10 ViT-B/16 16 10 5 32

# CoOp with Caltech101
bash run_flower.sh CoOp caltech101 ViT-B/32 8 15 3 16

# Baseline fine-tuning
bash run_flower.sh Baseline cifar10 RN50 0 5 2 64
```

## 📁 文件结构

```
docker/
├── server_flower.py           # 服务器实现
├── client_flower.py           # 客户端实现
├── docker-compose-flower.yml  # Docker Compose配置
├── Dockerfile.flower          # Docker镜像定义
├── requirements-flower.txt    # Python依赖
├── run_flower.sh             # 启动脚本
├── test_docker_flower.py     # 测试脚本
└── README_DOCKER.md          # 本文档
```

## 🧪 测试

```bash
# 运行完整测试
python test_docker_flower.py

# 快速测试构建
docker-compose -f docker-compose-flower.yml build

# 测试网络连通性
docker-compose -f docker-compose-flower.yml up -d
docker exec promptfl-flower-client1 ping -c 3 server
```

## 🔧 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 检查端口占用
   lsof -i :8080
   
   # 修改docker-compose-flower.yml中的端口映射
   ports:
     - "8081:8080"  # 改为8081
   ```

2. **内存不足**
   ```bash
   # 减少批次大小
   bash run_flower.sh PromptFL cifar10 ViT-B/16 16 10 5 16
   ```

3. **网络问题**
   ```bash
   # 重建网络
   docker-compose -f docker-compose-flower.yml down
   docker network prune
   docker-compose -f docker-compose-flower.yml up -d
   ```

### 调试命令

```bash
# 进入容器调试
docker exec -it promptfl-flower-server bash
docker exec -it promptfl-flower-client1 bash

# 查看详细日志
docker logs --details promptfl-flower-server

# 检查容器资源使用
docker stats
```

## 📊 监控和结果

### 日志位置
- 容器内日志: `/app/logs/`
- 主机映射: `./logs/`

### 结果输出
- 训练结果: `../output/`
- 模型检查点: `../output/checkpoints/`
- 可视化图表: `../output/plots/`

### 性能监控

```bash
# 实时监控资源使用
docker stats promptfl-flower-server promptfl-flower-client1 promptfl-flower-client2

# 查看网络流量
docker exec promptfl-flower-server netstat -i
```

## 🎯 高级配置

### 自定义网络

```yaml
# 在docker-compose-flower.yml中修改
networks:
  promptfl-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.22.0.0/16
```

### 资源限制

```yaml
# 为服务添加资源限制
services:
  server:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### 持久化存储

```yaml
# 添加数据卷
volumes:
  - ./persistent_data:/app/data
  - ./persistent_output:/app/output
```

## 🌸 与原版本的兼容性

这个Docker版本完全基于当前的完整版本，支持：

- ✅ 所有Trainer类型 (PromptFL, CoOp, Baseline)
- ✅ 所有数据集支持
- ✅ 完整的配置系统
- ✅ 原始脚本兼容性
- ✅ 分布式部署

## 📚 参考

- [Flower Documentation](https://flower.dev/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [PromptFL Paper](https://arxiv.org/abs/2208.11625)