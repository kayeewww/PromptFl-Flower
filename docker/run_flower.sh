#!/bin/bash

# PromptFL with Flower - Docker 启动脚本
# 基于当前完整版本的Docker部署

echo "🌸 PromptFL with Flower - Docker Deployment"
echo "============================================="

# 检查Docker和Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# 解析参数
TRAINER=${1:-"PromptFL"}
DATASET=${2:-"cifar10"}
BACKBONE=${3:-"ViT-B/16"}
N_CTX=${4:-"16"}
ROUNDS=${5:-"10"}
LOCAL_EPOCHS=${6:-"5"}
BATCH_SIZE=${7:-"32"}

echo "Configuration:"
echo "  Trainer: $TRAINER"
echo "  Dataset: $DATASET"
echo "  Backbone: $BACKBONE"
echo "  N_CTX: $N_CTX"
echo "  Rounds: $ROUNDS"
echo "  Local Epochs: $LOCAL_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "============================================="

# 创建必要的目录
mkdir -p ../data ../output logs

# 停止现有容器
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose-flower.yml down

# 构建镜像
echo "🔨 Building Docker images..."
docker-compose -f docker-compose-flower.yml build

# 启动服务
echo "🚀 Starting PromptFL with Flower..."

# 更新docker-compose文件中的参数
export TRAINER_TYPE=$TRAINER
export DATASET_NAME=$DATASET
export BACKBONE_NAME=$BACKBONE
export N_CTX_VALUE=$N_CTX
export NUM_ROUNDS=$ROUNDS
export LOCAL_EPOCHS_VALUE=$LOCAL_EPOCHS
export BATCH_SIZE_VALUE=$BATCH_SIZE

# 启动容器
docker-compose -f docker-compose-flower.yml up -d

echo "✅ Services started!"
echo ""
echo "📊 Monitor logs:"
echo "  Server:   docker logs -f promptfl-flower-server"
echo "  Client 1: docker logs -f promptfl-flower-client1"
echo "  Client 2: docker logs -f promptfl-flower-client2"
echo ""
echo "🛑 Stop services:"
echo "  docker-compose -f docker-compose-flower.yml down"
echo ""
echo "🔍 Check status:"
echo "  docker-compose -f docker-compose-flower.yml ps"