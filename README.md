# PromptFL with Docker + Flower

## Introduction
The FL training process comprises of two iterative phases, i.e., local training and global aggregation. Thus the learning performance is determined by both the effectiveness of the parameters from local training and smooth aggregation of them. However, these two requirements are not easy to satisfy in edge environment, i.e., edge users often have limited bandwidth and insufficient data, which can cause inefficient parameters aggregation, excessive training time and reduced model accuracy. FL inherently entails a large number of communication rounds and a large amount of labeled data for training, which are often unavailable for edge users. Such challenges are particularly salient under the combined effect of a long training process and unfavorable factors such as non-IID and unbalanced data, limited communication bandwidth, and unreliable and limited device availability.

We revisits the question of how FL mines the distributed data in iterative training rounds, and exploit the emerging foundation model (FM) to optimize the FL training. We investigate the behavior of the nascent model in a standard FL setting using popular off-the-shelf FMs, e.g., CLIP, and methods for FM adaptation. We propose PROMPTFL, a framework that replaces existing federated model training with prompt training, i.e., FL clients train prompts instead of a model, which can simultaneously exploit the insufficient local data and reduce the aggregation overhead. PROMPTFL ships an off-the-shelf public CLIP to users and apply continuous prompts (a.k.a. soft prompts) for FM adaptation, which requires very few data samples from edge users. The framework is technically very simple but effective.

This implementation uses **Docker** for containerization and **Flower** for federated learning orchestration, providing a scalable and production-ready deployment solution.

## How to Run

You can run the Docker-based federated learning system using Docker Compose with the Flower framework.

### Prerequisites

- Docker and Docker Compose installed
- Python 3.9+ (for local development)
- CUDA-compatible GPU (optional, for faster training)

### Quick Start

#### 1. Build and Start the Complete System

```bash
# Navigate to docker directory
cd docker

# Build and start all services (server + 2 clients)
docker-compose -f docker-compose-flower.yml up --build
```

#### 2. Monitor Training Progress

```bash
# View all logs in real-time
docker-compose -f docker-compose-flower.yml logs -f

# View specific component logs
docker logs -f promptfl-flower-server
docker logs -f promptfl-flower-client1
docker logs -f promptfl-flower-client2
```

#### 3. Stop the System

```bash
# Graceful shutdown
docker-compose -f docker-compose-flower.yml down

# Remove all containers and networks
docker-compose -f docker-compose-flower.yml down --volumes --remove-orphans
```

### Training Configuration

The system supports various configuration options through environment variables and command-line arguments:

#### Server Configuration

- `--host`: Server host address (default: 0.0.0.0)
- `--port`: Server port (default: 8080)
- `--rounds`: Number of federated learning rounds (default: 10)
- `--min-clients`: Minimum number of clients required (default: 2)
- `--trainer`: Training method (PromptFL, CoOp, Baseline)
- `--dataset`: Dataset name (cifar10, cifar100, caltech101, etc.)
- `--backbone`: CLIP backbone (ViT-B/16, RN50, etc.)
- `--n-ctx`: Number of context tokens (default: 16)

#### Client Configuration

- `--server-address`: Server address to connect to
- `--client-id`: Unique client identifier
- `--local-epochs`: Number of local training epochs (default: 5)
- `--batch-size`: Training batch size (default: 32)
- `--num-clients`: Total number of clients in federation

### Advanced Usage

#### Custom Configuration

You can modify the Docker Compose configuration to customize the training setup:

```yaml
# docker/docker-compose-flower.yml
services:
  server:
    command: >
      python docker/server_flower.py
      --host 0.0.0.0 
      --port 8080 
      --rounds 20                    # Increase rounds
      --min-clients 3                # Require 3 clients
      --trainer PromptFL
      --dataset cifar100             # Use CIFAR-100
      --backbone RN50                # Use ResNet-50
      --n-ctx 8                      # Reduce context tokens
```

#### Adding More Clients

To add additional clients, extend the Docker Compose file:

```yaml
  client3:
    build:
      context: ..
      dockerfile: docker/Dockerfile.flower
    container_name: promptfl-flower-client3
    command: >
      python docker/client_flower.py 
      --server-address server:8080 
      --client-id 2                  # Unique client ID
      --trainer PromptFL
      --dataset cifar10
      --backbone ViT-B/16
      --n-ctx 16
      --num-clients 3                # Update total clients
```

#### Different Training Methods

**PromptFL (M=16, end)**:
```bash
# Modify docker-compose-flower.yml
command: >
  python docker/server_flower.py
  --trainer PromptFL
  --dataset caltech101
  --backbone RN50
  --n-ctx 16
  --rounds 10
```

**FinetuningFL (Baseline)**:
```bash
# Modify docker-compose-flower.yml  
command: >
  python docker/server_flower.py
  --trainer Baseline
  --dataset caltech101
  --backbone RN50
  --rounds 10
```

**CoOp Training**:
```bash
# Modify docker-compose-flower.yml
command: >
  python docker/server_flower.py
  --trainer CoOp
  --dataset oxford_flowers
  --backbone ViT-B/16
  --n-ctx 16
  --rounds 15
```

### Development and Testing

#### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server locally
python docker/server_flower.py --host localhost --port 8080

# Run client locally (in separate terminal)
python docker/client_flower.py --server-address localhost:8080 --client-id 0
```

#### Testing

```bash
# Run integration tests
python test_promptfl_flower.py

# Run fast tests
python test_promptfl_flower_fast.py

# Run complete system test
python test_complete_system.py
```

### Output and Results

After the experiments, all results are saved to:
- `output/`: Training logs and model checkpoints
- `docker/logs/`: Container-specific logs
- `results/`: Aggregated federation results

The system provides detailed logging including:
- Training progress for each client
- Global model performance metrics
- Communication overhead statistics
- Resource utilization metrics

### Architecture

The Docker-based system consists of:

- **Server Container**: Runs the Flower server with PromptFL strategy
- **Client Containers**: Multiple federated learning clients
- **Shared Volumes**: For data, results, and configuration sharing
- **Network**: Isolated Docker network for secure communication

### Supported Datasets

- CIFAR-10/100
- Caltech-101
- Oxford Flowers
- Food-101
- Oxford Pets
- And more...

### Supported Backbones

- ViT-B/16 (Vision Transformer)
- RN50 (ResNet-50)
- RN101 (ResNet-101)
- ViT-B/32
- ViT-L/14

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change the port mapping in docker-compose.yml
2. **Memory issues**: Reduce batch size or number of clients
3. **CUDA errors**: Set `CUDA_VISIBLE_DEVICES=""` to use CPU only
4. **Network issues**: Ensure Docker network is properly configured

### Performance Optimization

- Use GPU-enabled Docker images for faster training
- Adjust batch sizes based on available memory
- Optimize the number of local epochs vs communication rounds
- Use data parallelism for large datasets

## Citation

If this code is useful in your research, you are encouraged to cite our academic paper:
```
@article{guo2022promptfl,
  title={PromptFL: Let Federated Participants Cooperatively Learn Prompts Instead of Models--Federated Learning in Age of Foundation Model},
  author={Guo, Tao and Guo, Song and Wang, Junxiao and Xu, Wenchao},
  journal={arXiv preprint arXiv:2208.11625},
  year={2022}
}
```

## Acknowledgments

We build and modify the code based on Dassl, CoOp, and Flower frameworks. This Docker-based implementation provides a scalable and production-ready deployment solution for PromptFL federated learning experiments.
