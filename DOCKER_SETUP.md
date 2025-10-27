# Docker Setup for Hybrid Deep-Learning FSA-Benchmark

This directory includes Docker configuration files for running the Hybrid Deep-Learning model for fail-slow disk detection in the FSA-Benchmark environment.

## Files

- **Dockerfile**: Multi-stage Docker image with Python 3.11, PyTorch, and all required dependencies
- **docker-compose.yml**: Orchestration file with multiple services (training, Jupyter, etc.)
- **.dockerignore**: Optimizes Docker build by excluding unnecessary files

## Quick Start

### 1. Build the Docker Image

```bash
# Using docker-compose (recommended)
docker-compose build

# Or using docker directly
docker build -t fsa-hybrid-model:latest .
```

### 2. Run the Hybrid Model

```bash
# Using docker-compose with pre-configured settings
docker-compose run --rm train

# Or using docker directly
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/index:/app/index:ro \
  -v $(pwd)/output:/app/output:rw \
  fsa-hybrid-model:latest \
  python scripts/hybrid_cnn_rnn_attention.py \
    --perseus_dir data \
    --index_file index/all_drive_info.csv \
    --out_dir output \
    --batch_size 128 \
    --test_fraction 0.1
```

### 3. Run Jupyter Lab for Interactive Analysis

```bash
# Using docker-compose
docker-compose up jupyter

# Then access: http://localhost:8888

# Or using docker directly
docker run -it --rm \
  -p 8888:8888 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/index:/app/index:ro \
  -v $(pwd)/output:/app/output:rw \
  -v $(pwd):/app \
  fsa-hybrid-model:latest \
  jupyter lab --ip=0.0.0.0 --allow-root
```

## Services

### `hybrid-model`
Main service for running the hybrid deep-learning model.

**Usage:**
```bash
docker-compose run --rm hybrid-model python scripts/hybrid_cnn_rnn_attention.py [options]
```

### `jupyter`
Jupyter Lab service for interactive notebook exploration and analysis.

**Usage:**
```bash
docker-compose up jupyter
# Access at http://localhost:8888
```

### `train`
Dedicated training service with optimized resource allocation.

**Usage:**
```bash
docker-compose run --rm train
```

## Volume Mounts

| Local Path | Container Path | Permission | Purpose |
|------------|----------------|-----------|---------|
| `./data` | `/app/data` | Read-only | Input time-series data |
| `./index` | `/app/index` | Read-only | Cluster indices and metadata |
| `./output` | `/app/output` | Read-write | Model outputs and predictions |
| `./scripts` | `/app/scripts` | Read-only | Python scripts |

## Environment Variables

Key environment variables set in the containers:

- `PYTHONUNBUFFERED=1` - Real-time output logging
- `PYTHONDONTWRITEBYTECODE=1` - Prevent .pyc file generation
- `PIP_NO_CACHE_DIR=1` - Smaller Docker image
- `TORCH_HOME=/tmp/.torch` - PyTorch cache location

## Resource Limits

### Default Configuration

- **CPU**: 2-4 cores
- **Memory**: 4-8 GB

### Customization

Edit `docker-compose.yml` to adjust resource limits:

```yaml
deploy:
  resources:
    limits:
      cpus: '8'      # Increase CPU cores
      memory: 16G    # Increase memory
```

## GPU Support (Optional)

To use GPU acceleration:

1. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

2. Update `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

3. Run training:

```bash
docker-compose run --rm train
```

## Advanced Usage

### Run Model with Custom Parameters

```bash
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/output:/app/output:rw \
  fsa-hybrid-model:latest \
  python scripts/hybrid_cnn_rnn_attention.py \
    --perseus_dir data \
    --index_file index/all_drive_info.csv \
    --out_dir output \
    --batch_size 256 \
    --test_fraction 0.1
```

### Interactive Shell

```bash
docker run -it --rm \
  -v $(pwd):/app \
  fsa-hybrid-model:latest \
  /bin/bash
```

### Execute Python Commands

```bash
docker run --rm \
  -v $(pwd):/app \
  fsa-hybrid-model:latest \
  python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Out of Memory (OOM) Errors

Reduce batch size or number of workers:

```bash
docker run --rm \
  -v $(pwd):/app \
  fsa-hybrid-model:latest \
  python scripts/hybrid_cnn_rnn_attention.py \
    --batch-size 64 \
    --num-workers 2
```

### Slow Performance

- Check available disk space
- Verify volume mount permissions
- Monitor CPU/memory usage:
  ```bash
  docker stats
  ```

### CUDA Not Available

If you need CPU-only execution, the image automatically falls back to CPU. For GPU support, ensure nvidia-docker is installed and configured.

## Cleanup

```bash
# Stop all running containers
docker-compose down

# Remove built image
docker-compose down --rmi local

# Or manually
docker rmi fsa-hybrid-model:latest

# Clean up volumes
docker volume prune
```

## Performance Tips

1. **Use SSD storage** for data and output directories
2. **Mount data with `:ro` flag** for read-only access to prevent accidental modifications
3. **Allocate adequate memory** (at least 8GB recommended)
4. **Use GPU** if available for significant speedup
5. **Run on systems with sufficient disk space** (100GB+ recommended)

## Additional Resources

- [Hybrid Model Documentation](../Hybrid-Deep-Model.ipynb)
- [Main README](../README.md)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Docker Documentation](https://docs.docker.com/)

## Support

For issues or questions:
1. Check the [main README](../README.md)
2. Review [Hybrid-Deep-Model.ipynb](../Hybrid-Deep-Model.ipynb)
3. Examine container logs: `docker logs fsa-hybrid-model`
