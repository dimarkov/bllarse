# bllarse
Fitting Bayesian last layers or Bayesian "heads" with sparse priors to embeddings of deep neural networks, and pruning their parameters with Bayesian model reduction.

## 🚀 Running the Docker container

Use the included `start_docker.sh` script to start a container with GPU access and your current working directory mounted:

```bash
./start_docker.sh                # Start an interactive shell
./start_docker.sh python         # Run Python in the container
./start_docker.sh bash           # Explicitly run bash
```

This asssumes you've already built the image:
```
docker build -t bllarse-dev .
``` 