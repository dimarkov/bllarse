# bllarse
Fitting Bayesian last layers or Bayesian "heads" with sparse priors to embeddings of deep neural networks, and pruning their parameters with Bayesian model reduction.

## 🚀 Running the Docker container

Use the included `start_docker.sh` script to start a container with GPU access and your current working directory mounted:

```bash
./start_docker.sh                # Start an interactive shell
```

This assumes you've already built the image:
```
docker build -t bllarse-dev .
``` 

Once inside an interactive shell on the container, you can create a virtual environment with `uv`, ensure required dependencies are installed using `uv sync` and then run code from within the virtual environment using

```bash
source .venv/bin/activate
```
