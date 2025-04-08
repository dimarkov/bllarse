FROM nvidia/cuda:12.8.1-devel-ubuntu20.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt install -y python3.12
RUN apt-get install -y git
RUN apt-get install -y libgl1 libsm6 libxext6