FROM mcr.microsoft.com/devcontainers/python:3

# Install additional packages
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        tig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir uv \
    && echo 'eval "$(uv generate-shell-completion bash < /dev/null 2>/dev/null)" || true' >> /etc/bash.bashrc
