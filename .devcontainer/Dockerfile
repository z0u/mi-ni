FROM node:24-slim AS node
FROM mcr.microsoft.com/devcontainers/python:3

# Install Node.js (for npx / agentic coding tools)
COPY --from=node /usr/local/bin/node         /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# Install additional packages
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        bat \
        fd-find \
        git-lfs \
        htop \
        jq \
        ripgrep \
        tig \
        tree \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Debian renames some tools — add canonical symlinks so agents can use expected names
RUN ln -s /usr/bin/batcat /usr/local/bin/bat \
    && ln -s /usr/bin/fdfind /usr/local/bin/fd

# Install Python packages
RUN pip install --no-cache-dir uv \
    && echo 'eval "$(uv generate-shell-completion bash < /dev/null 2>/dev/null)" || true' >> /etc/bash.bashrc
