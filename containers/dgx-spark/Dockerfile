ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
ARG NODE_IMAGE=node:22.16.0-bookworm-slim

FROM ${NODE_IMAGE} AS node_runtime

FROM ${PYTORCH_IMAGE}

COPY --from=node_runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node_runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-dgx.txt /tmp/vsa-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/vsa-requirements.txt \
    && python -m pip check

ENV PYTHON=python \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VSA_CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /workspace/vsaLLM
