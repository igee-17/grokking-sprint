# Sandbox for agent runs: the container only ever sees the exported workspace.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN npm install -g @anthropic-ai/claude-code
RUN useradd -m agent
USER agent
WORKDIR /work
