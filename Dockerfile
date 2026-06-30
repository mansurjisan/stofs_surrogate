# CPU-only image for STOFS-GNN.
# Builds the reproducible environment and runs the test suite by default.
# (A synthetic smoke-train entry point is added in a later phase; override CMD then.)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1 is torch's OpenMP runtime (not present in slim images).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch first, in its own layer, so a CUDA build is never pulled in and the
# (large) torch layer is cached independently of source changes.
RUN pip install --upgrade pip \
    && pip install "torch~=2.5" --index-url https://download.pytorch.org/whl/cpu

# Install dependencies using only the metadata + package source, for better layer
# caching (this layer is reused unless deps or package code change).
COPY pyproject.toml requirements.txt README.md ./
COPY stofs_surrogate ./stofs_surrogate
RUN pip install -e ".[dev]"

# Bring in the rest (tests, configs, scripts) for running the suite.
COPY . .

# Default: run the tests. Verify with:
#   docker build -t stofs-gnn . && docker run --rm stofs-gnn
CMD ["pytest", "-q"]
