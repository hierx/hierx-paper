# =============================================================================
# hierx-paper reproducibility image
#
# Two stages:
#   base    — Python 3.12 + pinned deps + hierx package
#   compute — Benchmark scripts + pre-computed results + figures
#
# Build and run (via docker compose):
#   docker compose run --rm hierx sh -c "./reproduce.sh figures && cp -r paper_figs_final /output/"
#   docker compose run --rm hierx sh -c "./reproduce.sh reproduce && cp -r paper_figs_final /output/"
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: base — system deps, pinned Python deps, hierx
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        osmium-tool \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/hierx-paper

COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# Install hierx (required for benchmarks and accessibility runs)
ARG HIERX_VERSION=0.1.1
RUN pip install --no-cache-dir "git+https://github.com/hierx/hierx.git@v${HIERX_VERSION}" \
    && python -c "from hierx import Hierarchy; print('hierx OK')"

# ---------------------------------------------------------------------------
# Stage 2: compute — scripts, pre-computed results, figures
# ---------------------------------------------------------------------------
FROM base AS compute

WORKDIR /workspace/hierx-paper

# Benchmark scripts + pre-computed JSON results
# (.dockerignore excludes *.npy, *.npz, and large *_data.json files)
COPY benchmarks_final/ benchmarks_final/

# Curated paper figures and data
COPY paper_figs_final/ paper_figs_final/

# Interactive map viewer and generation scripts
COPY interactive_maps/ interactive_maps/

# Pipeline smoke tests
COPY tests/ tests/

# Zenodo manifest for archived data
COPY zenodo_manifest.txt .

# Zenodo DOI — baked into image, overridable at runtime
ARG ZENODO_DOI=10.5281/zenodo.19062193
ENV ZENODO_DOI=${ZENODO_DOI}

# Orchestration
COPY reproduce.sh .
RUN chmod +x reproduce.sh

CMD ["./reproduce.sh", "figures"]
