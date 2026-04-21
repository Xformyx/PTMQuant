# syntax=docker/dockerfile:1.6
#
# PTMQuant runtime image
# ----------------------
# Bundles the Sage search engine binary and the diaquant Python pipeline
# (exposed as the `ptmquant` CLI).
#
# Build:
#   docker build -t ptmquant .
#
# Run:
#   docker run --rm \
#     -v /data/raw:/input \
#     -v /data/output:/output \
#     ptmquant run --config /input/config.yaml

FROM python:3.11-slim

ARG SAGE_VERSION=0.14.7
ARG SAGE_TARBALL=sage-v${SAGE_VERSION}-x86_64-unknown-linux-gnu.tar.gz
ARG SAGE_URL=https://github.com/lazear/sage/releases/download/v${SAGE_VERSION}/${SAGE_TARBALL}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        tar \
 && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    cd /tmp; \
    wget -q "${SAGE_URL}"; \
    tar xzf "${SAGE_TARBALL}"; \
    install -m 0755 sage-v${SAGE_VERSION}-x86_64-unknown-linux-gnu/sage /usr/local/bin/sage; \
    rm -rf /tmp/sage-*; \
    sage --version

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY diaquant ./diaquant
COPY configs ./configs

RUN pip install --upgrade pip \
 && pip install -e .

WORKDIR /work
VOLUME ["/input", "/output"]

ENTRYPOINT ["ptmquant"]
CMD ["--help"]
