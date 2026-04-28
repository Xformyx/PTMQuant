# syntax=docker/dockerfile:1.6
#
# PTMQuant / diaquant 0.5.0 runtime image
# ---------------------------------------
# Bundles:
#   * The Sage search engine binary (Rust, MIT-licensed).
#   * The diaquant Python pipeline (exposed as the `ptmquant` / `diaquant` CLI).
#   * AlphaPeptDeep (0.5.0 opt-in) together with its CPU-only PyTorch runtime
#     and the pretrained transformer models, so predicted spectral libraries
#     and AlphaPeptDeep-based RT rescoring work completely offline.
#
# Build arguments
# ---------------
#   WITH_ALPHAPEPTDEEP (default: 1)
#       Install AlphaPeptDeep + torch (CPU) and download the pretrained models
#       at build time.  Set to 0 for a lean image that mirrors the 0.4.x
#       behaviour — the 0.5.0 code still loads thanks to the lazy import and
#       the `pred_lib_fallback_in_silico: true` default, so the pipeline simply
#       skips the predicted-library step when AlphaPeptDeep is missing.
#
#   SAGE_VERSION (default: 0.14.7)
#       Sage release to bundle.
#
# Build:
#   # default: full 0.5.0 image (~2.2 GB) including AlphaPeptDeep + models
#   docker build -t ptmquant:0.5.0 .
#
#   # lean 0.4.x-compatible image (~350 MB), AlphaPeptDeep disabled
#   docker build --build-arg WITH_ALPHAPEPTDEEP=0 -t ptmquant:0.5.0-lean .
#
# Run:
#   docker run --rm \
#       -v /data/raw:/input \
#       -v /data/output:/output \
#       ptmquant:0.5.0 run --config /input/config.yaml

FROM python:3.11-slim

# ---- Build-time arguments ------------------------------------------------
ARG SAGE_VERSION=0.14.7
ARG SAGE_TARBALL=sage-v${SAGE_VERSION}-x86_64-unknown-linux-gnu.tar.gz
ARG SAGE_URL=https://github.com/lazear/sage/releases/download/v${SAGE_VERSION}/${SAGE_TARBALL}
ARG WITH_ALPHAPEPTDEEP=1

# ---- Runtime environment -------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Force peptdeep to use CPU so the image works on any host without a GPU
    PEPTDEEP_DEVICE=cpu \
    # Make peptdeep / alphabase cache directories predictable so that the
    # models downloaded during image build are reused at runtime
    HOME=/root \
    PEPTDEEP_HOME=/root/peptdeep

# ---- Base OS packages + Sage binary -------------------------------------
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

# ---- Python dependencies (cached layer, before source copy) -------------
# Copy only the build metadata first so pip can be cached whenever diaquant's
# source code (but not its declared dependencies) changes.
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./

RUN pip install --upgrade pip

# ---- Install diaquant + AlphaPeptDeep (optional) ------------------------
# When WITH_ALPHAPEPTDEEP=1 we install the CPU-only PyTorch wheel first so
# pip picks the small (~200 MB) CPU build instead of the ~2 GB CUDA one, then
# install diaquant with the [deeplearning] extra.  When disabled, we fall
# back to the lean diaquant install that mirrors the 0.4.x image size.
COPY diaquant ./diaquant
COPY configs ./configs

RUN set -eux; \
    if [ "${WITH_ALPHAPEPTDEEP}" = "1" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu \
            "torch==2.2.2" "torchvision==0.17.2" ; \
        pip install -e ".[deeplearning]" ; \
        # Download the pretrained AlphaPeptDeep transformer models once at
        # build time so `docker run` never needs network access.  Failures
        # here are non-fatal: the image still works thanks to diaquant's
        # `pred_lib_fallback_in_silico` flag.
        ( peptdeep install-models || \
          echo "[warn] peptdeep install-models failed at build time; \
predicted_library will fall back to Sage's built-in theoretical library" ) ; \
    else \
        pip install -e . ; \
    fi

# ---- Runtime volumes and entrypoint -------------------------------------
WORKDIR /work
VOLUME ["/input", "/output"]

ENTRYPOINT ["diaquant"]
CMD ["--help"]
