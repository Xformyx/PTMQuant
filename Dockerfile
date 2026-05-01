# syntax=docker/dockerfile:1.6
#
# PTMQuant / diaquant runtime image
# ---------------------------------
# Bundles:
#   * The Sage search engine binary (Rust, MIT-licensed).
#   * The diaquant Python pipeline (exposed as the `ptmquant` / `diaquant` CLI).
#   * AlphaPeptDeep (opt-in, default ON) together with its CPU-only PyTorch
#     runtime and the pretrained transformer models, so predicted spectral
#     libraries and AlphaPeptDeep-based RT rescoring work completely offline.
#
# The image version == `diaquant/__init__.py::__version__` thanks to the
# dynamic version wiring in pyproject.toml; it is surfaced via the
# `org.opencontainers.image.version` OCI label and `diaquant --version`
# so that downstream platforms can verify which release actually ran.
#
# Build arguments
# ---------------
#   WITH_ALPHAPEPTDEEP (default: 1)
#       Install AlphaPeptDeep + torch (CPU) and download the pretrained models
#       at build time.  Set to 0 for a lean image (~350 MB) — the pipeline
#       transparently falls back to Sage's theoretical library via the
#       `pred_lib_fallback_in_silico: true` default.
#
#   PEPTDEEP_STRICT (default: 0)
#       When 1, fail the image build if `peptdeep install-models` cannot
#       fetch the pretrained models.  Default leaves the warning behaviour
#       unchanged so CI can still build images in network-constrained
#       environments, but writes /etc/ptmquant/peptdeep_status.txt
#       (`ok` / `failed`) which run_manifest.json surfaces at run time.
#
#   SAGE_VERSION (default: 0.14.7)
#       Sage release to bundle.
#
# Build:
#   docker build -t ptmquant:latest .
#   # lean variant
#   docker build --build-arg WITH_ALPHAPEPTDEEP=0 -t ptmquant:lean .
#
# Run:
#   docker run --rm --user $(id -u):$(id -g) \
#       -v /data/raw:/input \
#       -v /data/output:/output \
#       -v /data/predicted_lib_cache:/cache/predicted_libs \
#       ptmquant:latest run --config /input/config.yaml

FROM python:3.11-slim

# ---- Build-time arguments ------------------------------------------------
ARG SAGE_VERSION=0.14.7
ARG SAGE_TARBALL=sage-v${SAGE_VERSION}-x86_64-unknown-linux-gnu.tar.gz
ARG SAGE_URL=https://github.com/lazear/sage/releases/download/v${SAGE_VERSION}/${SAGE_TARBALL}
ARG WITH_ALPHAPEPTDEEP=1
ARG PEPTDEEP_STRICT=0
# Filled by the GHCR workflow via --build-arg; kept here so OCI labels
# always end up with *some* value when the image is built manually.
ARG DIAQUANT_VERSION=unknown
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

# ---- OCI image metadata --------------------------------------------------
LABEL org.opencontainers.image.title="PTMQuant (diaquant)" \
      org.opencontainers.image.description="Open-source DIA proteomics quantification pipeline with universal PTM support (Sage + directLFQ + AlphaPeptDeep)" \
      org.opencontainers.image.source="https://github.com/Xformyx/PTMQuant" \
      org.opencontainers.image.documentation="https://github.com/Xformyx/PTMQuant/blob/main/README.md" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="Xformyx" \
      org.opencontainers.image.version="${DIAQUANT_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      io.ptmquant.sage.version="${SAGE_VERSION}" \
      io.ptmquant.with_alphapeptdeep="${WITH_ALPHAPEPTDEEP}"

# ---- Runtime environment -------------------------------------------------
# HOME is overridden below once the non-root user is created; the initial
# value still lets `peptdeep install-models` write to /root/peptdeep during
# the build, after which the artefacts are moved to /opt/peptdeep.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Force peptdeep to use CPU so the image works on any host without a GPU
    PEPTDEEP_DEVICE=cpu \
    # Shared predicted-library cache: matches PTM-platform bind mount
    # (see v0.5.3 release notes).  Users can override this at run time.
    PTMQUANT_LIB_CACHE_DIR=/cache/predicted_libs

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

# peptdeep writes its models to $HOME/peptdeep by default.  We first download
# them to /root/peptdeep (build-time HOME), then relocate to /opt/peptdeep so
# the non-root user added further down can still read them without needing
# write access to /root.
ENV HOME=/root \
    PEPTDEEP_HOME=/root/peptdeep

RUN set -eux; \
    mkdir -p /etc/ptmquant; \
    if [ "${WITH_ALPHAPEPTDEEP}" = "1" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu \
            "torch==2.2.2" "torchvision==0.17.2" ; \
        pip install -e ".[deeplearning]" ; \
        # ---- v0.5.9 P0: pin transformers/numba/numpy to AlphaPeptDeep's tested
        # versions so peptdeep's transformer-based models can be imported.
        # Without these pins, peptdeep>=1.4 silently pulls the latest
        # transformers (4.50+) which moved `GenerationMixin` out of
        # `transformers.generation` and breaks every model load with
        # `ImportError: cannot import name 'GenerationMixin'`.
        # See alphapeptdeep/requirements/requirements.txt (transformers==4.47.0).
        pip install --no-cache-dir \
            "transformers==4.47.0" "numba==0.60.0" "numpy<2" ; \
        # ---- v0.5.9 P0: fail-fast import smoke test BEFORE downloading models.
        # If peptdeep cannot even be imported, fail the build immediately so a
        # broken image is never published to GHCR.  Previously this silently
        # fell through to runtime where every job hit the same ImportError.
        python -c "from peptdeep.pretrained_models import ModelManager; print('peptdeep import OK')" ; \
        # Download the pretrained AlphaPeptDeep transformer models once at
        # build time so `docker run` never needs network access.
        if peptdeep install-models ; then \
            echo "ok ${DIAQUANT_VERSION}" > /etc/ptmquant/peptdeep_status.txt ; \
        else \
            echo "failed ${DIAQUANT_VERSION}" > /etc/ptmquant/peptdeep_status.txt ; \
            if [ "${PEPTDEEP_STRICT}" = "1" ]; then \
                echo "[fatal] peptdeep install-models failed and PEPTDEEP_STRICT=1"; \
                exit 1 ; \
            else \
                echo "[warn] peptdeep install-models failed; predicted_library will fall back to Sage's built-in theoretical library at run time (run_manifest.json will record peptdeep_status=failed)" ; \
            fi ; \
        fi ; \
        # Relocate model cache out of /root so the non-root user can read it
        if [ -d /root/peptdeep ]; then \
            mv /root/peptdeep /opt/peptdeep ; \
        else \
            mkdir -p /opt/peptdeep ; \
        fi ; \
    else \
        pip install -e . ; \
        echo "disabled ${DIAQUANT_VERSION}" > /etc/ptmquant/peptdeep_status.txt ; \
        mkdir -p /opt/peptdeep ; \
    fi; \
    chmod -R a+rX /opt/peptdeep /etc/ptmquant

# ---- Non-root user (uid 1000 matches the default docker host uid) -------
# Mounted volumes keep their host ownership; this also means files written
# to /output are owned by uid 1000, which is what PTM-platform expects.
RUN useradd --create-home --uid 1000 --shell /bin/bash ptmq \
 && mkdir -p /cache/predicted_libs /work /input /output \
 && chown -R ptmq:ptmq /app /cache /work /home/ptmq

ENV HOME=/home/ptmq \
    PEPTDEEP_HOME=/opt/peptdeep

USER ptmq

# ---- Runtime volumes and entrypoint -------------------------------------
WORKDIR /work
VOLUME ["/input", "/output", "/cache/predicted_libs"]

# `diaquant --version` reads the dynamic version from the installed package
# metadata; the healthcheck also exercises the Python import path so a broken
# install is caught early.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD diaquant --version >/dev/null 2>&1 || exit 1

ENTRYPOINT ["diaquant"]
CMD ["--help"]
