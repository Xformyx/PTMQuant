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
# Sage ships separate tarballs for x86_64 and aarch64 (ARM64).
# We detect the build platform at RUN time so the same Dockerfile works on
# both Intel/AMD hosts (CI, Linux servers) and Apple-Silicon Macs.
# TARGETARCH is set automatically by Docker BuildKit to "amd64" or "arm64".
ARG SAGE_VERSION_ARG=${SAGE_VERSION}
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

# ---- Base OS packages ----------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        tar \
 && apt-get install -y --no-install-recommends \
        $(apt-cache search '^libicu[0-9]' 2>/dev/null | sort -rn | head -1 | awk '{print $1}') \
 && rm -rf /var/lib/apt/lists/*

# ---- .NET 8 Runtime (for pythonnet coreclr mode) -------------------------
# AlphaDIA's alpharaw component uses Python.NET (pythonnet) to interface
# with .NET assemblies (Thermo .raw reader, etc.).  On ARM64 Linux the
# Debian mono-runtime-sgen package ships ONLY the mono-sgen executable and
# does NOT install libmonosgen-2.0.so, so clr_loader's find_libmono() always
# returns None and pythonnet fails with "RuntimeError: Could not find libmono".
# .NET 8 Runtime is fully supported on both aarch64 and x86_64 Linux and
# provides the hostfxr / coreclr shared libraries that clr_loader discovers
# via DOTNET_ROOT.  Setting PYTHONNET_RUNTIME=coreclr tells pythonnet to
# bypass Mono entirely and use the coreclr backend instead.
ARG DOTNET_CHANNEL=8.0
RUN wget -q https://dot.net/v1/dotnet-install.sh -O /tmp/dotnet-install.sh \
 && chmod +x /tmp/dotnet-install.sh \
 && /tmp/dotnet-install.sh --channel ${DOTNET_CHANNEL} --runtime dotnet \
        --install-dir /opt/dotnet --no-path \
 && rm /tmp/dotnet-install.sh \
 && /opt/dotnet/dotnet --info | grep -E 'Version|Arch'

ENV DOTNET_ROOT=/opt/dotnet \
    PATH=/opt/dotnet:${PATH} \
    PYTHONNET_RUNTIME=coreclr

ARG SAGE_VERSION=${SAGE_VERSION_ARG}
RUN set -eux; \
    cd /tmp; \
    # Detect the container's CPU architecture and pick the matching Sage tarball.
    # Docker BuildKit sets TARGETARCH to "amd64" on Intel/AMD and "arm64" on Apple Silicon.
    ARCH="$(uname -m)"; \
    case "${ARCH}" in \
        x86_64)   SAGE_ARCH="x86_64-unknown-linux-gnu" ;; \
        aarch64)  SAGE_ARCH="aarch64-unknown-linux-gnu" ;; \
        *)        echo "Unsupported architecture: ${ARCH}" >&2; exit 1 ;; \
    esac; \
    SAGE_TARBALL="sage-v${SAGE_VERSION}-${SAGE_ARCH}.tar.gz"; \
    SAGE_URL="https://github.com/lazear/sage/releases/download/v${SAGE_VERSION}/${SAGE_TARBALL}"; \
    wget -q "${SAGE_URL}"; \
    tar xzf "${SAGE_TARBALL}"; \
    install -m 0755 "sage-v${SAGE_VERSION}-${SAGE_ARCH}/sage" /usr/local/bin/sage; \
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
        # ---- v0.5.9.2 P0: install dask so directLFQ uses parallel
        # protein-intensity estimation.  Without it directLFQ falls back
        # to a single-threaded path and a 50k-precursor x 12-sample
        # phospho job can take 15+ hours in the final 95% step (observed
        # in the field on a 16-core Mac Studio @ 96GB).  With dask the
        # same step runs 5-10x faster and peak RSS drops because dask
        # processes proteins in chunks instead of loading the full pivot
        # at once.
        pip install --no-cache-dir \
            "dask[dataframe]>=2024.5" ; \
        # ---- v0.5.9.2 P0: fail-fast smoke test for the dask path so a
        # regression of this acceleration never reaches GHCR.
        python -c "import dask, dask.dataframe; print('dask path OK, dask=' + dask.__version__)" ; \
        # ---- v0.6.0a2 P0: install AlphaDIA in an *isolated venv* so its
        # exact pins (scipy==1.12, torch>=2.5+cu12, transformers 4.51, etc.)
        # do NOT clobber the v0.5.10 CPU-only stack we just built.
        # v0.6.0a1 (5b875e1) attempted a flat `pip install alphadia[stable]`
        # in the main env; alphadia upgraded torch 2.2.2+cpu -> 2.6.0+cu124
        # but left torchvision 0.17.2 behind, producing
        # `RuntimeError: operator torchvision::nms does not exist` at the
        # next `from peptdeep.pretrained_models import ModelManager` smoke
        # test.  The venv approach is also forward-compatible with future
        # alphadia releases that will likely keep tightening their pins.
        # Footprint: ~1.5 GB extra image size (torch 2.6 + cuda runtimes),
        # which is still acceptable for the v0.6 generation upgrade.
        # Default behaviour is already 'no system site packages'; explicit comment for clarity.
        python -m venv /opt/alphadia-venv ; \
        /opt/alphadia-venv/bin/pip install --upgrade pip ; \
        /opt/alphadia-venv/bin/pip install --no-cache-dir "alphadia[stable]" ; \
        # Expose only the alphadia CLI on PATH.  We deliberately do NOT
        # symlink the venv's `python` so callers who type `python` from
        # the main env still get our v0.5.10 stack with peptdeep+torch 2.2.
        ln -s /opt/alphadia-venv/bin/alphadia /usr/local/bin/alphadia ; \
        # Fail-fast smoke test that uses the venv's python directly so we
        # don't accidentally import the alphadia bits via the main env.
        /opt/alphadia-venv/bin/python -c "import alphadia; print('alphadia import OK, version=' + getattr(alphadia, '__version__', 'unknown'))" ; \
        alphadia --version && echo "alphadia --version OK" ; \
        alphadia --help >/dev/null 2>&1 && echo "alphadia --help OK" ; \
        # Verify the v0.5.10 main env was NOT mutated (regression guard).
        python -c "import torch, transformers, numba, numpy; print(f'main-env intact: torch={torch.__version__} transformers={transformers.__version__} numba={numba.__version__} numpy={numpy.__version__}')" ; \
        python -c "from peptdeep.pretrained_models import ModelManager; print('peptdeep still importable in main env')" ; \
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
