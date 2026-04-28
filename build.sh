#!/usr/bin/env bash
#
# Build the PTMQuant Docker image.
#
# Usage:
#   ./build.sh                        # default image (includes peptdeep if Dockerfile default)
#   ./build.sh -t ptmquant:0.5.0      # custom tag
#   ./build.sh --platform linux/amd64 # force amd64 (Apple Silicon: Sage binary is x86_64)
#   ./build.sh --no-cache             # rebuild from scratch
#   ./build.sh --build-arg WITH_ALPHAPEPTDEEP=0   # lean image (no AlphaPeptDeep / torch)
#
# Any extra args are forwarded to `docker build`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

IMAGE_NAME="${IMAGE_NAME:-ptmquant}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PLATFORM="${PLATFORM:-}"

EXTRA_ARGS=()
TAG_OVERRIDE=""
PLATFORM_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tag)
            TAG_OVERRIDE="$2"
            shift 2
            ;;
        --platform)
            PLATFORM_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "${TAG_OVERRIDE}" ]]; then
    FULL_TAG="${TAG_OVERRIDE}"
else
    FULL_TAG="${IMAGE_NAME}:${IMAGE_TAG}"
fi

if [[ -n "${PLATFORM_OVERRIDE}" ]]; then
    PLATFORM="${PLATFORM_OVERRIDE}"
fi

# Auto-select linux/amd64 on Apple Silicon so the amd64 Sage binary works.
if [[ -z "${PLATFORM}" ]]; then
    HOST_OS="$(uname -s)"
    HOST_ARCH="$(uname -m)"
    if [[ "${HOST_OS}" == "Darwin" && "${HOST_ARCH}" == "arm64" ]]; then
        PLATFORM="linux/amd64"
        echo "[build.sh] Apple Silicon detected; defaulting --platform=${PLATFORM}"
    fi
fi

PLATFORM_ARGS=()
if [[ -n "${PLATFORM}" ]]; then
    PLATFORM_ARGS+=("--platform" "${PLATFORM}")
fi

echo "[build.sh] Building image: ${FULL_TAG}"
[[ -n "${PLATFORM}" ]] && echo "[build.sh] Target platform : ${PLATFORM}"

docker build \
    ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} \
    -t "${FULL_TAG}" \
    -f Dockerfile \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    .

echo
echo "[build.sh] Done. Try:"
echo "  docker run --rm ${FULL_TAG} --help"
echo "  docker run --rm \\"
echo "    -v /data/raw:/input \\"
echo "    -v /data/output:/output \\"
echo "    ${FULL_TAG} \\"
echo "    run --config /input/config.yaml"
