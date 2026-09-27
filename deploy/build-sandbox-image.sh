#!/usr/bin/env bash
# deploy/build-sandbox-image.sh
# ─────────────────────────────────────────────────────────────────────────────
# Build the dedicated code-execution sandbox image and print the pinned ref
# to configure in .env (SANDBOX_IMAGE=...).
#
# The sandbox validator REJECTS mutable tags, so this script pins by digest.
# The image must then be present on whatever daemon the JARVIS api service
# talks to (the sandbox never pulls at runtime).
#
# Usage:
#   ./deploy/build-sandbox-image.sh [version-tag]      # default: 1.0.0
#
# Before building: verify the python:3.12-slim digest for your architecture
# and put it into deploy/Dockerfile.sandbox (FROM line placeholder).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VERSION="${1:-1.0.0}"
IMAGE="jarvis-sandbox:${VERSION}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker CLI not found." >&2
    exit 1
fi

if grep -q "REPLACE_WITH_VERIFIED_DIGEST" "${SCRIPT_DIR}/Dockerfile.sandbox"; then
    echo "ERROR: deploy/Dockerfile.sandbox still pins the placeholder digest." >&2
    echo "Verify the python:3.12-slim digest for your architecture and edit the FROM line:" >&2
    echo "  docker pull python:3.12-slim && docker images --digests python" >&2
    exit 1
fi

echo "==> Building ${IMAGE}"
docker build -t "${IMAGE}" -f "${SCRIPT_DIR}/Dockerfile.sandbox" "${SCRIPT_DIR}"

DIGEST="$(docker images --digests --format '{{.Digest}}' "${IMAGE}" | head -n1)"
if [ -z "${DIGEST}" ]; then
    echo "ERROR: could not read image digest." >&2
    exit 1
fi

PINNED="${IMAGE}@${DIGEST}"
echo "==> Sanity run (interpreter only):"
docker run --rm --network none "${PINNED}" python3 --version

echo
echo "==> Add to .env:"
echo "    SANDBOX_IMAGE=${PINNED}"
echo
echo "==> Ensure the daemon JARVIS uses has this image locally:"
echo "    docker pull ${PINNED}"
