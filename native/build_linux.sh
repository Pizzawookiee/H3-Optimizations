#!/usr/bin/env bash
# Build the Linux shared library in Docker and drop it into native/bin/.
#
# Needs Docker running; no GPU, and it works from a Windows host. Compiling
# does not require the hardware -- only the per-device self-test at startup
# does, and that runs on the user's machine.
#
# Usage:  native/build_linux.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=h3-int8-attention-linux-build
OUTPUT="${HERE}/bin/libh3_int8_attention.so"

if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Start Docker Desktop (or dockerd) and retry." >&2
    exit 1
fi

echo "Building ${IMAGE}..."
docker build -f "${HERE}/Dockerfile.linux" -t "${IMAGE}" "${HERE}"

echo "Extracting the library..."
mkdir -p "${HERE}/bin"
container="$(docker create "${IMAGE}")"
trap 'docker rm -f "${container}" >/dev/null 2>&1 || true' EXIT
docker cp "${container}:/src/build/libh3_int8_attention.so" "${OUTPUT}"

echo
ls -la "${OUTPUT}"
echo
echo "Architectures:"
docker run --rm "${IMAGE}" cuobjdump --list-elf build/libh3_int8_attention.so \
    | sed 's/.*\.\(sm_[0-9a-z]*\)\.cubin/\1/' | sort -u | tr '\n' ' '
echo
echo "Done. Commit native/bin/libh3_int8_attention.so."
