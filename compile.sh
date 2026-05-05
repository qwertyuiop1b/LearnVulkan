#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHADER_DIR="${SCRIPT_DIR}/shaders"

if [[ "${OS:-}" == "Windows_NT" ]]; then
  GLSLC_NAME="glslc.exe"
else
  GLSLC_NAME="glslc"
fi

if [[ -n "${VULKAN_SDK:-}" && -x "${VULKAN_SDK}/bin/${GLSLC_NAME}" ]]; then
  GLSLC="${VULKAN_SDK}/bin/${GLSLC_NAME}"
elif command -v "${GLSLC_NAME}" >/dev/null 2>&1; then
  GLSLC="${GLSLC_NAME}"
elif command -v glslc >/dev/null 2>&1; then
  GLSLC="glslc"
else
  echo "Error: glslc not found. Install Vulkan SDK or add glslc to PATH."
  echo "Tip: set VULKAN_SDK to your SDK root path."
  exit 1
fi

echo "Using compiler: ${GLSLC}"
"${GLSLC}" "${SHADER_DIR}/simple.vert" -o "${SHADER_DIR}/simple.vert.spv"
"${GLSLC}" "${SHADER_DIR}/simple.frag" -o "${SHADER_DIR}/simple.frag.spv"
echo "Shader compilation finished."