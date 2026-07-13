# Source me: resolves VKDEV to your discrete GPU's ggml-vulkan device index BY
# NAME. Vulkan enumeration order flips across boots (iGPU/dGPU swap), so a
# hardcoded GGML_VK_VISIBLE_DEVICES index silently lands the model on the iGPU
# (vm_validate ENOMEM -> vk::DeviceLostError). Cost us four GLM runs on 2026-07-12.
#
# Set VK_GPU_NAME to a substring of your GPU as llama-bench prints it, e.g.:
#   VK_GPU_NAME="RTX 4090" source bench/vkdev.sh
# List devices with: llama.cpp/build-vk/bin/llama-bench --list-devices
VK_GPU_NAME=${VK_GPU_NAME:-Radeon RX 9070}
VKDEV=$(env -u GGML_VK_VISIBLE_DEVICES timeout 20 llama.cpp/build-vk/bin/llama-bench --list-devices 2>&1 |
        sed -n "s/^ggml_vulkan: \([0-9]\{1,\}\) = .*${VK_GPU_NAME}.*/\1/p" | head -1)
: "${VKDEV:=0}"
export VKDEV
