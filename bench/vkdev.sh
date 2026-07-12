# Source me: resolves VKDEV to the dGPU's ggml-vulkan device index by NAME.
# Vulkan device enumeration order flips across boots (iGPU/dGPU swap), so a
# hardcoded GGML_VK_VISIBLE_DEVICES index silently lands models on the iGPU
# (vm_validate ENOMEM -> vk::DeviceLostError). Cost us four GLM runs on 2026-07-12.
VKDEV=$(env -u GGML_VK_VISIBLE_DEVICES timeout 20 llama.cpp/build-vk/bin/llama-bench --list-devices 2>&1 |
        sed -n 's/^ggml_vulkan: \([0-9]\{1,\}\) = AMD Radeon RX 9070.*/\1/p' | head -1)
: "${VKDEV:=0}"
export VKDEV
