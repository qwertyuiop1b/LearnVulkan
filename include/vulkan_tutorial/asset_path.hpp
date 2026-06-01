#pragma once

/**
 * @file asset_path.hpp
 * @brief 解析 assets/ 目录路径（由 CMake 注入 ASSET_DIR）
 */

#include <filesystem>
#include <string>

namespace vulkan_tutorial {

inline std::string resolveAssetPath(const std::string& relativePath) {
#ifdef ASSET_DIR
    const std::filesystem::path assetRoot = ASSET_DIR;
#else
    const std::filesystem::path assetRoot = "assets";
#endif
    return (assetRoot / relativePath).string();
}

} // namespace vulkan_tutorial
