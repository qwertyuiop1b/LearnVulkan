#version 450
// ═══════════════════════════════════════════════════════════════════════════
// 顶点着色器：Uniform Buffer 版（第10章起使用）
// 支持 MVP 矩阵变换
// ═══════════════════════════════════════════════════════════════════════════

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

// Uniform Buffer Object（UBO）：每帧从 CPU 更新的常量数据
// set = 0：描述符集索引
// binding = 0：绑定点索引（与 VkDescriptorSetLayoutBinding.binding 对应）
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;       // 模型矩阵（旋转/平移/缩放）
    mat4 view;        // 视图矩阵（相机位置/朝向）
    mat4 projection;  // 投影矩阵（透视/正交）
} ubo;

layout(location = 0) out vec3 fragColor;

void main()
{
    // MVP 变换：将顶点从模型空间 → 世界空间 → 相机空间 → 裁剪空间
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
    fragColor   = inColor;
}
