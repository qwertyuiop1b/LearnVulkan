#version 450
// 顶点着色器：3D 位置版（vec3）+ UBO MVP 变换
// 用于第12章（深度缓冲）和第14章（MSAA）

layout(location = 0) in vec3 inPosition;   // 3D 坐标
layout(location = 1) in vec3 inColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) out vec3 fragColor;

void main()
{
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor   = inColor;
}
