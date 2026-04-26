#version 450
// 实例化渲染顶点着色器（第17章）
// gl_InstanceIndex 标识当前实例编号

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

// 实例数据：每个实例的变换信息（通过实例顶点缓冲传入）
layout(location = 2) in vec3 instancePos;    // 世界坐标偏移
layout(location = 3) in vec3 instanceColor;  // 实例颜色
layout(location = 4) in float instanceScale; // 缩放

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) out vec3 fragColor;

void main()
{
    // 每个实例独立缩放和平移（不需要单独的 model 矩阵 UBO！）
    vec3 worldPos = inPosition * instanceScale + instancePos;
    gl_Position   = ubo.projection * ubo.view * vec4(worldPos, 1.0);
    fragColor     = inColor * instanceColor;
}
