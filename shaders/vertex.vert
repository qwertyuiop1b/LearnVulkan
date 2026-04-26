#version 450
// ═══════════════════════════════════════════════════════════════════════════
// 顶点着色器：顶点缓冲版（第09章起使用）
// ═══════════════════════════════════════════════════════════════════════════

// layout(location = N) in：从顶点缓冲读取的顶点属性
// location 必须与 VkVertexInputAttributeDescription.location 对应
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main()
{
    gl_Position = vec4(inPosition, 0.0, 1.0);
    fragColor   = inColor;
}
