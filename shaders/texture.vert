#version 450
// 顶点着色器：纹理版（第11章起使用）

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;   // UV 坐标

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main()
{
    gl_Position  = ubo.projection * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor    = inColor;
    fragTexCoord = inTexCoord;
}
