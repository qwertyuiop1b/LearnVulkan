#version 450
// 阴影贴图深度通道顶点着色器（第18章）
// 从光源视角渲染场景，只写深度值

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform LightSpacePC {
    mat4 lightSpaceMatrix;  // 光源的 projection * view
} pc;

void main()
{
    gl_Position = pc.lightSpaceMatrix * vec4(inPosition, 1.0);
}
