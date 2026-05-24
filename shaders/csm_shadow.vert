#version 450
// 级联阴影深度通道顶点着色器（第45章）
// 每级联通过 Push Constant 传入对应的 lightSpaceMatrix

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform ShadowPC {
    mat4 lightSpaceMatrix;
} pc;

void main()
{
    gl_Position = pc.lightSpaceMatrix * vec4(inPosition, 1.0);
}
