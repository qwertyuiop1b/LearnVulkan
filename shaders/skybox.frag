#version 450
// 天空盒片段着色器：采样立方体贴图

layout(location = 0) in vec3 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform samplerCube skyboxSampler;

void main()
{
    outColor = texture(skyboxSampler, fragTexCoord);
}
