#version 450
// 片段着色器：纹理采样（第11章起使用）

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

// 组合图像采样器（Combined Image Sampler）
// set=0 binding=1：纹理绑定在第1个绑定点
layout(set = 0, binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main()
{
    // texture() 函数：使用采样器对纹理进行采样
    // fragTexCoord 经过光栅化插值，范围 [0,1]
    outColor = texture(texSampler, fragTexCoord);
}
