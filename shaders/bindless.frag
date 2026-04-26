#version 450
// Bindless 片段着色器（第22章）
// 通过 descriptor indexing 从超大纹理数组中采样（无需重绑定）

#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in  vec4 fragColor;
layout(location = 1) in  flat uint texIndex;  // 从顶点着色器传来的纹理索引

// Bindless 纹理数组：一次绑定所有纹理，通过运行时索引访问
// descriptor_count = 4096 表示数组最多 4096 个纹理
layout(set = 0, binding = 0) uniform sampler2D textures[];   // 变长数组

layout(location = 0) out vec4 outColor;

void main()
{
    // nonuniformEXT: 告诉驱动同一 wave 内不同线程可能访问不同纹理索引
    // （不加此修饰符在某些 GPU 上会产生错误）
    vec4 texColor = texture(textures[nonuniformEXT(texIndex)], vec2(0.5));
    outColor = fragColor * texColor;
}
