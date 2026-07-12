#version 450
// Bindless 顶点着色器（第22章）
// 使用 Buffer Device Address 直接通过 GPU 指针访问顶点数据

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

// Buffer Reference：GPU 端指针类型
// 通过 push constant 传入 GPU 地址，着色器直接解引用
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer VertexBuffer {
    float data[];
};

layout(push_constant) uniform PC {
    mat4 mvp;
    VertexBuffer vertexPtr;
    uint textureIndex;
    float time;
} pc;

layout(location = 0) out vec4 fragColor;
layout(location = 1) flat out uint texIndex;

void main()
{
    // 通过 GPU 指针直接访问数据（无需绑定描述符集！）
    uint base = gl_VertexIndex * 5;
    vec3 pos = vec3(pc.vertexPtr.data[base], pc.vertexPtr.data[base + 1], pc.vertexPtr.data[base + 2]);

    gl_Position = pc.mvp * vec4(pos, 1.0);
    fragColor   = vec4(1.0);
    texIndex    = pc.textureIndex;
}
