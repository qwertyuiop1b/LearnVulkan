#version 450
// Bindless 顶点着色器（第22章）
// 使用 Buffer Device Address 直接通过 GPU 指针访问顶点数据

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

// Buffer Reference：GPU 端指针类型
// 通过 push constant 传入 GPU 地址，着色器直接解引用
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer VertexBuffer {
    vec3 positions[];
};
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer ColorBuffer {
    vec4 colors[];
};

layout(push_constant) uniform PC {
    VertexBuffer vertexPtr;   // GPU 指针！不是 binding，是直接的 64-bit 地址
    ColorBuffer  colorPtr;
    mat4 mvp;
} pc;

layout(location = 0) out vec4 fragColor;

void main()
{
    // 通过 GPU 指针直接访问数据（无需绑定描述符集！）
    vec3 pos   = pc.vertexPtr.positions[gl_VertexIndex];
    vec4 color = pc.colorPtr.colors[gl_VertexIndex];

    gl_Position = pc.mvp * vec4(pos, 1.0);
    fragColor   = color;
}
