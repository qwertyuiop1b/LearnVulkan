#version 450
// Transform Feedback 捕获顶点着色器（第34章）
// 位置经过 MVP 变换后写入捕获缓冲

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

// xfb_buffer(0) xfb_stride(24)：输出到缓冲0，每组24字节
layout(xfb_buffer = 0, xfb_offset = 0,  location = 2) out vec3 xfbPosition;
layout(xfb_buffer = 0, xfb_offset = 12, location = 3) out vec3 xfbColor;

layout(push_constant) uniform PC { mat4 mvp; } pc;

layout(location = 0) out vec3 fragColor;

void main()
{
    vec4 worldPos = pc.mvp * vec4(inPosition, 1.0);
    gl_Position   = worldPos;
    fragColor     = inColor;

    // 写入 Transform Feedback 缓冲（GPU 捕获顶点变换后的数据）
    xfbPosition = worldPos.xyz;
    xfbColor    = inColor;
}
