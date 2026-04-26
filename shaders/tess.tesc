#version 450
// 细分控制着色器（Hull Shader / Tessellation Control Shader）
// 第27章：曲面细分
//
// TCS 的职责：
//   1. 决定细分级别（如何把图元细分成更多三角形）
//   2. 计算细分后的控制点
//
// 每个 patch 调用一次（类似 Task Shader），
// 每个控制点（input_vertices）各一个线程
#extension GL_ARB_tessellation_shader : enable

// 每个 patch 有 3 个控制点（三角形 patch）
layout(vertices = 3) out;

// 从顶点着色器传入的数据
layout(location = 0) in  vec3 inPos[];
layout(location = 1) in  vec3 inNormal[];
layout(location = 2) in  vec3 inColor[];

// 传给 TES 的数据
layout(location = 0) out vec3 outPos[];
layout(location = 1) out vec3 outNormal[];
layout(location = 2) out vec3 outColor[];

layout(push_constant) uniform PC {
    mat4  mvp;
    float tessLevel;   // 细分级别（由 CPU 控制）
    float time;
} pc;

void main()
{
    // gl_InvocationID = 当前控制点在 patch 内的索引 (0/1/2)
    // TCS 每个控制点输出必须用 gl_InvocationID 索引（不能用其他变量！）

    // 传递控制点数据给 TES
    outPos[gl_InvocationID]    = inPos[gl_InvocationID];
    outNormal[gl_InvocationID] = inNormal[gl_InvocationID];
    outColor[gl_InvocationID]  = inColor[gl_InvocationID];

    // 仅在第一个控制点线程中设置细分级别（避免重复写入）
    if (gl_InvocationID == 0) {
        // gl_TessLevelOuter[0..3]：外部边的细分数（三角形模式用0/1/2）
        // gl_TessLevelInner[0..1]：内部的细分数
        float level = max(1.0, pc.tessLevel);
        gl_TessLevelOuter[0] = level;
        gl_TessLevelOuter[1] = level;
        gl_TessLevelOuter[2] = level;
        gl_TessLevelInner[0] = level;
    }
}
