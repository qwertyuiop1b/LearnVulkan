#version 450
// 细分求值着色器（Domain Shader / Tessellation Evaluation Shader）
// 第27章：曲面细分
//
// TES 的职责：
//   为细分后的每个顶点计算最终位置
//   gl_TessCoord：当前顶点在 patch 内的重心坐标（u,v,w）
//
// triangles = 三角形细分模式
// equal_spacing = 均匀间隔（ccw 表示逆时针）
#extension GL_ARB_tessellation_shader : enable

layout(triangles, equal_spacing, ccw) in;

layout(location = 0) in  vec3 inPos[];
layout(location = 1) in  vec3 inNormal[];
layout(location = 2) in  vec3 inColor[];

layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform PC {
    mat4  mvp;
    float tessLevel;
    float time;
} pc;

void main()
{
    // gl_TessCoord.xyz = 重心坐标 (u, v, w)，满足 u+v+w=1
    vec3 u = gl_TessCoord.x * inPos[0];
    vec3 v = gl_TessCoord.y * inPos[1];
    vec3 w = gl_TessCoord.z * inPos[2];
    vec3 pos = u + v + w;

    // 曲面置换：沿法线方向向外凸起（sin波纹效果）
    vec3 normal = normalize(
        gl_TessCoord.x * inNormal[0] +
        gl_TessCoord.y * inNormal[1] +
        gl_TessCoord.z * inNormal[2]
    );
    float wave = 0.08 * sin(pos.x * 8.0 + pc.time * 2.0)
               * cos(pos.y * 8.0 + pc.time * 1.5);
    pos += normal * wave;

    gl_Position = pc.mvp * vec4(pos, 1.0);

    fragColor = gl_TessCoord.x * inColor[0]
              + gl_TessCoord.y * inColor[1]
              + gl_TessCoord.z * inColor[2];
}
