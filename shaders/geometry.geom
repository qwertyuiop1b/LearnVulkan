#version 450
// 几何着色器（第34章）：法线可视化
//
// 几何着色器的能力：
//   - 接收图元（点/线/三角形），输出不同数量的图元
//   - 可以"放大"（每个三角形→多个三角形）
//   - 可以"缩小"（丢弃不需要的图元）
//   - 访问整个图元的所有顶点数据
//
// 本例：将每个三角形的法线绘制为绿色线段
// 输出 = 原始三角形 + 3条法线线段

layout(triangles) in;   // 输入：三角形（3个顶点）
layout(line_strip, max_vertices = 14) out;  // 输出：线段（最多14个顶点）
// 14 = 原始三角形(4顶点轮廓) + 3条法线(每条2顶点)×3 = 4+6 = 10，留余量

layout(location = 0) in  vec3 inPos[];    // [3] 三角形的3个顶点位置
layout(location = 1) in  vec3 inColor[];  // [3] 颜色

layout(location = 0) out vec4 fragColor;

layout(push_constant) uniform PC {
    mat4  mvp;
    float normalLength;  // 法线显示长度
    float time;
} pc;

// 计算三角形法线
vec3 calcNormal()
{
    vec3 v0 = inPos[0];
    vec3 v1 = inPos[1];
    vec3 v2 = inPos[2];
    return normalize(cross(v1 - v0, v2 - v0));
}

void main()
{
    vec3 normal = calcNormal();

    // ── 输出原始三角形（线框）────────────────────────────────────────────
    for (int i = 0; i < 3; ++i) {
        gl_Position = pc.mvp * vec4(inPos[i], 1.0);
        fragColor   = vec4(inColor[i], 1.0);
        EmitVertex();   // ← 输出顶点！
    }
    // 闭合三角形
    gl_Position = pc.mvp * vec4(inPos[0], 1.0);
    fragColor   = vec4(inColor[0], 1.0);
    EmitVertex();
    EndPrimitive();   // ← 完成当前图元，开始下一个

    // ── 输出每个顶点的法线线段 ───────────────────────────────────────────
    vec3 faceCenter = (inPos[0] + inPos[1] + inPos[2]) / 3.0;

    // 从三角形中心出发，画法线线段
    // 线段起点
    gl_Position = pc.mvp * vec4(faceCenter, 1.0);
    fragColor   = vec4(0.2, 1.0, 0.2, 1.0);   // 绿色
    EmitVertex();

    // 线段终点（沿法线方向延伸 normalLength）
    gl_Position = pc.mvp * vec4(faceCenter + normal * pc.normalLength, 1.0);
    fragColor   = vec4(0.0, 1.0, 0.0, 1.0);
    EmitVertex();
    EndPrimitive();
}
