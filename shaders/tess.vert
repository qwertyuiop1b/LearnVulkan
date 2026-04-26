#version 450
// 曲面细分顶点着色器（传递数据给 TCS）
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 0) out vec3 outPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outColor;
void main() {
    outPos    = inPosition;
    outNormal = inNormal;
    outColor  = inColor;
    gl_Position = vec4(inPosition, 1.0);  // TCS 会重新计算
}
