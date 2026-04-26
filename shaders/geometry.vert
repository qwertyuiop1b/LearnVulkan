#version 450
// 几何着色器输入顶点着色器（第34章）
layout(location = 0) in  vec3 inPosition;
layout(location = 1) in  vec3 inColor;
layout(location = 0) out vec3 outPos;
layout(location = 1) out vec3 outColor;
void main() {
    outPos   = inPosition;
    outColor = inColor;
    gl_Position = vec4(inPosition, 1.0);
}
