#version 450
// Mesh Shader 管线的片段着色器（与普通管线完全相同）
layout(location = 0) in  vec3 inColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(inColor, 1.0); }
