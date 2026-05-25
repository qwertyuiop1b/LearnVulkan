#version 450
// 第80章：植被渲染 — 草叶片元
layout(location=0) in vec3 fragColor;
layout(location=1) in vec2 fragUV;
layout(location=2) in float fragAO;
layout(location=0) out vec4 outColor;

void main() {
    // 简单 Alpha 修剪（草叶轮廓）
    vec2  c    = fragUV * 2.0 - 1.0;
    float mask = 1.0 - abs(c.x) * (1.0 - fragUV.y*0.8);
    if (mask < 0.1) discard;
    outColor = vec4(fragColor * fragAO, 1.0);
}
