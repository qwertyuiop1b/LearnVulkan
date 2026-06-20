#version 450
// 第72章：运动模糊 — 输出速度向量（NDC 空间位移）
layout(location=0) in vec4 currClip;
layout(location=1) in vec4 prevClip;
layout(location=0) out vec2 outVelocity;

void main() {
    vec2 curr = (currClip.xy / currClip.w) * 0.5 + 0.5;
    vec2 prev = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
    outVelocity = curr - prev;   // 单位：UV 空间位移
}
