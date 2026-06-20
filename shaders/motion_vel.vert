#version 450
// 第72章：运动模糊 — 速度缓冲生成
layout(location=0) in vec3 inPosition;

layout(binding=0) uniform MotionUBO {
    mat4 currentMVP;
    mat4 prevMVP;
} ubo;

layout(location=0) out vec4 currClip;
layout(location=1) out vec4 prevClip;

void main() {
    currClip    = ubo.currentMVP * vec4(inPosition,1.0);
    prevClip    = ubo.prevMVP    * vec4(inPosition,1.0);
    gl_Position = currClip;
}
