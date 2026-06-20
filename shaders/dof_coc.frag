#version 450
// 第71章：景深 — CoC（Circle of Confusion）计算
layout(location=0) in vec2 inUV;
layout(location=0) out vec4 outCoc;   // r=coc大小(负=前景,正=背景), gba=原始颜色

layout(set=0,binding=0) uniform sampler2D sceneTex;
layout(set=0,binding=1) uniform sampler2D depthTex;

layout(push_constant) uniform DofPC {
    mat4  invProjView;
    vec4  cameraPos;
    float focusDistance;  // 焦点距离（世界空间）
    float focusRange;     // 焦深范围
    float maxCoc;         // 最大散景半径（像素）
    float pad;
} pc;

float reconstructDepth(vec2 uv) {
    float d = texture(depthTex, uv).r;
    vec4 ndcPos = vec4(uv*2.0-1.0, d, 1.0);
    vec4 worldH = pc.invProjView * ndcPos;
    return length(worldH.xyz/worldH.w - pc.cameraPos.xyz);
}

void main() {
    float depth = reconstructDepth(inUV);
    float coc   = (depth - pc.focusDistance) / pc.focusRange;
    coc = clamp(coc, -1.0, 1.0) * pc.maxCoc;
    vec3 color  = texture(sceneTex, inUV).rgb;
    outCoc = vec4(coc, color);
}
