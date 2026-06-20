#version 450
// OIT 合成：不透明背景 + 累积透明层

layout(set = 0, binding = 0) uniform sampler2D opaqueTex;
layout(set = 0, binding = 1) uniform sampler2D accumTex;
layout(set = 0, binding = 2) uniform sampler2D revealTex;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(opaqueTex, 0));
    vec3 opaque = texture(opaqueTex, uv).rgb;
    vec4 accum = texture(accumTex, uv);
    float reveal = texture(revealTex, uv).r;
    vec3 avgColor = accum.rgb / max(accum.a, 1e-5);
    // reveal = Π(1-αi)，透明层贡献 (1-reveal)
    vec3 transparent = avgColor * (1.0 - reveal);
    outColor = vec4(transparent + opaque * reveal, 1.0);
}
