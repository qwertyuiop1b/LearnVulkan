#version 450
// TAA 解析：当前帧与历史帧混合 + 邻域钳制

layout(set = 0, binding = 0) uniform sampler2D currentColor;
layout(set = 0, binding = 1) uniform sampler2D historyColor;

layout(push_constant) uniform PushConstants {
    float blendFactor;
    vec2 invResolution;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = gl_FragCoord.xy * pc.invResolution;
    vec4 current = texture(currentColor, uv);
    vec4 history = texture(historyColor, uv);
    vec2 texel = 1.0 / textureSize(currentColor, 0);
    vec3 minColor = current.rgb;
    vec3 maxColor = current.rgb;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec3 sampleColor = texture(currentColor, uv + vec2(x, y) * texel).rgb;
            minColor = min(minColor, sampleColor);
            maxColor = max(maxColor, sampleColor);
        }
    }
    history.rgb = clamp(history.rgb, minColor, maxColor);
    vec3 resolved = mix(current.rgb, history.rgb, pc.blendFactor);
    outColor = vec4(resolved, 1.0);
}
