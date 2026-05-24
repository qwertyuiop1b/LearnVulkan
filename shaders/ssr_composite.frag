#version 450
// SSR 合成（第47章）
// 将屏幕空间反射与场景颜色混合

// 全屏 UV 由 gl_FragCoord 推导

layout(set = 0, binding = 0) uniform sampler2D sceneColor;
layout(set = 0, binding = 1) uniform sampler2D reflectionTex;

layout(push_constant) uniform PC {
    float mixStrength;
} pc;

layout(location = 0) out vec4 outColor;

void main()
{
    vec2 vUV = gl_FragCoord.xy / vec2(textureSize(sceneColor, 0));
    vec3 base = texture(sceneColor, vUV).rgb;
    vec4 refl = texture(reflectionTex, vUV);
    vec3 color = mix(base, refl.rgb, refl.a * pc.mixStrength);
    outColor = vec4(color, 1.0);
}
