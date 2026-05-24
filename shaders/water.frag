#version 450

layout(location = 0) in vec4 clipSpacePos;
layout(location = 1) in vec3 fragPos;
layout(location = 2) in vec3 toCamera;
layout(location = 3) in vec2 waterUV;
layout(location = 4) in float time;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D reflectionTex;
layout(set = 0, binding = 2) uniform sampler2D refractionTex;

layout(push_constant) uniform WaterPC {
    float reflectStrength;
    float refractStrength;
    float distortStrength;
    float murkiness;      // 水体浑浊度（影响透明度）
} pc;

// 程序化法线波浪（代替法线贴图，无需纹理文件）
vec3 waveNormal(vec2 uv, float t)
{
    vec2 uv0 = uv * 0.5 + vec2( 0.03,  0.05) * t;
    vec2 uv1 = uv * 0.8 + vec2(-0.05,  0.02) * t;
    // 用差分近似法线
    float h00 = sin(uv0.x * 6.28 + t) * cos(uv0.y * 6.28 + t * 0.7);
    float h10 = sin((uv0.x + 0.01) * 6.28 + t) * cos(uv0.y * 6.28 + t * 0.7);
    float h01 = sin(uv0.x * 6.28 + t) * cos((uv0.y + 0.01) * 6.28 + t * 0.7);
    float h00b = sin(uv1.x * 10.0 + t * 1.3) * sin(uv1.y * 10.0 + t * 0.9) * 0.5;
    float h10b = sin((uv1.x + 0.01) * 10.0 + t * 1.3) * sin(uv1.y * 10.0 + t * 0.9) * 0.5;
    float h01b = sin(uv1.x * 10.0 + t * 1.3) * sin((uv1.y + 0.01) * 10.0 + t * 0.9) * 0.5;
    float dX = ((h10 - h00) + (h10b - h00b));
    float dY = ((h01 - h00) + (h01b - h00b));
    return normalize(vec3(-dX * 0.5, 1.0, -dY * 0.5));
}

void main()
{
    // NDC → UV（用于采样 RTT）
    vec2 ndc     = clipSpacePos.xy / clipSpacePos.w;
    vec2 screenUV = ndc * 0.5 + 0.5;

    // 程序化波浪法线
    vec3 normal = waveNormal(waterUV, time * 0.4);

    // 扰动 UV 模拟折射/反射失真
    vec2 distort = normal.xz * pc.distortStrength;
    vec2 reflectUV  = vec2(screenUV.x, 1.0 - screenUV.y) + distort;
    vec2 refractUV  = screenUV + distort;
    reflectUV = clamp(reflectUV, 0.001, 0.999);
    refractUV = clamp(refractUV, 0.001, 0.999);

    vec3 reflectColor = texture(reflectionTex, reflectUV).rgb;
    vec3 refractColor = texture(refractionTex, refractUV).rgb;

    // Fresnel（Schlick 近似）
    vec3 viewDir  = normalize(toCamera);
    float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);
    fresnel = mix(0.02, 1.0, fresnel);

    // 浅水颜色（给折射加一点蓝绿色调）
    vec3 waterTint   = vec3(0.0, 0.35, 0.45);
    refractColor = mix(refractColor, waterTint, pc.murkiness);

    vec3 finalColor = mix(refractColor, reflectColor, fresnel * pc.reflectStrength);

    // 镜面高光
    vec3 lightDir = normalize(vec3(1.0, 3.0, 1.5));
    vec3 halfDir  = normalize(viewDir + lightDir);
    float spec    = pow(max(dot(normal, halfDir), 0.0), 64.0) * 0.8;
    finalColor   += vec3(spec);

    outColor = vec4(finalColor, 1.0);
}
