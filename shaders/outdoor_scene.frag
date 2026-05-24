#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec4 shadowCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform SceneUBO {
    mat4 model; mat4 view; mat4 proj;
    mat4 lightMVP;
    vec4 lightDir;
} ubo;

layout(set = 0, binding = 1) uniform sampler2D shadowMap;

float sampleShadow(vec4 sc)
{
    vec3  proj    = sc.xyz / sc.w;
    proj.xy       = proj.xy * 0.5 + 0.5;
    float depth   = proj.z;
    if (depth > 1.0 || proj.x < 0.0 || proj.x > 1.0
        || proj.y < 0.0 || proj.y > 1.0) return 1.0;
    float shadow  = 0.0;
    vec2  texelSize = 1.0 / textureSize(shadowMap, 0);
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float sd = texture(shadowMap, proj.xy + vec2(x,y)*texelSize).r;
            shadow += sd >= depth - 0.002 ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}

void main()
{
    vec3  N       = normalize(fragNormal);
    vec3  L       = normalize(-ubo.lightDir.xyz);
    float NdotL   = max(dot(N, L), 0.0);
    float shadow  = sampleShadow(shadowCoord);
    vec3  color   = fragColor * (0.15 + NdotL * shadow * 0.85);
    outColor = vec4(color, 1.0);
}
