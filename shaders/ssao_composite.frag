#version 450
// SSAO 合成（第46章）
// 简单 Blinn-Phong 光照 × 环境光遮蔽

// 全屏四边形 UV 由 gl_FragCoord 推导

layout(set = 0, binding = 0) uniform sampler2D gAlbedo;
layout(set = 0, binding = 1) uniform sampler2D gNormal;
layout(set = 0, binding = 2) uniform sampler2D gPosition;
layout(set = 0, binding = 3) uniform sampler2D aoTex;

layout(set = 0, binding = 4) uniform LightUBO {
    vec4 lightDir;
    vec4 lightColor;
    vec4 cameraPos;
} light;

layout(location = 0) out vec4 outColor;

void main()
{
    vec2 vUV = gl_FragCoord.xy / vec2(textureSize(gAlbedo, 0));
    vec3 albedo   = texture(gAlbedo, vUV).rgb;
    vec3 normal   = normalize(texture(gNormal, vUV).xyz * 2.0 - 1.0);
    vec3 worldPos = texture(gPosition, vUV).xyz;
    float ao      = texture(aoTex, vUV).r;

    vec3 V = normalize(light.cameraPos.xyz - worldPos);
    vec3 L = normalize(-light.lightDir.xyz);
    vec3 H = normalize(V + L);
    float diff = max(dot(normal, L), 0.0);
    float spec = pow(max(dot(normal, H), 0.0), 32.0);

    vec3 ambient  = albedo * 0.15 * ao;
    vec3 diffuse  = albedo * diff * light.lightColor.rgb;
    vec3 specular = vec3(0.04) * spec * light.lightColor.rgb;
    outColor = vec4(ambient + diffuse + specular, 1.0);
}
