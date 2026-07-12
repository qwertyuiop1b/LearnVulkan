#version 450

layout(set = 0, binding = 0) uniform sampler2D inPosition;
layout(set = 0, binding = 1) uniform sampler2D inNormal;
layout(set = 0, binding = 2) uniform sampler2D inAlbedo;

struct Light {
    vec4 position;
    vec4 color;
};

layout(set = 0, binding = 3) uniform LightUBO {
    Light lights[4];
    vec4 cameraPos;
} ubo;

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

void main()
{
    vec3 fragPos = texture(inPosition, fragUV).rgb;
    vec3 rawNormal = texture(inNormal, fragUV).rgb;
    vec3 albedo = texture(inAlbedo, fragUV).rgb;
    if (dot(albedo, albedo) < 0.001) {
        outColor = vec4(0.01, 0.01, 0.02, 1.0);
        return;
    }

    vec3 normal = normalize(rawNormal * 2.0 - 1.0);
    vec3 viewDirection = normalize(ubo.cameraPos.xyz - fragPos);
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        vec3 toLight = ubo.lights[i].position.xyz - fragPos;
        float distanceToLight = length(toLight);
        float radius = ubo.lights[i].position.w;
        if (distanceToLight > radius)
            continue;
        vec3 lightDirection = toLight / max(distanceToLight, 0.0001);
        float diffuseTerm = max(dot(normal, lightDirection), 0.0);
        vec3 halfway = normalize(lightDirection + viewDirection);
        float specularTerm = pow(max(dot(normal, halfway), 0.0), 32.0);
        float attenuation = 1.0 - smoothstep(radius * 0.5, radius, distanceToLight);
        vec3 radiance = ubo.lights[i].color.rgb * ubo.lights[i].color.a * attenuation;
        diffuse += diffuseTerm * radiance;
        specular += specularTerm * radiance * 0.3;
    }
    outColor = vec4(vec3(0.06) * albedo + diffuse * albedo + specular, 1.0);
}
