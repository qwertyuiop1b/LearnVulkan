#version 450

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragTangent;
layout(location = 4) in vec3 fragBitangent;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D diffuseMap;
layout(set = 0, binding = 2) uniform sampler2D normalMap;

layout(set = 0, binding = 3) uniform Lighting {
    vec3 lightDir;
    vec3 viewPos;
} lighting;

void main() {
    vec3 normalTex = texture(normalMap, fragTexCoord).rgb * 2.0 - 1.0;
    mat3 tbn = mat3(normalize(fragTangent), normalize(fragBitangent), normalize(fragNormal));
    vec3 normal = normalize(tbn * normalTex);
    vec3 lightDir = normalize(-lighting.lightDir);
    float diff = max(dot(normal, lightDir), 0.15);
    vec3 albedo = texture(diffuseMap, fragTexCoord).rgb;
    outColor = vec4(albedo * diff, 1.0);
}
