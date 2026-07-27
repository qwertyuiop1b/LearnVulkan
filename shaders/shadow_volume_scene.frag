#version 450
layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec3 color;
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 projection;
    vec4 lightPosition;
    vec4 cameraPosition;
} scene;
layout(push_constant) uniform LightingMode { uint mode; } pc;
layout(location = 0) out vec4 outColor;
void main() {
    if (pc.mode == 0u) {
        outColor = vec4(color * 0.13, 1.0);
        return;
    }
    vec3 normal = normalize(worldNormal);
    vec3 toLight = scene.lightPosition.xyz - worldPosition;
    float distanceToLight = length(toLight);
    vec3 lightDirection = toLight / max(distanceToLight, 0.0001);
    float diffuse = max(dot(normal, lightDirection), 0.0);
    vec3 viewDirection = normalize(scene.cameraPosition.xyz - worldPosition);
    vec3 halfDirection = normalize(lightDirection + viewDirection);
    float specular = pow(max(dot(normal, halfDirection), 0.0), 48.0);
    float attenuation = 1.0 / (1.0 + 0.045 * distanceToLight + 0.012 * distanceToLight * distanceToLight);
    vec3 direct = (color * diffuse + vec3(0.32) * specular) * attenuation * 2.8;
    outColor = vec4(direct, 1.0);
}
