#version 450

layout(push_constant) uniform CameraParams {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 renderParams;
} pc;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in float materialNoise;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(worldNormal);
    vec3 viewDirection = normalize(pc.cameraPosition.xyz - worldPosition);
    vec3 lightDirection = normalize(vec3(-0.52, 0.76, 0.39));
    vec3 halfVector = normalize(lightDirection + viewDirection);

    float heightBlend = smoothstep(-2.5, 2.7, worldPosition.y);
    float strata = 0.5 + 0.5 * sin(worldPosition.y * 5.4 + materialNoise * 7.0);
    float mineral = smoothstep(0.64, 0.88, materialNoise + strata * 0.12);
    vec3 deepRock = vec3(0.055, 0.075, 0.105);
    vec3 weatheredRock = vec3(0.24, 0.30, 0.34);
    vec3 copperVein = vec3(0.78, 0.26, 0.075);
    vec3 albedo = mix(deepRock, weatheredRock, heightBlend * 0.72 + materialNoise * 0.20);
    albedo = mix(albedo, copperVein, mineral * 0.58);

    float diffuse = max(dot(normal, lightDirection), 0.0);
    float skyFill = normal.y * 0.5 + 0.5;
    float specular = pow(max(dot(normal, halfVector), 0.0), mix(18.0, 62.0, mineral));
    float fresnel = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.2);
    vec3 color = albedo * (0.11 + diffuse * 1.05 + skyFill * 0.18);
    color += vec3(1.0, 0.56, 0.23) * specular * (0.20 + mineral * 0.85);
    color += vec3(0.11, 0.39, 0.72) * fresnel * pc.renderParams.z;

    float distanceToCamera = distance(pc.cameraPosition.xyz, worldPosition);
    float fog = 1.0 - exp(-distanceToCamera * 0.018);
    vec3 fogColor = vec3(0.025, 0.055, 0.095);
    color = mix(color, fogColor, clamp(fog, 0.0, 0.52));
    color *= pc.renderParams.y;
    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
