#version 450

layout(push_constant) uniform SceneParams {
    mat4 viewProjection;
    vec4 cameraTime;
    vec4 terrainWind;
} pc;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 bladeUv;
layout(location = 3) flat in float colorVariation;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 normal = normalize(gl_FrontFacing ? worldNormal : -worldNormal);
    vec3 sunDirection = normalize(vec3(-0.44, 0.82, -0.36));
    vec3 viewDirection = normalize(pc.cameraTime.xyz - worldPosition);
    float diffuse = max(dot(normal, sunDirection), 0.0);
    float backLight = pow(max(dot(-sunDirection, viewDirection), 0.0), 5.0) *
                      (0.25 + bladeUv.y * 0.55);

    vec3 baseDark = mix(vec3(0.025, 0.12, 0.025), vec3(0.05, 0.19, 0.035), colorVariation);
    vec3 tipLight = mix(vec3(0.18, 0.40, 0.055), vec3(0.30, 0.52, 0.08), colorVariation);
    vec3 base = mix(baseDark, tipLight, smoothstep(0.05, 1.0, bladeUv.y));
    float centerVein = exp(-abs(bladeUv.x - 0.5) * 13.0) * 0.12;
    vec3 color = base * (0.28 + diffuse * 0.95 + backLight) + centerVein * vec3(0.20, 0.34, 0.06);

    float distanceToCamera = distance(worldPosition, pc.cameraTime.xyz);
    float fog = 1.0 - exp(-distanceToCamera * 0.0085);
    color = mix(color, vec3(0.48, 0.62, 0.72), clamp(fog, 0.0, 0.76));
    outColor = vec4(color, 1.0);
}
