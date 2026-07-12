#version 450

layout(push_constant) uniform SceneParams {
    mat4 viewProjection;
    vec4 cameraTime;
    vec4 terrainWind;
} pc;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in float moisture;
layout(location = 0) out vec4 outColor;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    vec3 normal = normalize(worldNormal);
    float slope = 1.0 - normal.y;
    float relativeHeight = worldPosition.y / max(pc.terrainWind.y, 0.001) + 0.43;
    float rock = smoothstep(0.17, 0.48, slope);
    float dry = 1.0 - smoothstep(0.20, 0.48, moisture);
    float snow = smoothstep(0.72, 0.92, relativeHeight);

    vec3 soil = mix(vec3(0.20, 0.13, 0.065), vec3(0.29, 0.22, 0.105), dry);
    vec3 meadow = mix(vec3(0.095, 0.22, 0.055), vec3(0.055, 0.18, 0.075), moisture);
    vec3 stone = vec3(0.28, 0.30, 0.29);
    vec3 base = mix(meadow, soil, dry * 0.62);
    base = mix(base, stone, rock);
    base = mix(base, vec3(0.76, 0.80, 0.78), snow);
    base *= mix(0.88, 1.10, hash21(floor(worldPosition.xz * 1.4)));

    vec3 sunDirection = normalize(vec3(-0.44, 0.82, -0.36));
    float diffuse = max(dot(normal, sunDirection), 0.0);
    float distanceToCamera = distance(worldPosition, pc.cameraTime.xyz);
    float fog = 1.0 - exp(-distanceToCamera * 0.0085);
    vec3 color = base * (0.27 + diffuse * 0.92);
    color = mix(color, vec3(0.48, 0.62, 0.72), clamp(fog, 0.0, 0.78));
    outColor = vec4(color, 1.0);
}
