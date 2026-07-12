#version 450

layout(push_constant) uniform WorldPush {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 params;
    vec4 controls;
} pc;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 surfaceUV;
layout(location = 3) flat in uint materialId;
layout(location = 0) out vec4 outColor;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    vec3 normal = normalize(worldNormal);
    vec3 sun = normalize(vec3(-0.45, 0.82, -0.35));
    vec3 view = normalize(pc.cameraPosition.xyz - worldPosition);
    float diffuse = max(dot(normal, sun), 0.0);
    vec3 color;

    if (materialId == 1u) {
        float flow = sin(surfaceUV.x * 7.0 - pc.params.x * 3.2 + sin(surfaceUV.x * 1.7) * 2.0);
        float bankFoam = smoothstep(0.62, 0.96, abs(surfaceUV.y * 2.0 - 1.0));
        vec3 reflected = vec3(0.18, 0.43, 0.59);
        float fresnel = pow(1.0 - max(dot(normal, view), 0.0), 4.0);
        color = mix(vec3(0.025, 0.19, 0.25), reflected, fresnel);
        color += vec3(0.08, 0.18, 0.19) * flow * 0.16;
        color = mix(color, vec3(0.75, 0.88, 0.86), bankFoam * (0.38 + flow * 0.18));
    } else if (materialId == 2u) {
        float edge = smoothstep(0.68, 0.96, abs(surfaceUV.y * 2.0 - 1.0));
        float centerLine = 1.0 - smoothstep(0.025, 0.055, abs(surfaceUV.y - 0.5));
        color = mix(vec3(0.12, 0.115, 0.105), vec3(0.30, 0.27, 0.20), edge);
        color = mix(color, vec3(0.74, 0.62, 0.24), centerLine * step(0.44, fract(surfaceUV.x)));
        color *= 0.35 + diffuse * 0.75;
    } else {
        float height = worldPosition.y;
        float slope = 1.0 - normal.y;
        vec3 grass = mix(vec3(0.075, 0.20, 0.055), vec3(0.20, 0.31, 0.085), hash21(floor(worldPosition.xz * 0.8)));
        vec3 dirt = vec3(0.31, 0.23, 0.13);
        vec3 rock = vec3(0.27, 0.28, 0.27);
        color = mix(grass, dirt, smoothstep(-3.0, -1.0, height));
        color = mix(color, rock, smoothstep(0.30, 0.62, slope));
        color *= 0.22 + diffuse * 0.86;
    }

    float distanceToCamera = distance(worldPosition, pc.cameraPosition.xyz);
    float fog = 1.0 - exp(-distanceToCamera * 0.0042);
    color = mix(color, vec3(0.47, 0.61, 0.72), clamp(fog, 0.0, 0.76));
    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
