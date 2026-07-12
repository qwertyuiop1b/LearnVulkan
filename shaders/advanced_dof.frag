#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;

float sphereHit(vec3 ro, vec3 rd, vec3 center, float radius) {
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float h = b * b - c;
    return h > 0.0 ? -b - sqrt(h) : 1e20;
}

vec3 shadeScene(vec3 ro, vec3 rd) {
    float time = pc.p[0].z;
    float closest = 1e20;
    vec3 normal = vec3(0.0, 1.0, 0.0);
    vec3 albedo = vec3(0.08, 0.10, 0.13);

    vec3 centers[5] = vec3[](vec3(-2.4, -0.05, 0.4), vec3(-0.6, -0.25, -1.7),
                             vec3(1.25, 0.05, 0.1), vec3(2.75, -0.38, -2.4),
                             vec3(0.25, 1.35, -4.1));
    vec3 colors[5] = vec3[](vec3(0.95, 0.17, 0.08), vec3(0.06, 0.45, 1.0),
                            vec3(0.98, 0.66, 0.08), vec3(0.20, 0.88, 0.36),
                            vec3(0.72, 0.16, 0.95));
    for (int i = 0; i < 5; ++i) {
        centers[i].y += sin(time * 0.7 + float(i) * 1.9) * 0.08;
        float radius = 0.72 + float(i % 2) * 0.18;
        float hit = sphereHit(ro, rd, centers[i], radius);
        if (hit > 0.0 && hit < closest) {
            closest = hit;
            vec3 position = ro + rd * hit;
            normal = normalize(position - centers[i]);
            albedo = colors[i];
        }
    }

    if (abs(rd.y) > 1e-5) {
        float planeHit = (-0.95 - ro.y) / rd.y;
        if (planeHit > 0.0 && planeHit < closest) {
            closest = planeHit;
            vec3 position = ro + rd * planeHit;
            normal = vec3(0.0, 1.0, 0.0);
            float grid = step(0.94, max(fract(position.x * 0.5), fract(position.z * 0.5)));
            albedo = mix(vec3(0.055, 0.065, 0.08), vec3(0.21, 0.24, 0.28), grid);
        }
    }

    if (closest > 1e19) {
        float horizon = pow(max(0.0, 1.0 - abs(rd.y)), 5.0);
        return mix(vec3(0.015, 0.025, 0.055), vec3(0.16, 0.24, 0.36), horizon);
    }

    vec3 position = ro + rd * closest;
    vec3 lightDir = normalize(vec3(-0.45, 0.82, 0.34));
    float diffuse = max(dot(normal, lightDir), 0.0);
    float rim = pow(1.0 - max(dot(normal, -rd), 0.0), 3.0);
    float shadowPattern = 0.82 + 0.18 * sin(position.x * 2.3 + position.z * 1.7);
    return albedo * (0.14 + diffuse * shadowPattern * 1.15) + rim * vec3(0.12, 0.24, 0.42);
}

void main() {
    vec2 screen = inUV * 2.0 - 1.0;
    screen.x *= pc.p[0].x / max(pc.p[0].y, 1.0);
    vec3 eye = pc.p[1].xyz;
    vec3 forward = normalize(pc.p[2].xyz);
    vec3 right = normalize(pc.p[3].xyz);
    vec3 up = normalize(pc.p[4].xyz);
    float tanHalfFov = pc.p[2].w;
    vec3 baseDirection = normalize(forward + (right * screen.x + up * screen.y) * tanHalfFov);

    float focusDistance = pc.p[5].x;
    float aperture = pc.p[5].y;
    int sampleCount = clamp(int(pc.p[5].z + 0.5), 1, 12);
    vec3 focusPoint = eye + baseDirection * focusDistance /
                      max(dot(baseDirection, forward), 0.08);
    vec3 accumulated = vec3(0.0);
    float pixelSeed = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    for (int sampleIndex = 0; sampleIndex < 12; ++sampleIndex) {
        if (sampleIndex >= sampleCount)
            break;
        float fi = float(sampleIndex) + pixelSeed;
        float radius = sqrt((fi + 0.5) / float(sampleCount));
        float angle = fi * 2.39996323;
        vec2 disk = vec2(cos(angle), sin(angle)) * radius * aperture;
        vec3 rayOrigin = eye + right * disk.x + up * disk.y;
        vec3 rayDirection = normalize(focusPoint - rayOrigin);
        accumulated += shadeScene(rayOrigin, rayDirection);
    }
    vec3 color = accumulated / float(sampleCount);
    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
