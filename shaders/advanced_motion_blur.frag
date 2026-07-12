#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;

float sphereHit(vec3 ro, vec3 rd, vec3 center, float radius) {
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float h = b * b - dot(oc, oc) + radius * radius;
    return h > 0.0 ? -b - sqrt(h) : 1e20;
}

vec3 renderAtTime(vec3 ro, vec3 rd, float time) {
    float nearest = 1e20;
    vec3 normal = vec3(0.0, 1.0, 0.0);
    vec3 albedo = vec3(0.0);
    for (int i = 0; i < 6; ++i) {
        float phase = float(i) * 1.0472;
        vec3 center = vec3(sin(time * (0.75 + float(i) * 0.05) + phase) * (2.5 + float(i % 2)),
                           -0.15 + cos(time * 1.2 + phase) * 0.55,
                           -1.2 - float(i) * 0.85);
        float hit = sphereHit(ro, rd, center, 0.58 + float(i % 3) * 0.09);
        if (hit > 0.0 && hit < nearest) {
            nearest = hit;
            normal = normalize(ro + rd * hit - center);
            albedo = 0.52 + 0.48 * cos(vec3(0.0, 2.1, 4.2) + phase);
        }
    }
    if (abs(rd.y) > 1e-5) {
        float hit = (-1.0 - ro.y) / rd.y;
        if (hit > 0.0 && hit < nearest) {
            nearest = hit;
            vec3 p = ro + rd * hit;
            normal = vec3(0.0, 1.0, 0.0);
            float lines = smoothstep(0.91, 0.99, max(fract(p.x), fract(p.z)));
            albedo = mix(vec3(0.035, 0.045, 0.065), vec3(0.20, 0.25, 0.31), lines);
        }
    }
    if (nearest > 1e19)
        return mix(vec3(0.015, 0.025, 0.05), vec3(0.12, 0.20, 0.31), pow(1.0 - abs(rd.y), 4.0));
    vec3 light = normalize(vec3(-0.36, 0.86, 0.31));
    float diffuse = max(dot(normal, light), 0.0);
    float specular = pow(max(dot(reflect(-light, normal), -rd), 0.0), 36.0);
    return albedo * (0.12 + diffuse * 1.12) + specular * vec3(1.0, 0.72, 0.42);
}

void main() {
    vec2 screen = inUV * 2.0 - 1.0;
    screen.x *= pc.p[0].x / max(pc.p[0].y, 1.0);
    vec3 eye = pc.p[1].xyz;
    vec3 rd = normalize(pc.p[2].xyz +
                        (pc.p[3].xyz * screen.x + pc.p[4].xyz * screen.y) * pc.p[2].w);
    float shutter = pc.p[5].x;
    int sampleCount = clamp(int(pc.p[5].y + 0.5), 1, 16);
    float centerTime = pc.p[0].z;
    vec3 color = vec3(0.0);
    for (int i = 0; i < 16; ++i) {
        if (i >= sampleCount)
            break;
        float samplePosition = (float(i) + 0.5) / float(sampleCount) - 0.5;
        color += renderAtTime(eye, rd, centerTime + samplePosition * shutter);
    }
    color /= float(sampleCount);
    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
