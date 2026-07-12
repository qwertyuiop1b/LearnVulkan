#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;

struct Hit {
    float distance;
    vec3 position;
    vec3 normal;
    vec3 albedo;
};

float sphereHit(vec3 ro, vec3 rd, vec3 c, float r) {
    vec3 oc = ro - c;
    float b = dot(oc, rd);
    float h = b * b - dot(oc, oc) + r * r;
    return h > 0.0 ? -b - sqrt(h) : 1e20;
}

Hit traceScene(vec3 ro, vec3 rd) {
    Hit hit;
    hit.distance = 1e20;
    hit.albedo = vec3(0.0);
    vec3 centers[4] = vec3[](vec3(-1.65, -0.1, -1.2), vec3(0.0, -0.32, -2.8),
                             vec3(1.55, 0.0, -1.3), vec3(0.5, 1.15, -4.3));
    vec3 colors[4] = vec3[](vec3(0.95, 0.06, 0.035), vec3(0.72, 0.74, 0.78),
                            vec3(0.04, 0.82, 0.16), vec3(0.04, 0.22, 0.95));
    for (int i = 0; i < 4; ++i) {
        float t = sphereHit(ro, rd, centers[i], 0.82);
        if (t > 0.0 && t < hit.distance) {
            hit.distance = t;
            hit.position = ro + rd * t;
            hit.normal = normalize(hit.position - centers[i]);
            hit.albedo = colors[i];
        }
    }
    if (abs(rd.y) > 1e-5) {
        float t = (-1.0 - ro.y) / rd.y;
        if (t > 0.0 && t < hit.distance) {
            hit.distance = t;
            hit.position = ro + rd * t;
            hit.normal = vec3(0.0, 1.0, 0.0);
            float side = smoothstep(0.0, 2.7, abs(hit.position.x));
            hit.albedo = mix(vec3(0.62), mix(vec3(0.78, 0.08, 0.04), vec3(0.04, 0.58, 0.10), side), 0.28);
        }
    }
    return hit;
}

vec3 rayDirection(vec2 uv) {
    vec2 screen = uv * 2.0 - 1.0;
    screen.x *= pc.p[0].x / max(pc.p[0].y, 1.0);
    return normalize(pc.p[2].xyz +
                     (pc.p[3].xyz * screen.x + pc.p[4].xyz * screen.y) * pc.p[2].w);
}

void main() {
    vec3 eye = pc.p[1].xyz;
    vec3 rd = rayDirection(inUV);
    Hit center = traceScene(eye, rd);
    if (center.distance > 1e19) {
        outColor = vec4(vec3(0.025, 0.04, 0.07), 1.0);
        return;
    }

    vec3 lightDirection = normalize(vec3(-0.42, 0.86, 0.28));
    float direct = max(dot(center.normal, lightDirection), 0.0);
    vec3 indirect = vec3(0.0);
    float weightSum = 0.0;
    float radius = pc.p[5].x;
    int sampleCount = clamp(int(pc.p[5].y + 0.5), 1, 16);
    float seed = fract(sin(dot(gl_FragCoord.xy, vec2(91.7, 37.1))) * 47453.5453);
    for (int i = 0; i < 16; ++i) {
        if (i >= sampleCount)
            break;
        float fi = float(i) + seed;
        float ring = sqrt((fi + 0.5) / float(sampleCount));
        float angle = fi * 2.39996323;
        vec2 sampleUv = inUV + vec2(cos(angle), sin(angle)) * ring * radius;
        if (any(lessThan(sampleUv, vec2(0.0))) || any(greaterThan(sampleUv, vec2(1.0))))
            continue;
        Hit neighbor = traceScene(eye, rayDirection(sampleUv));
        if (neighbor.distance > 1e19)
            continue;
        vec3 delta = neighbor.position - center.position;
        float distanceSquared = max(dot(delta, delta), 0.02);
        vec3 direction = delta * inversesqrt(distanceSquared);
        float geometry = max(dot(center.normal, direction), 0.0) *
                         max(dot(neighbor.normal, -direction), 0.0);
        float weight = geometry / (1.0 + distanceSquared * 0.7);
        indirect += neighbor.albedo * weight;
        weightSum += weight;
    }
    indirect /= max(weightSum, 0.18);
    vec3 color = center.albedo * (0.09 + direct * 0.88 + indirect * pc.p[5].z);
    float contact = smoothstep(0.0, 0.45, center.position.y + 1.0);
    color *= mix(0.64, 1.0, contact);
    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
