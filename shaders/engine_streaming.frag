#version 450

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0, std430) readonly buffer ChunkTable { vec4 chunks[]; };
layout(push_constant) uniform Push { vec4 p[8]; } pc;

float terrain(vec2 xz, out float resident, out float edge) {
    float grid = pc.p[1].x;
    vec2 gridPos = xz * 0.42 + grid * 0.5;
    ivec2 cell = ivec2(floor(gridPos));
    if (any(lessThan(cell, ivec2(0))) || any(greaterThanEqual(cell, ivec2(int(grid))))) {
        resident = 0.0; edge = 0.0; return -2.0;
    }
    vec4 chunk = chunks[cell.y * int(grid) + cell.x];
    resident = chunk.x;
    vec2 f = fract(gridPos);
    edge = 1.0 - smoothstep(0.43, 0.5, max(abs(f.x - 0.5), abs(f.y - 0.5)));
    float hills = sin(xz.x * 0.55 + chunk.y * 5.0) * cos(xz.y * 0.48) * 0.42;
    return hills + chunk.y * 0.55 - 0.35;
}

void main() {
    vec2 q = vUv * 2.0 - 1.0;
    q.x *= pc.p[0].x / pc.p[0].y;
    vec3 ro = pc.p[2].xyz;
    vec3 rd = normalize(pc.p[3].xyz + q.x * pc.p[4].xyz * pc.p[3].w + q.y * pc.p[5].xyz * pc.p[3].w);
    vec3 sky = mix(vec3(0.025, 0.055, 0.10), vec3(0.22, 0.42, 0.58), max(rd.y, 0.0));
    float t = 0.2;
    float hitResident = 0.0, hitEdge = 0.0;
    bool hit = false;
    for (int i = 0; i < 96; ++i) {
        vec3 pos = ro + rd * t;
        float r, e;
        float d = pos.y - terrain(pos.xz, r, e);
        if (d < 0.015) { hit = true; hitResident = r; hitEdge = e; break; }
        t += clamp(d * 0.42, 0.035, 0.42);
        if (t > 38.0) break;
    }
    vec3 color = sky;
    if (hit) {
        vec3 pos = ro + rd * t;
        float r0, e0, rx, ex, rz, ez;
        float h = terrain(pos.xz, r0, e0);
        float hx = terrain(pos.xz + vec2(0.05, 0), rx, ex);
        float hz = terrain(pos.xz + vec2(0, 0.05), rz, ez);
        vec3 n = normalize(vec3(h - hx, 0.05, h - hz));
        float light = 0.18 + 0.82 * max(dot(n, normalize(vec3(-0.5, 0.8, 0.35))), 0.0);
        vec3 loaded = mix(vec3(0.08, 0.20, 0.12), vec3(0.34, 0.48, 0.17), h * 0.7 + 0.5);
        vec3 missing = vec3(0.025, 0.035, 0.055);
        color = mix(missing, loaded * light, hitResident);
        color += vec3(0.08, 0.45, 0.8) * (1.0 - hitEdge) * hitResident * 0.45;
        color = mix(color, sky, smoothstep(16.0, 36.0, t));
    }
    outColor = vec4(pow(color, vec3(0.4545)), 1.0);
}
