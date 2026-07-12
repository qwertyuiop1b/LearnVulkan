#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(std430, set = 0, binding = 0) readonly buffer LutBuffer { vec4 colors[]; } lut;
layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;

vec3 fetchLut(ivec3 coordinate, int size) {
    coordinate = clamp(coordinate, ivec3(0), ivec3(size - 1));
    int index = coordinate.x + coordinate.y * size + coordinate.z * size * size;
    return lut.colors[index].rgb;
}

vec3 sampleLut(vec3 color, int size) {
    vec3 position = clamp(color, 0.0, 1.0) * float(size - 1);
    ivec3 base = ivec3(floor(position));
    vec3 fraction = fract(position);
    vec3 c00 = mix(fetchLut(base + ivec3(0, 0, 0), size), fetchLut(base + ivec3(1, 0, 0), size), fraction.x);
    vec3 c10 = mix(fetchLut(base + ivec3(0, 1, 0), size), fetchLut(base + ivec3(1, 1, 0), size), fraction.x);
    vec3 c01 = mix(fetchLut(base + ivec3(0, 0, 1), size), fetchLut(base + ivec3(1, 0, 1), size), fraction.x);
    vec3 c11 = mix(fetchLut(base + ivec3(0, 1, 1), size), fetchLut(base + ivec3(1, 1, 1), size), fraction.x);
    return mix(mix(c00, c10, fraction.y), mix(c01, c11, fraction.y), fraction.z);
}

vec3 sourceImage(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;
    float angle = atan(p.y, p.x);
    float radius = length(p);
    vec3 wheel = 0.52 + 0.48 * cos(angle + vec3(0.0, 4.18879, 2.09439));
    vec3 background = mix(vec3(0.025, 0.035, 0.055), vec3(0.46, 0.58, 0.72), uv.y);
    float disk = 1.0 - smoothstep(0.52, 0.535, radius);
    vec3 color = mix(background, wheel * (1.15 - radius * 0.55), disk);
    float bars = step(0.5, fract(uv.x * 8.0));
    float lower = 1.0 - smoothstep(0.0, 0.02, abs(uv.y - 0.14));
    color = mix(color, mix(vec3(0.05), vec3(0.95), bars), lower * 0.75);
    return color;
}

void main() {
    vec3 source = sourceImage(inUV);
    int size = int(pc.p[0].x + 0.5);
    vec3 graded = sampleLut(source / (source + vec3(1.0)), size);
    vec3 original = source / (source + vec3(1.0));
    vec3 color = mix(original, graded, pc.p[0].z);
    float divider = 1.0 - smoothstep(0.0, 0.004, abs(inUV.x - 0.5));
    color = inUV.x < 0.5 ? original : color;
    color = mix(color, vec3(0.92), divider);
    outColor = vec4(color, 1.0);
}
