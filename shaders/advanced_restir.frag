#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
struct Reservoir { vec4 sampleWeight; vec4 state; };
layout(std430, set = 0, binding = 0) readonly buffer ReservoirA { Reservoir dataA[]; } a;
layout(std430, set = 0, binding = 1) readonly buffer ReservoirB { Reservoir dataB[]; } b;
layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;

vec3 lightColor(uint index) {
    return 0.42 + 0.58 * cos(float(index) * 1.37 + vec3(0.0, 2.1, 4.2));
}

void main() {
    uvec2 size = uvec2(pc.p[1].xy + 0.5);
    uvec2 pixel = min(uvec2(inUV * vec2(size)), size - uvec2(1));
    uint index = pixel.x + pixel.y * size.x;
    uint parity = uint(pc.p[0].w + 0.5) & 1u;
    Reservoir reservoir = parity == 0u ? a.dataA[index] : b.dataB[index];
    uint lightIndex = uint(reservoir.sampleWeight.w + 0.5);
    vec2 delta = reservoir.sampleWeight.xy - inUV;
    float distanceSquared = dot(delta, delta);
    vec2 p = inUV * 2.0 - 1.0;
    float surface = sin(p.x * 8.0) * cos(p.y * 7.0) * 0.12;
    vec3 normal = normalize(vec3(-cos(p.x * 8.0) * 0.6, sin(p.y * 7.0) * 0.52, 1.0));
    vec3 lightDirection = normalize(vec3(delta * 3.5, 0.32 + surface));
    float diffuse = max(dot(normal, lightDirection), 0.0);
    float attenuation = 1.0 / (0.03 + distanceSquared * 12.0);
    vec3 albedo = mix(vec3(0.09, 0.12, 0.16), vec3(0.42, 0.38, 0.31), inUV.y);
    vec3 color = albedo * (0.06 + lightColor(lightIndex) * diffuse * attenuation * 0.34);
    float lightMarker = 1.0 - smoothstep(0.005, 0.014, sqrt(distanceSquared));
    color += lightColor(lightIndex) * lightMarker * 1.8;
    if (pc.p[2].z > 0.5) {
        float age = reservoir.state.z / 32.0;
        color = mix(color, vec3(fract(float(lightIndex) * 0.618), clamp(age, 0.0, 1.0), 1.0 - clamp(age, 0.0, 1.0)), 0.72);
    }
    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
