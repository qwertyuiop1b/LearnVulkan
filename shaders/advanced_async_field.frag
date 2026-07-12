#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(std430, set = 0, binding = 0) readonly buffer FieldBuffer { vec4 values[]; } field;
layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;

vec4 sampleField(vec2 uv) {
    uvec2 size = uvec2(pc.p[1].xy + 0.5);
    uvec2 pixel = min(uvec2(clamp(uv, 0.0, 0.9999) * vec2(size)), size - uvec2(1));
    return field.values[pixel.x + pixel.y * size.x];
}

void main() {
    vec2 texel = 1.0 / pc.p[1].xy;
    vec4 center = sampleField(inUV);
    float dx = sampleField(inUV + vec2(texel.x, 0.0)).a - sampleField(inUV - vec2(texel.x, 0.0)).a;
    float dy = sampleField(inUV + vec2(0.0, texel.y)).a - sampleField(inUV - vec2(0.0, texel.y)).a;
    vec3 normal = normalize(vec3(-dx * 2.8, -dy * 2.8, 1.0));
    float lighting = 0.45 + 0.55 * max(dot(normal, normalize(vec3(-0.35, 0.55, 0.76))), 0.0);
    vec3 color = center.rgb * lighting;
    color += pow(max(center.a - 0.35, 0.0), 2.0) * vec3(0.22, 0.62, 1.0);
    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
