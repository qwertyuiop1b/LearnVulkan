#version 450
// TAA 场景片元：简单 Lambert 光照（高对比边缘便于观察抗锯齿）

layout(location = 0) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 n = normalize(fragNormal);
    vec3 lightDir = normalize(vec3(0.35, 1.0, 0.45));
    float diff = max(dot(n, lightDir), 0.0);
    vec3 base = vec3(0.15, 0.55, 0.95);
    vec3 lit = base * (0.2 + 0.8 * diff);
    outColor = vec4(lit, 1.0);
}
