#version 450

layout(location = 0) in vec2 vLocal;
layout(location = 1) in vec3 vColor;
layout(location = 0) out vec4 outColor;

void main() {
    float r2 = dot(vLocal, vLocal);
    float alpha = 1.0 - smoothstep(0.82, 1.0, r2);
    float nz = sqrt(max(1.0 - r2, 0.0));
    vec3 n = normalize(vec3(vLocal, nz));
    float light = 0.18 + 0.82 * max(dot(n, normalize(vec3(-0.4, 0.7, 0.6))), 0.0);
    outColor = vec4(vColor * light + pow(nz, 18.0) * 0.45, alpha);
}
