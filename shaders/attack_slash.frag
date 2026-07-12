#version 450

layout(push_constant) uniform AttackPush {
    float time;
    float unusedX;
    float unusedY;
    float viewportWidth;
    float viewportHeight;
} pc;

layout(location = 0) in vec2 ribbonUv;
layout(location = 1) in vec3 worldPosition;
layout(location = 2) in vec3 worldNormal;
layout(location = 3) flat in uint ribbonId;
layout(location = 0) out vec4 outColor;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    float cycle = fract(pc.time / 3.0);
    float reveal = smoothstep(0.025, 0.43, cycle);
    float fade = 1.0 - smoothstep(0.76, 0.98, cycle);

    // A narrow moving head makes the slash cut through space instead of fading in uniformly.
    float head = 1.0 - smoothstep(reveal - 0.035, reveal + 0.012, ribbonUv.x);
    float tail = smoothstep(0.0, 0.055, ribbonUv.x);
    float life = head * tail * fade;
    if (life < 0.002)
        discard;

    float across = abs(ribbonUv.y * 2.0 - 1.0);
    float edge = smoothstep(1.0, 0.58, across);
    float hotCore = exp(-across * across * 13.0);
    float striation = 0.78 + 0.22 * sin(ribbonUv.x * 115.0 + across * 9.0);

    vec3 viewDirection = normalize(vec3(0.0, 0.0, 3.35) - worldPosition);
    vec3 lightDirection = normalize(vec3(-0.35, 0.8, 0.55));
    vec3 normal = normalize(worldNormal);
    if (!gl_FrontFacing)
        normal = -normal;
    float diffuse = 0.25 + 0.75 * max(dot(normal, lightDirection), 0.0);
    float fresnel = pow(1.0 - abs(dot(normal, viewDirection)), 2.2);
    vec3 halfVector = normalize(lightDirection + viewDirection);
    float specular = pow(max(dot(normal, halfVector), 0.0), 46.0);

    vec3 outerColor;
    vec3 coreColor;
    if (ribbonId == 0u) {
        outerColor = vec3(1.0, 0.075, 0.018);
        coreColor = vec3(1.0, 0.88, 0.16);
    } else {
        outerColor = vec3(0.24, 0.035, 1.0);
        coreColor = vec3(0.38, 0.84, 1.0);
    }

    vec3 material = mix(outerColor, coreColor, hotCore);
    material *= (0.58 + diffuse * 0.72) * striation;
    material += coreColor * specular * 1.6;
    material += mix(outerColor, coreColor, 0.55) * fresnel * 0.75;

    // Jagged erosion gives the broad tail a torn, forceful silhouette.
    float noise = hash21(floor(vec2(ribbonUv.x * 96.0, ribbonUv.y * 18.0)));
    float tornTail = smoothstep(0.0, 0.18, ribbonUv.x + noise * 0.10);
    float alpha = life * edge * tornTail;
    alpha *= 0.84 + hotCore * 0.16;

    outColor = vec4(clamp(material, 0.0, 1.0), alpha);
}
