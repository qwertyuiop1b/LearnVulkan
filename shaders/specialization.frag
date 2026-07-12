#version 450

layout(constant_id = 0) const uint COLOR_MODE = 0;
layout(constant_id = 1) const float STRIPE_FREQUENCY = 8.0;

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    if (COLOR_MODE == 0) {
        outColor = vec4(fragColor, 1.0);
    } else if (COLOR_MODE == 1) {
        outColor = vec4(vec3(dot(fragColor, vec3(0.2126, 0.7152, 0.0722))), 1.0);
    } else {
        float stripe = step(0.5, fract(gl_FragCoord.x / STRIPE_FREQUENCY));
        vec3 a = fragColor;
        vec3 b = vec3(1.0) - fragColor * 0.45;
        outColor = vec4(mix(a, b, stripe), 1.0);
    }
}
