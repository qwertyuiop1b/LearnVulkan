#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;

float rayleighPhase(float cosine) {
    return 0.0596831 * (1.0 + cosine * cosine);
}

float miePhase(float cosine, float g) {
    float gg = g * g;
    return 0.0795775 * (1.0 - gg) /
           max(pow(1.0 + gg - 2.0 * g * cosine, 1.5), 1e-4);
}

void main() {
    vec2 screen = inUV * 2.0 - 1.0;
    screen.x *= pc.p[0].x / max(pc.p[0].y, 1.0);
    vec3 rd = normalize(pc.p[2].xyz +
                        (pc.p[3].xyz * screen.x + pc.p[4].xyz * screen.y) * pc.p[2].w);
    vec3 sunDirection = normalize(pc.p[5].xyz);
    float sunIntensity = pc.p[5].w;
    float rayleighScale = pc.p[6].x;
    float mieScale = pc.p[6].y;
    float g = pc.p[6].z;
    float cameraHeight = pc.p[6].w;

    const vec3 betaRayleigh = vec3(0.34, 0.72, 1.35);
    const vec3 betaMie = vec3(0.72, 0.62, 0.52);
    const int STEP_COUNT = 36;
    float maxDistance = rd.y < -0.02 ? cameraHeight / max(-rd.y, 0.02) : 90.0;
    float stepLength = min(maxDistance, 90.0) / float(STEP_COUNT);
    vec3 transmittance = vec3(1.0);
    vec3 scattering = vec3(0.0);
    float cosine = dot(rd, sunDirection);
    float phaseR = rayleighPhase(cosine);
    float phaseM = miePhase(cosine, g);
    for (int stepIndex = 0; stepIndex < STEP_COUNT; ++stepIndex) {
        float distanceAlongRay = (float(stepIndex) + 0.5) * stepLength;
        float height = max(cameraHeight + rd.y * distanceAlongRay, 0.0);
        float densityR = exp(-height * 0.115) * rayleighScale;
        float densityM = exp(-height * 0.62) * mieScale;
        float sunOpticalDepth = max(0.0, 1.0 - sunDirection.y) * (densityR * 1.4 + densityM * 0.8);
        vec3 sunTransmittance = exp(-(betaRayleigh * densityR + betaMie * densityM) * sunOpticalDepth);
        vec3 localScatter = betaRayleigh * densityR * phaseR + betaMie * densityM * phaseM;
        scattering += transmittance * sunTransmittance * localScatter * stepLength * sunIntensity;
        transmittance *= exp(-(betaRayleigh * densityR + betaMie * densityM) * stepLength * 0.11);
    }

    vec3 color = scattering;
    if (rd.y < 0.0) {
        float groundFade = exp(rd.y * 18.0);
        vec3 ground = vec3(0.045, 0.055, 0.045) * (0.35 + max(sunDirection.y, 0.0));
        color = mix(ground, color, groundFade);
    }
    float sunDisk = smoothstep(0.9992, 0.99985, cosine);
    color += transmittance * vec3(1.0, 0.72, 0.38) * sunDisk * sunIntensity * 2.2;
    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
