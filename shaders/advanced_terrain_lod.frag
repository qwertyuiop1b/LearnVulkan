#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform AdvancedPush { vec4 p[8]; } pc;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x),
               mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1, 1)), f.x), f.y) * 2.0 - 1.0;
}

float terrainHeight(vec2 p, float distanceToCamera) {
    float lodBias = pc.p[5].y;
    float lod = clamp(floor(log2(max(distanceToCamera * 0.12, 1.0)) + lodBias), 0.0, 6.0);
    float cell = exp2(lod) * 0.045;
    vec2 snapped = floor(p / cell) * cell;
    vec2 blend = smoothstep(0.0, 1.0, fract(p / cell));
    vec2 samplePosition = mix(snapped, snapped + cell, blend);
    float height = 0.0;
    float amplitude = 1.0;
    float frequency = 0.055;
    for (int octave = 0; octave < 6; ++octave) {
        if (float(octave) > 5.0 - lod * 0.65)
            break;
        height += noise2(samplePosition * frequency + float(octave) * 17.0) * amplitude;
        frequency *= 2.03;
        amplitude *= 0.48;
    }
    return height * pc.p[5].x;
}

float traceTerrain(vec3 ro, vec3 rd, out vec3 position) {
    float distanceAlongRay = 0.2;
    float previousDistance = distanceAlongRay;
    for (int stepIndex = 0; stepIndex < 128; ++stepIndex) {
        position = ro + rd * distanceAlongRay;
        float height = terrainHeight(position.xz, distanceAlongRay);
        if (position.y <= height) {
            float low = previousDistance;
            float high = distanceAlongRay;
            for (int refine = 0; refine < 7; ++refine) {
                float middle = (low + high) * 0.5;
                vec3 samplePosition = ro + rd * middle;
                if (samplePosition.y > terrainHeight(samplePosition.xz, middle))
                    low = middle;
                else
                    high = middle;
            }
            position = ro + rd * high;
            return high;
        }
        previousDistance = distanceAlongRay;
        distanceAlongRay += clamp((position.y - height) * 0.32, 0.12, 2.6);
        if (distanceAlongRay > 180.0)
            break;
    }
    return 1e20;
}

void main() {
    vec2 screen = inUV * 2.0 - 1.0;
    screen.x *= pc.p[0].x / max(pc.p[0].y, 1.0);
    vec3 ro = pc.p[1].xyz;
    vec3 rd = normalize(pc.p[2].xyz +
                        (pc.p[3].xyz * screen.x + pc.p[4].xyz * screen.y) * pc.p[2].w);
    vec3 position;
    float distanceAlongRay = traceTerrain(ro, rd, position);
    if (distanceAlongRay > 1e19) {
        vec3 sky = mix(vec3(0.26, 0.43, 0.66), vec3(0.62, 0.76, 0.86), max(rd.y, 0.0));
        outColor = vec4(sky, 1.0);
        return;
    }
    float epsilon = max(0.08, distanceAlongRay * 0.003);
    float centerHeight = terrainHeight(position.xz, distanceAlongRay);
    vec3 normal = normalize(vec3(centerHeight - terrainHeight(position.xz + vec2(epsilon, 0), distanceAlongRay),
                                 epsilon,
                                 centerHeight - terrainHeight(position.xz + vec2(0, epsilon), distanceAlongRay)));
    float normalizedHeight = centerHeight / max(pc.p[5].x, 0.01) * 0.5 + 0.5;
    float slope = 1.0 - normal.y;
    vec3 grass = vec3(0.08, 0.25, 0.055);
    vec3 rock = vec3(0.30, 0.31, 0.30);
    vec3 snow = vec3(0.78, 0.84, 0.87);
    vec3 color = mix(grass, rock, smoothstep(0.25, 0.62, slope));
    color = mix(color, snow, smoothstep(0.72, 0.92, normalizedHeight));
    float diffuse = max(dot(normal, normalize(vec3(-0.45, 0.82, -0.35))), 0.0);
    color *= 0.20 + diffuse * 0.92;
    if (pc.p[5].z > 0.5) {
        float lod = clamp(floor(log2(max(distanceAlongRay * 0.12, 1.0)) + pc.p[5].y), 0.0, 6.0);
        float cell = exp2(lod) * 0.72;
        vec2 grid = abs(fract(position.xz / cell) - 0.5);
        float line = 1.0 - smoothstep(0.44, 0.49, max(grid.x, grid.y));
        color = mix(color, vec3(0.05, 0.65, 0.92), line * 0.38);
    }
    float fog = 1.0 - exp(-distanceAlongRay * 0.012);
    color = mix(color, vec3(0.48, 0.63, 0.76), clamp(fog, 0.0, 0.82));
    outColor = vec4(color, 1.0);
}
