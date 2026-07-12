#version 450

layout(push_constant) uniform CavePush {
    vec4 cameraPosition;
    vec4 cameraForward;
    vec4 cameraRight;
    vec4 cameraUp;
    vec4 viewport;    // aspect, tan(fov / 2), time, surface detail
    vec4 destruction; // world-space center, radius
    vec4 effects;     // blast strength, pulse age, fog density, crystal glow
} pc;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

const float FAR_DISTANCE = 52.0;
const float SURFACE_EPSILON = 0.0025;
const int MAX_STEPS = 156;

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float valueNoise(vec3 p) {
    vec3 cell = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(cell + vec3(0, 0, 0));
    float n100 = hash13(cell + vec3(1, 0, 0));
    float n010 = hash13(cell + vec3(0, 1, 0));
    float n110 = hash13(cell + vec3(1, 1, 0));
    float n001 = hash13(cell + vec3(0, 0, 1));
    float n101 = hash13(cell + vec3(1, 0, 1));
    float n011 = hash13(cell + vec3(0, 1, 1));
    float n111 = hash13(cell + vec3(1, 1, 1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

float fbm(vec3 p) {
    float sum = 0.0;
    float amplitude = 0.52;
    mat3 rotation = mat3(0.00, 0.80, 0.60,
                         -0.80, 0.36, -0.48,
                         -0.60, -0.48, 0.64);
    for (int octave = 0; octave < 4; ++octave) {
        sum += amplitude * valueNoise(p);
        p = rotation * p * 2.03 + vec3(7.1, 13.7, 3.4);
        amplitude *= 0.49;
    }
    return sum;
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float radius) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - radius;
}

float sdCappedCone(vec3 p, float height, float lowerRadius, float upperRadius) {
    vec2 q = vec2(length(p.xz), p.y);
    vec2 k1 = vec2(upperRadius, height);
    vec2 k2 = vec2(upperRadius - lowerRadius, 2.0 * height);
    vec2 ca = vec2(q.x - min(q.x, q.y < 0.0 ? lowerRadius : upperRadius), abs(q.y) - height);
    vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    float signValue = cb.x < 0.0 && ca.y < 0.0 ? -1.0 : 1.0;
    return signValue * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

float sdOctahedron(vec3 p, float size) {
    p = abs(p);
    float m = p.x + p.y + p.z - size;
    vec3 q;
    if (3.0 * p.x < m) {
        q = p.xyz;
    } else if (3.0 * p.y < m) {
        q = p.yzx;
    } else if (3.0 * p.z < m) {
        q = p.zxy;
    } else {
        return m * 0.57735027;
    }
    float k = clamp(0.5 * (q.z - q.y + size), 0.0, size);
    return length(vec3(q.x, q.y - size + k, q.z - k));
}

mat2 rotate2d(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, -s, s, c);
}

// Hash-free geometry noise keeps the inner ray-march loop affordable on tile GPUs.
float surfaceNoise(vec3 p) {
    float broad = sin(p.x * 1.17 + sin(p.z * 0.71)) *
                  sin(p.y * 1.31 - p.z * 0.43);
    float fine = sin(dot(p, vec3(2.73, -1.91, 2.17))) * 0.5 +
                 sin(dot(p, vec3(-3.11, 2.37, 1.43))) * 0.5;
    return broad * 0.66 + fine * 0.34;
}

// Positive values are navigable air. Zero is a surface, negative is solid rock.
vec2 caveMap(vec3 p) {
    float time = pc.viewport.z;
    float detail = pc.viewport.w;

    vec2 windingCenter = vec2(sin(p.z * 0.17) * 0.95 + sin(p.z * 0.43) * 0.18,
                              cos(p.z * 0.13) * 0.34);
    vec2 tunnelPoint = p.xy - windingCenter;
    tunnelPoint.x *= 0.82;
    float tunnelRadius = 4.25 + 0.42 * sin(p.z * 0.31) + 0.20 * sin(p.z * 0.83);
    float tunnelAir = tunnelRadius - length(tunnelPoint);

    vec3 chamberPoint = p - vec3(0.4, -0.25, -3.0);
    chamberPoint *= vec3(0.76, 0.92, 0.62);
    float chamberAir = 6.15 - length(chamberPoint);
    float air = max(tunnelAir, chamberAir);
    float material = 1.0;

    // A branching passage makes the silhouette read as a volume instead of a tube.
    vec3 branch = p - vec3(-1.0, 0.25, -5.4);
    branch.xz = rotate2d(0.78) * branch.xz;
    float branchAir = 2.05 - length(branch.yz * vec2(1.0, 0.82));
    if (branchAir > air) {
        air = branchAir;
        material = 2.0;
    }

    // GPU CSG destruction: union a fractured spherical void with the cave air.
    vec3 blastPoint = p - pc.destruction.xyz;
    float fracture = surfaceNoise(blastPoint * 2.1 + 5.0) * 0.23 * pc.effects.x;
    float blastAir = pc.destruction.w - length(blastPoint) + fracture;
    if (blastAir > air) {
        air = blastAir;
        material = 5.0;
    }

    // Rock detail is applied after the broad CSG so it remains stable while editing.
    float rockNoise = surfaceNoise(p * 0.72 + vec3(2.0, 7.0, 13.0));
    air -= rockNoise * mix(0.04, 0.24, detail);

    // Solid formations use ordinary outside-positive SDFs and are intersected with the air volume.
    float pillar = sdCapsule(p, vec3(-2.45, -3.1, -4.4), vec3(-2.15, 2.75, -4.15), 0.44);
    pillar += surfaceNoise(p * 2.2) * 0.055;
    if (pillar < air) {
        air = pillar;
        material = 3.0;
    }

    vec3 stalactitePoint = p - vec3(2.05, 2.28, -2.45);
    stalactitePoint.y *= -1.0;
    float stalactite = sdCappedCone(stalactitePoint, 1.65, 0.13, 0.62);
    if (stalactite < air) {
        air = stalactite;
        material = 3.0;
    }

    vec3 stalagmitePoint = p - vec3(0.35, -2.82, -5.75);
    float stalagmite = sdCappedCone(stalagmitePoint, 1.18, 0.72, 0.08);
    if (stalagmite < air) {
        air = stalagmite;
        material = 3.0;
    }

    // Faceted emissive crystals are actual SDF geometry, not a screen-space overlay.
    vec3 crystalPoint = p - vec3(1.72, -2.48, -4.72);
    crystalPoint.xz = rotate2d(0.36) * crystalPoint.xz;
    float crystal = sdOctahedron(crystalPoint * vec3(1.65, 0.62, 1.65), 0.72) * 0.72;
    if (crystal < air) {
        air = crystal;
        material = 4.0;
    }

    vec3 crystalPoint2 = p - vec3(2.34, -2.72, -4.22);
    crystalPoint2.xz = rotate2d(-0.28) * crystalPoint2.xz;
    float crystal2 = sdOctahedron(crystalPoint2 * vec3(1.9, 0.72, 1.9), 0.52) * 0.65;
    if (crystal2 < air) {
        air = crystal2;
        material = 4.0;
    }

    // A subtle vibration around the fresh fracture sells the impact without moving all geometry.
    float shockRadius = pc.effects.y * 7.5;
    float shockBand = exp(-abs(length(blastPoint) - shockRadius) * 5.0) * exp(-pc.effects.y * 1.8);
    air += shockBand * 0.035 * pc.effects.x * sin(time * 36.0);
    return vec2(air, material);
}

vec3 surfaceNormal(vec3 p) {
    const vec2 e = vec2(SURFACE_EPSILON * 2.0, 0.0);
    return normalize(vec3(caveMap(p + e.xyy).x - caveMap(p - e.xyy).x,
                          caveMap(p + e.yxy).x - caveMap(p - e.yxy).x,
                          caveMap(p + e.yyx).x - caveMap(p - e.yyx).x));
}

float ambientOcclusion(vec3 p, vec3 normal) {
    float occlusion = 0.0;
    float weight = 1.0;
    for (int sampleIndex = 1; sampleIndex <= 5; ++sampleIndex) {
        float distanceAlongNormal = 0.075 * float(sampleIndex);
        float sampledDistance = caveMap(p + normal * distanceAlongNormal).x;
        occlusion += max(distanceAlongNormal - sampledDistance, 0.0) * weight;
        weight *= 0.63;
    }
    return clamp(1.0 - occlusion * 1.7, 0.16, 1.0);
}

float softShadow(vec3 origin, vec3 direction, float maxDistance) {
    float shadow = 1.0;
    float travel = 0.035;
    for (int stepIndex = 0; stepIndex < 42 && travel < maxDistance; ++stepIndex) {
        float distanceToSurface = caveMap(origin + direction * travel).x;
        if (distanceToSurface < SURFACE_EPSILON) {
            return 0.0;
        }
        shadow = min(shadow, 13.0 * distanceToSurface / travel);
        travel += clamp(distanceToSurface * 0.72, 0.025, 0.55);
    }
    return clamp(shadow, 0.0, 1.0);
}

vec3 rockAlbedo(vec3 p, vec3 normal, float material) {
    float largePattern = fbm(p * 0.38 + normal * 0.45);
    float finePattern = valueNoise(p * 4.2);
    vec3 warmRock = vec3(0.285, 0.205, 0.145);
    vec3 coldRock = vec3(0.105, 0.145, 0.158);
    vec3 base = mix(coldRock, warmRock, smoothstep(0.30, 0.78, largePattern));
    base *= mix(0.68, 1.28, finePattern);
    float mineralVein = smoothstep(0.76, 0.88, fbm(p * 1.75 + vec3(8.0)));
    base = mix(base, vec3(0.42, 0.31, 0.20), mineralVein * 0.48);
    if (material == 3.0) {
        base *= vec3(0.82, 0.90, 0.88);
    } else if (material == 5.0) {
        base = mix(base, vec3(0.42, 0.16, 0.065), 0.48);
    }
    return base;
}

vec3 skyColor(vec3 rayDirection) {
    float vertical = clamp(rayDirection.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3(0.008, 0.012, 0.019), vec3(0.018, 0.030, 0.044), vertical);
}

void main() {
    vec2 screen = vec2(inUV.x * 2.0 - 1.0, 1.0 - inUV.y * 2.0);
    screen.x *= pc.viewport.x;
    vec3 rayOrigin = pc.cameraPosition.xyz;
    vec3 rayDirection = normalize(pc.cameraForward.xyz +
                                  pc.cameraRight.xyz * screen.x * pc.viewport.y +
                                  pc.cameraUp.xyz * screen.y * pc.viewport.y);

    float travel = 0.02;
    float material = 0.0;
    bool hit = false;
    float glowIntegral = 0.0;
    float blastFog = 0.0;
    for (int stepIndex = 0; stepIndex < MAX_STEPS; ++stepIndex) {
        vec3 position = rayOrigin + rayDirection * travel;
        vec2 scene = caveMap(position);
        float distanceToSurface = scene.x;
        float crystalDistance = length(position - vec3(1.9, -2.45, -4.55));
        glowIntegral += exp(-crystalDistance * 1.45) * 0.014;

        float shockRadius = pc.effects.y * 7.5;
        float blastDistance = length(position - pc.destruction.xyz);
        blastFog += exp(-abs(blastDistance - shockRadius) * 2.8) * exp(-pc.effects.y * 1.55) * 0.018;

        if (distanceToSurface < SURFACE_EPSILON * max(1.0, travel * 0.08)) {
            hit = true;
            material = scene.y;
            break;
        }
        travel += clamp(distanceToSurface * 0.68, 0.018, 0.72);
        if (travel > FAR_DISTANCE) {
            break;
        }
    }

    vec3 color = skyColor(rayDirection);
    if (hit) {
        vec3 position = rayOrigin + rayDirection * travel;
        vec3 normal = surfaceNormal(position);
        vec3 viewDirection = -rayDirection;
        vec3 keyLightPosition = vec3(-1.8, 2.3, 1.8);
        vec3 toLight = keyLightPosition - position;
        float lightDistance = length(toLight);
        vec3 lightDirection = toLight / max(lightDistance, 0.001);
        float attenuation = 1.0 / (1.0 + 0.055 * lightDistance * lightDistance);
        float diffuse = max(dot(normal, lightDirection), 0.0);
        float shadow = softShadow(position + normal * 0.018, lightDirection, lightDistance);
        float ao = ambientOcclusion(position, normal);
        float rim = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.0);
        vec3 halfVector = normalize(lightDirection + viewDirection);
        float specular = pow(max(dot(normal, halfVector), 0.0), material == 4.0 ? 72.0 : 24.0);

        if (material == 4.0) {
            float crystalBands = 0.62 + 0.38 * sin(position.y * 16.0 + position.x * 7.0);
            vec3 crystal = mix(vec3(0.03, 0.22, 0.36), vec3(0.12, 0.90, 1.22), crystalBands);
            color = crystal * (0.75 + pc.effects.w * 1.5) + specular * vec3(0.8, 1.0, 1.0) * 2.2;
            color += rim * vec3(0.06, 0.68, 1.0) * 1.4;
        } else {
            vec3 albedo = rockAlbedo(position, normal, material);
            vec3 coolAmbient = vec3(0.075, 0.115, 0.145) * (0.38 + 0.62 * max(normal.y, 0.0));
            vec3 warmDirect = vec3(1.15, 0.73, 0.40) * diffuse * shadow * attenuation * 3.2;
            color = albedo * (coolAmbient + warmDirect) * ao;
            color += specular * shadow * attenuation * vec3(1.0, 0.72, 0.42) * 0.65;
            color += rim * vec3(0.018, 0.055, 0.078) * ao;

            if (material == 5.0) {
                float fractureHeat = exp(-pc.effects.y * 1.9) * pc.effects.x;
                float cracks = smoothstep(0.70, 0.92, fbm(position * 5.1 + vec3(pc.viewport.z * 0.8)));
                color += vec3(1.35, 0.19, 0.025) * cracks * fractureHeat * 2.6;
            }
        }

        float distanceFog = 1.0 - exp(-travel * travel * pc.effects.z * 0.022);
        color = mix(color, vec3(0.018, 0.034, 0.046), clamp(distanceFog, 0.0, 0.86));
    }

    color += glowIntegral * pc.effects.w * vec3(0.03, 0.48, 0.92);
    color += blastFog * pc.effects.x * vec3(1.20, 0.16, 0.018);

    // Filmic compression preserves emissive crystal and fracture highlights on an SDR swapchain.
    color = max(color, vec3(0.0));
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
