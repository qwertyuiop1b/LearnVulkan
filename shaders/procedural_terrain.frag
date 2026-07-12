#version 450

layout(set = 0, binding = 0) uniform sampler2D terrainData;

layout(push_constant) uniform TerrainParams {
    float time;
    float aspect;
    float heightScale;
    float terrainScale;
} pc;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

vec4 sampleTerrain(vec2 worldXZ) {
    vec2 uv = worldXZ / pc.terrainScale + 0.5;
    return texture(terrainData, uv);
}

float terrainHeight(vec2 worldXZ) {
    return sampleTerrain(worldXZ).r * pc.heightScale;
}

void main() {
    vec2 screen = inUV * 2.0 - 1.0;
    screen.x *= pc.aspect;

    float orbit = pc.time * 0.075;
    vec3 camera = vec3(cos(orbit) * 19.0, 12.5, sin(orbit) * 19.0);
    vec3 target = vec3(0.0, 3.4, 0.0);
    vec3 forward = normalize(target - camera);
    vec3 right = normalize(cross(forward, vec3(0, 1, 0)));
    vec3 up = cross(right, forward);
    vec3 ray = normalize(forward + right * screen.x * 0.58 + up * screen.y * 0.58);

    bool hit = false;
    float distanceAlongRay = 0.1;
    float previousDistance = distanceAlongRay;
    vec3 position = camera;
    for (int i = 0; i < 180; ++i) {
        position = camera + ray * distanceAlongRay;
        vec2 terrainUV = position.xz / pc.terrainScale + 0.5;
        if (distanceAlongRay > 75.0 || any(lessThan(terrainUV, vec2(0.0))) || any(greaterThan(terrainUV, vec2(1.0))))
            break;
        float height = terrainHeight(position.xz);
        if (position.y <= height) {
            hit = true;
            float low = previousDistance;
            float high = distanceAlongRay;
            for (int refine = 0; refine < 7; ++refine) {
                float middle = (low + high) * 0.5;
                vec3 probe = camera + ray * middle;
                if (probe.y > terrainHeight(probe.xz))
                    low = middle;
                else
                    high = middle;
            }
            distanceAlongRay = high;
            position = camera + ray * distanceAlongRay;
            break;
        }
        previousDistance = distanceAlongRay;
        float clearance = max(position.y - height, 0.0);
        distanceAlongRay += clamp(clearance * 0.18, 0.12, 0.72);
    }

    vec3 skyTop = vec3(0.16, 0.32, 0.55);
    vec3 skyBottom = vec3(0.72, 0.82, 0.88);
    vec3 color = mix(skyBottom, skyTop, clamp(ray.y * 0.65 + 0.42, 0.0, 1.0));

    if (hit) {
        vec2 texelWorld = vec2(pc.terrainScale / float(textureSize(terrainData, 0).x));
        float centerHeight = terrainHeight(position.xz);
        float rightHeight = terrainHeight(position.xz + vec2(texelWorld.x, 0));
        float upHeight = terrainHeight(position.xz + vec2(0, texelWorld.x));
        vec3 normal = normalize(vec3(centerHeight - rightHeight, texelWorld.x, centerHeight - upHeight));
        vec4 terrain = sampleTerrain(position.xz);
        float normalizedHeight = centerHeight / pc.heightScale;
        float slope = 1.0 - normal.y;

        vec3 sand = vec3(0.48, 0.40, 0.25);
        vec3 grass = vec3(0.12, 0.31, 0.10);
        vec3 wetGrass = vec3(0.055, 0.20, 0.12);
        vec3 rock = vec3(0.30, 0.31, 0.30);
        vec3 snow = vec3(0.82, 0.88, 0.90);
        vec3 ground = mix(sand, mix(grass, wetGrass, terrain.g), smoothstep(0.16, 0.32, normalizedHeight));
        ground = mix(ground, rock, clamp(slope * 2.2 + terrain.b * 0.55, 0.0, 1.0));
        ground = mix(ground, snow, smoothstep(0.76, 0.91, normalizedHeight));

        vec3 lightDirection = normalize(vec3(-0.45, 0.82, -0.35));
        float diffuse = max(dot(normal, lightDirection), 0.0);
        float ambient = 0.24 + normal.y * 0.12;
        float fog = 1.0 - exp(-distanceAlongRay * distanceAlongRay * 0.0014);
        color = ground * (ambient + diffuse * 0.82);
        color = mix(color, skyBottom, clamp(fog, 0.0, 0.72));
    }

    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}
