#version 450

layout(location=0) in vec2 vUv;
layout(location=0) out vec4 outColor;
struct Entity { vec4 positionRadius; vec4 colorType; };
layout(set=0,binding=0,std430) readonly buffer Entities { Entity entities[]; };
layout(push_constant) uniform Push { vec4 p[8]; } pc;

float terrain(vec2 p) { return sin(p.x*0.42)*cos(p.y*0.35)*0.32 - 0.05; }
float hash21(vec2 p) { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }

void main() {
    vec2 q=vUv*2.0-1.0; q.x*=pc.p[0].x/pc.p[0].y;
    vec3 ro=pc.p[2].xyz;
    vec3 rd=normalize(pc.p[3].xyz+q.x*pc.p[4].xyz*pc.p[3].w+q.y*pc.p[5].xyz*pc.p[3].w);
    vec3 sky=mix(vec3(0.025,0.05,0.10),vec3(0.30,0.46,0.62),max(rd.y,0.0));
    float best=1e5; vec3 objectColor=vec3(0); bool entityHit=false;
    int activeCount=min(int(pc.p[0].w)+1,int(pc.p[1].z));
    for(int i=0;i<activeCount;++i){
        vec3 oc=ro-entities[i].positionRadius.xyz;
        float b=dot(oc,rd), c=dot(oc,oc)-entities[i].positionRadius.w*entities[i].positionRadius.w;
        float h=b*b-c;
        if(h>0.0){ float t=-b-sqrt(h); if(t>0.0&&t<best){best=t;objectColor=entities[i].colorType.rgb;entityHit=true;} }
    }
    float terrainT=0.2; bool groundHit=false;
    for(int i=0;i<80;++i){
        vec3 pos=ro+rd*terrainT; float d=pos.y-terrain(pos.xz);
        if(d<0.012){groundHit=true;break;}
        terrainT+=clamp(d*0.45,0.03,0.45); if(terrainT>35.0)break;
    }
    vec3 color=sky;
    if(groundHit&&terrainT<best){
        vec3 p=ro+rd*terrainT; float h=terrain(p.xz);
        vec3 n=normalize(vec3(h-terrain(p.xz+vec2(.04,0)),.04,h-terrain(p.xz+vec2(0,.04))));
        float l=.18+.82*max(dot(n,normalize(vec3(-.5,.8,.3))),0.0);
        color=mix(vec3(.04,.12,.07),vec3(.22,.34,.09),h+.5)*l;
        color=mix(color,sky,smoothstep(18.0,35.0,terrainT));
    } else if(entityHit) {
        vec3 p=ro+rd*best; vec3 center=vec3(0); float radius=1.0;
        for(int i=0;i<activeCount;++i){float d=distance(p,entities[i].positionRadius.xyz)-entities[i].positionRadius.w;if(abs(d)<.035){center=entities[i].positionRadius.xyz;radius=entities[i].positionRadius.w;break;}}
        vec3 n=normalize(p-center); float l=.2+.8*max(dot(n,normalize(vec3(-.4,.8,.5))),0.0);
        color=objectColor*l+pow(max(dot(reflect(-normalize(vec3(-.4,.8,.5)),n),-rd),0.0),24.0)*.5;
    }
    float weather=pc.p[1].y;
    vec2 rainUv=vec2(vUv.x*110.0+pc.p[0].z*1.7,vUv.y*45.0-pc.p[0].z*12.0);
    float rain=step(0.965,hash21(floor(rainUv)))*smoothstep(.5,0.0,fract(rainUv.y));
    color+=vec3(.35,.65,1.0)*rain*weather*.75;
    color=mix(color,vec3(dot(color,vec3(.333))),weather*.18);
    outColor=vec4(pow(color,vec3(.4545)),1);
}
