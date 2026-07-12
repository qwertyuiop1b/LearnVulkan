#version 450
layout(location=0) in vec2 inUV; layout(location=0) out vec4 outColor;
layout(set=0,binding=0,std430) readonly buffer Data { vec4 values[]; } data;
void main(){ uint i=min(uint(inUV.x*1024.),1023u); vec4 v=data.values[i]; float vis=1.-smoothstep(.008,.018,abs(inUV.y-.2-v.x*.25)); float lod=1.-smoothstep(.008,.018,abs(inUV.y-.48-v.y*.09)); float bucket=1.-smoothstep(.008,.018,abs(inUV.y-.76-v.z*.02)); outColor=vec4(vec3(.008,.014,.028)+vis*vec3(.1,1,.35)+lod*vec3(1,.45,.06)+bucket*vec3(.2,.35,1),1); }
