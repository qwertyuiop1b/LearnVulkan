#version 450
layout(location=0) in vec2 inUV; layout(location=0) out vec4 outColor;
layout(set=0,binding=0,std430) readonly buffer Data { vec4 values[]; } data;
void main(){ uint i=min(uint(inUV.x*1024.),1023u); vec4 v=data.values[i]; float a=1.-smoothstep(.008,.018,abs(inUV.y-.22-v.x*.26)); float b=1.-smoothstep(.008,.018,abs(inUV.y-.5-v.y*.14)); float c=1.-smoothstep(.008,.018,abs(inUV.y-.78-v.z*.12)); outColor=vec4(vec3(.008,.014,.026)+a*vec3(.1,1,.3)+b*vec3(1,.35,.05)+c*vec3(.25,.5,1),1); }
