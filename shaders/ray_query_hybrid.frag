#version 450
layout(location=0) in vec2 inUV;layout(location=0)out vec4 outColor;
layout(set=0,binding=0,std430)readonly buffer Data{vec4 values[];}data;
void main(){uint i=min(uint(inUV.x*1024.),1023u);vec4 v=data.values[i];float a=1.-smoothstep(.008,.018,abs(inUV.y-.22-v.x*.22));float b=1.-smoothstep(.008,.018,abs(inUV.y-.5-v.y*.22));float c=1.-smoothstep(.008,.018,abs(inUV.y-.78-v.z*.2));outColor=vec4(vec3(.006,.012,.026)+a*vec3(1,.3,.06)+b*vec3(.15,.8,1)+c*vec3(.35,1,.25),1);}
